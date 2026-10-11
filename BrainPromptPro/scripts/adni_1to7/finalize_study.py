"""Finite continuation: wait for the selected subject check, freeze, evaluate TEST.

The primary candidate is already selected from the completed main validation
matrix. Neither supplemental nor TEST scores can change that decision. A new
selection lock/output is mandatory; failures stop further dispatch.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--gpus',default='0,3,6');a=p.parse_args()
    root=a.root.resolve();code=Path(__file__).resolve().parents[3];q=root/'queues/phase8_frozen_test';q.mkdir(parents=True,exist_ok=True)
    lockstream=(q/'controller.lock').open('a');fcntl.flock(lockstream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (q/'state.private.json').exists() or (root/'selection_lock.json').exists():raise FileExistsError('Existing finalization needs a reviewed recovery, never automatic replay')
    script=Path(__file__).with_name('evaluate_frozen.py');scripts=[Path(__file__),script,Path(__file__).with_name('summarize_validation.py'),Path(__file__).with_name('summarize_frozen.py')];hashes={str(p):sha(p) for p in scripts}
    started=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    def state(phase,**extra):
        tmp=q/'state.tmp.json';tmp.write_text(json.dumps(dict(pid=os.getpid(),phase=phase,started_utc=started,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256=hashes,**extra),indent=2)+'\n');tmp.replace(q/'state.private.json')
    def command(module,*args):subprocess.run([sys.executable,'-B','-m',module,*map(str,args)],cwd=code,check=True)
    try:
        while True:
            dependency=json.loads((root/'queues/phase7_subject_candidate/state.private.json').read_text())
            if dependency['failed']:raise RuntimeError('Supplemental training failed')
            if dependency.get('phase')=='COMPLETE':break
            os.kill(dependency['queue_pid'],0);state('WAITING_SUBJECT_VALIDATION',test_accessed=False);time.sleep(10)
        if any(sha(p)!=h for p,h in hashes.items()):raise RuntimeError('Evaluation source changed')
        state('VERIFYING_DEVELOPMENT',test_accessed=False)
        main_report=root/'reports/phase5_complete_host_20261011/validation_summary.public.json';main=json.loads(main_report.read_text())
        if main['verification']!='PASSED' or main.get('protocol')!='sample':raise ValueError('Main validation is not verified')
        subject_out=root/'reports/subject_comparison_complete_20261011'
        command('BrainPromptPro.scripts.adni_1to7.summarize_validation','--root',root,'--protocol','subject','--output',subject_out)
        subject=json.loads((subject_out/'validation_summary.public.json').read_text())
        decision=json.loads((root/'candidate_selection.private.json').read_text());primary=decision['primary_candidate']
        if decision['validation_summary_sha256']!=sha(main_report) or decision['test_accessed']:raise ValueError('Main selection changed')
        freeze=json.loads((root/'queues/phase5_three_seed/recipe_freeze.private.json').read_text());models=freeze['main_matrix']+freeze['controls']
        expected={(protocol,exp,s) for protocol,exps in [('sample',models),('subject',list(dict.fromkeys(['A0',primary])))] for exp in exps for s in [42,3407,2026]}
        rows={}
        for protocol,report in [('sample',main),('subject',subject)]:
            for row in report['runs']:
                identity=(protocol,row['experiment'],row['seed'])
                if identity in expected:rows[identity]=row
        if set(rows)!=expected:raise ValueError('Missing formal comparison before TEST')
        entries=[]
        for protocol,exp,seed in sorted(expected):
            row=rows[(protocol,exp,seed)];run=root/('experiments' if protocol=='sample' else 'subject_experiments')/exp/f'seed{seed}'
            record=json.loads((run/'result.json').read_text())
            if sha(run/'best_validation.pt')!=row['checkpoint_sha256']:raise ValueError('Checkpoint changed after verification')
            for path,expected_sha in record['source_sha256'].items():
                if sha(path)!=expected_sha:raise ValueError('Training source changed')
            entries.append(dict(protocol=protocol,experiment=exp,seed=seed,run=str(run),checkpoint_sha256=row['checkpoint_sha256'],dataset_hash=row['dataset_hash'],split_hash=row['split_hash'],training_result_sha256=sha(run/'result.json')))
        exposure=root/'reports/cross_protocol_exposure_20261011.private.json'
        if not exposure.exists():raise ValueError('Cross-protocol exposure audit missing')
        doc=dict(cross_protocol_exposure_audit_sha256=sha(exposure),isolation_scope='Within each protocol only; cross-protocol and historical development exposure exists',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),development_complete=True,test_selection_policy='NO_TEST_BASED_SELECTION',primary_candidate=primary,selection_decision_sha256=sha(root/'candidate_selection.private.json'),main_validation_sha256=sha(main_report),subject_validation_sha256=sha(subject_out/'validation_summary.public.json'),evaluator_sha256=sha(script),protocol_hashes={p:next(x['split_hash'] for x in entries if x['protocol']==p) for p in ['sample','subject']},evaluations=entries)
        selection=root/'selection_lock.json'
        with selection.open('x') as stream:json.dump(doc,stream,indent=2)
        lockhash=sha(selection);state('FROZEN',selection_lock_sha256=lockhash,evaluations=len(entries),test_accessed=False)
        gpus=[int(x) for x in a.gpus.split(',')]
        if len(gpus)!=len(set(gpus)):raise ValueError('Duplicate GPU slots')
        # Resource check never stops or replaces another user's process.
        for gpu in gpus:
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<5000:raise RuntimeError('Insufficient free GPU memory; retain lock and request reviewed continuation')
        batches=[entries[i::len(gpus)] for i in range(len(gpus))];failed=__import__('threading').Event()
        def worker(gpu,batch):
            completed=[]
            for entry in batch:
                if failed.is_set():break
                protocol,exp,seed=entry['protocol'],entry['experiment'],entry['seed'];output=root/'final_test'/protocol/exp/f'seed{seed}'
                argv=[sys.executable,'-B',str(script),'--run',entry['run'],'--output',str(output),'--protocol',protocol,'--role','test','--selection-lock',str(selection),'--selection-lock-sha256',lockhash,'--device','cuda:0']
                with (q/f'{protocol}_{exp}_seed{seed}.log').open('x') as stream:
                    value=subprocess.run(argv,cwd=code,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=stream,stderr=subprocess.STDOUT)
                row=dict(protocol=protocol,experiment=exp,seed=seed,returncode=value.returncode,output=str(output));completed.append(row);print(json.dumps(row),flush=True)
                if value.returncode:failed.set();break
            return completed
        state('EVALUATING_TEST',selection_lock_sha256=lockhash,evaluations=len(entries),test_accessed=True)
        results=[]
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            for done in as_completed([pool.submit(worker,gpu,batch) for gpu,batch in zip(gpus,batches)]):results+=done.result()
        (q/'evaluation_status.private.json').write_text(json.dumps(results,indent=2)+'\n')
        if failed.is_set() or len(results)!=len(entries):raise RuntimeError('Partial locked TEST; preserve outputs and inspect before any recovery')
        report=root/'reports/frozen_test_complete_20261011'
        command('BrainPromptPro.scripts.adni_1to7.summarize_frozen','--lock',selection,'--output',report)
        state('COMPLETE',selection_lock_sha256=lockhash,evaluations=len(entries),test_accessed=True,report=str(report))
    except Exception as exc:
        state('FAILED',error=str(exc),selection_lock_exists=(root/'selection_lock.json').exists());raise


if __name__=='__main__':main()
