"""Detached finite GPU queue; lock excludes duplicate controllers, no overwrites."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from BrainPromptPro.scripts.adni_1to7.audit_dataset import sha,write_json


def completed(entry):
    out=Path(entry['output']);data=json.loads((out/'result.json').read_text())
    if data['experiment']!=entry['experiment'] or data['seed']!=entry['seed'] or data['smoke_epochs']:
        raise ValueError('Completed job identity differs from queue declaration')
    if data['config_sha256']!=sha(entry['config']):raise ValueError('Completed config differs')
    cfg=json.loads(Path(entry['config']).read_text())
    split_path=cfg['study']['sample_split' if str(entry.get('split','0'))=='0' else 'subject_split']
    if data['split_hash']!=sha(split_path):raise ValueError('Completed split differs')
    for field in ['initial','reference']:
        dependency=entry.get(field+'_checkpoint')
        if data[field+'_sha256']!=(sha(dependency) if dependency else None):raise ValueError('Completed dependency differs')
    root=Path(entry['code_root']).resolve()
    if not any(Path(p).is_relative_to(root) for p in data['source_sha256']):raise ValueError('Completed code root differs')
    if any(sha(p)!=h for p,h in data['source_sha256'].items()):raise ValueError('Completed source changed')
    cp=out/('best_training.pt' if data['stage']=='pretrain' else 'best_validation.pt')
    if sha(cp)!=data['checkpoint_sha256']:raise ValueError('Existing result checkpoint mismatch')


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--gpus',default='0,3,5,7');p.add_argument('--after-state',type=Path,help='Wait for this queue to finish successfully before using its GPU slots');a=p.parse_args()
    a.state.parent.mkdir(parents=True,exist_ok=True)
    lock=a.state.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=json.loads(a.jobs.read_text());pending=[];finished=[];active={};failed=False
    for entry in jobs:
        out=Path(entry['output']);result=out/'result.json'
        if result.exists():
            completed(entry)
            finished.append(dict(**entry,status='ALREADY_COMPLETE'));continue
        # Existing partial output is resumed only by explicit job declaration.
        if out.exists() and not entry.get('resume',False):raise FileExistsError('Partial output requires reviewed resume job')
        pending.append(entry)
    if a.after_state:
        if a.after_state.resolve()==a.state.resolve():raise ValueError('Queue cannot depend on itself')
        while True:
            dependency=json.loads(a.after_state.read_text())
            if dependency['failed']:
                write_json(a.state,dict(queue_pid=os.getpid(),waiting_for=str(a.after_state),running=[],pending=pending,finished=finished,failed=True,error='dependency_failed'))
                raise RuntimeError('Dependency queue failed; inspect before continuing')
            if not dependency['running'] and not dependency['pending']:break
            temporary=a.state.with_suffix('.tmp.json')
            write_json(temporary,dict(queue_pid=os.getpid(),waiting_for=str(a.after_state),running=[],pending=pending,finished=finished,failed=False))
            temporary.replace(a.state);time.sleep(10)
    gpus=[int(x) for x in a.gpus.split(',')];start=time.time()
    if len(set(gpus))!=len(gpus):raise ValueError('Duplicate GPU slots')
    while pending or active:
        for gpu in gpus:
            if gpu in active and active[gpu]['proc'].poll() is not None:
                x=active.pop(gpu);x['stream'].close();code=x['proc'].returncode
                if code==0:
                    try:completed(x['entry'])
                    except (FileNotFoundError,ValueError,KeyError):code=99
                failed|=code!=0
                finished.append(dict(**x['entry'],gpu=gpu,returncode=code,status='COMPLETE' if code==0 else 'FAILED'))
            if gpu not in active and pending and not failed:
                entry=pending.pop(0);out=Path(entry['output']);log=a.state.parent/(entry['job_id']+'.log')
                # Recovery receives a new log; no old evidence is overwritten.
                stream=log.open('x');command=[sys.executable,'-B','-m','BrainPromptPro.scripts.adni_1to7.train','--config',entry['config'],'--experiment',entry['experiment'],'--seed',str(entry['seed']),'--device','cuda:0','--output',str(out)]
                if entry.get('resume'):command+=['--resume']
                command+=['--split',str(entry.get('split','0'))]
                for field,flag in [('reference_checkpoint','--reference-checkpoint'),('initial_checkpoint','--initial-checkpoint')]:
                    if entry.get(field):command+=[flag,entry[field]]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                proc=subprocess.Popen(command,cwd=entry['code_root'],env=env,stdout=stream,stderr=subprocess.STDOUT)
                active[gpu]=dict(entry=dict(entry,command=command,log=str(log),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),proc=proc,stream=stream)
        temporary=a.state.with_suffix('.tmp.json')
        write_json(temporary,dict(queue_pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(start)),running=[dict(**x['entry'],pid=x['proc'].pid,gpu=g) for g,x in active.items()],pending=pending,finished=finished,failed=failed))
        temporary.replace(a.state)
        if failed and not active:break
        if active:time.sleep(3)
    temporary=a.state.with_suffix('.tmp.json')
    write_json(temporary,dict(queue_pid=os.getpid(),running=[],pending=pending,finished=finished,failed=failed,phase='FAILED' if failed else 'COMPLETE'))
    temporary.replace(a.state)
    if failed:raise RuntimeError('Job failed; review evidence before resume')
    print('QUEUE_COMPLETE',flush=True)

if __name__=='__main__':main()
