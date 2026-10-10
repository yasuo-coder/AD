"""Detached finite GPU queue; lock excludes duplicate controllers, no overwrites."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from BrainPromptPro.scripts.adni_1to7.audit_dataset import sha,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--gpus',default='0,3,5,7');a=p.parse_args()
    a.state.parent.mkdir(parents=True,exist_ok=True)
    lock=a.state.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=json.loads(a.jobs.read_text());pending=[];finished=[];active={};failed=False
    for entry in jobs:
        out=Path(entry['output']);result=out/'result.json'
        if result.exists():
            data=json.loads(result.read_text());cp=out/('best_training.pt' if data['stage']=='pretrain' else 'best_validation.pt')
            if sha(cp)!=data['checkpoint_sha256']:raise ValueError('Existing result checkpoint mismatch')
            finished.append(dict(**entry,status='ALREADY_COMPLETE'));continue
        # Existing partial output is resumed only by explicit job declaration.
        if out.exists() and not entry.get('resume',False):raise FileExistsError('Partial output requires reviewed resume job')
        pending.append(entry)
    gpus=[int(x) for x in a.gpus.split(',')];start=time.time()
    while pending or active:
        for gpu in gpus:
            if gpu in active and active[gpu]['proc'].poll() is not None:
                x=active.pop(gpu);x['stream'].close();code=x['proc'].returncode;failed|=code!=0
                finished.append(dict(**x['entry'],gpu=gpu,returncode=code,status='COMPLETE' if code==0 else 'FAILED'))
            if gpu not in active and pending and not failed:
                entry=pending.pop(0);out=Path(entry['output']);log=a.state.parent/(entry['job_id']+'.log')
                # Recovery receives a new log; no old evidence is overwritten.
                stream=log.open('x');command=[sys.executable,'-B','-m','BrainPromptPro.scripts.adni_1to7.train','--config',entry['config'],'--experiment',entry['experiment'],'--seed',str(entry['seed']),'--device','cuda:0','--output',str(out)]
                if entry.get('resume'):command+=['--resume']
                for field,flag in [('reference_checkpoint','--reference-checkpoint'),('initial_checkpoint','--initial-checkpoint')]:
                    if entry.get(field):command+=[flag,entry[field]]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                proc=subprocess.Popen(command,cwd=entry['code_root'],env=env,stdout=stream,stderr=subprocess.STDOUT)
                active[gpu]=dict(entry=dict(entry,command=command,log=str(log),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),proc=proc,stream=stream)
        write_json(a.state,dict(queue_pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(start)),running=[dict(**x['entry'],pid=x['proc'].pid,gpu=g) for g,x in active.items()],pending=pending,finished=finished,failed=failed))
        if failed and not active:break
        if active:time.sleep(3)
    if failed:raise RuntimeError('Job failed; review evidence before resume')
    print('QUEUE_COMPLETE',flush=True)

if __name__=='__main__':main()
