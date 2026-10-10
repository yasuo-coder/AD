"""Bounded GPU queue for train/validation-only Pro candidates."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
root=Path('/home/panyao/AD/BrainPromptPro/outputs/optimizations')/a.run_id
jobs=[(c,f) for f in range(10) for c in ('stable','stable_weighted')]
# One job on GPUs 6/7, which also have unrelated lightweight processes.
slots=[('5',0),('6',0),('7',0),('5',1)]
active={};done=[];failed=[];start=time.monotonic()
(root/'worker_logs').mkdir(exist_ok=False)
while jobs or active:
    for slot,item in list(active.items()):
        rc=item['process'].poll()
        if rc is None:continue
        item['log'].close();event=dict(candidate=item['candidate'],fold=item['fold'],gpu=slot[0],returncode=rc)
        (done if rc==0 else failed).append(event);del active[slot]
        print(json.dumps(dict(event='completed' if rc==0 else 'failed',**event)),flush=True)
    if not failed:
        for slot in slots:
            if slot in active or not jobs:continue
            candidate,fold=jobs.pop(0)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=slot[0],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='41',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            log=(root/'worker_logs'/f'{candidate}_{fold}.log').open('w')
            command=[sys.executable,'-m','BrainPromptPro.optimization_cv','train','--run-id',a.run_id,'--candidate',candidate,'--fold',str(fold)]
            proc=subprocess.Popen(command,cwd='/home/panyao/AD',env=env,stdout=log,stderr=subprocess.STDOUT)
            active[slot]=dict(process=proc,log=log,candidate=candidate,fold=fold)
            print(json.dumps(dict(event='started',candidate=candidate,fold=fold,gpu=slot[0],pid=proc.pid)),flush=True)
    status=dict(completed=done,failed=failed,pending=jobs,
                active=[dict(candidate=i['candidate'],fold=i['fold'],gpu=s[0],pid=i['process'].pid) for s,i in active.items()],
                elapsed_seconds=time.monotonic()-start)
    temp=root/'queue_status.tmp.json';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(root/'queue_status.json')
    if failed and not active:break
    if active:time.sleep(10)
print(json.dumps(dict(event='finished',completed=len(done),failed=failed,elapsed_seconds=time.monotonic()-start)),flush=True)
sys.exit(1 if failed else 0)
