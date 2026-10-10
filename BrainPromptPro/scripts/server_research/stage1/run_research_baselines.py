import json,os,subprocess,time
from pathlib import Path
root=Path('/home/panyao/AD/BrainPromptPro/outputs/research/final_optimization_20261007')
py='/home/panyao/miniconda3/envs/brainprompt/bin/python'
jobs=[(name,profile,42) for profile in ('unweighted','sqrt_inverse_frequency') for name in ('BrainPrompt','CurrentPro_legacy','CurrentPro_stable')]
active={};finished=[];start=time.monotonic();logroot=root/'queue_logs';logroot.mkdir(exist_ok=True)
while jobs or active:
    for gpu in (4,5):
        if gpu in active and active[gpu]['p'].poll() is not None:
            item=active.pop(gpu);item['log'].close()
            result=dict(model=item['job'][0],profile=item['job'][1],seed=item['job'][2],gpu=gpu,returncode=item['p'].returncode)
            finished.append(result);print(json.dumps(result),flush=True)
        if gpu not in active and jobs:
            name,profile,seed=job=jobs.pop(0)
            out=root/'experiments'/name/profile/f'seed_{seed}'
            if (out/'result.json').exists():
                finished.append(dict(model=name,profile=profile,seed=seed,status='already_complete'));continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            log=(logroot/f'{name}_{profile}_{seed}.log').open('w')
            p=subprocess.Popen([py,'-m','BrainPromptPro.research_runner','train','--model',name,'--profile',profile,'--seed',str(seed),'--device','cuda:0'],cwd='/home/panyao/AD',env=env,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=dict(p=p,log=log,job=job)
            print(json.dumps(dict(started=job,gpu=gpu,pid=p.pid)),flush=True)
    status=dict(finished=finished,running=[dict(gpu=g,job=v['job'],pid=v['p'].pid) for g,v in active.items()],pending=jobs,elapsed_seconds=time.monotonic()-start)
    temp=root/'baseline_queue_status.tmp.json';temp.write_text(json.dumps(status,indent=2));temp.replace(root/'baseline_queue_status.json')
    if active:time.sleep(5)
assert all(r.get('returncode',0)==0 for r in finished),finished
print('BASELINE_DEVELOPMENT_COMPLETE',flush=True)
