import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','single','combination'],required=True);p.add_argument('--models',nargs='+');p.add_argument('--gpus',nargs='+',type=int,default=[1,4,5,7]);a=p.parse_args()
root=Path('/home/panyao/AD/BrainPromptPro/outputs/research/final_optimization_20261007/stage2_20261008');py='/home/panyao/miniconda3/envs/brainprompt/bin/python'
models=a.models or (['BP_control','M1','M2','M3','M4','M5'] if a.phase=='smoke' else ['M1','M2','M3','M4','M5'])
seeds=[42] if a.phase=='smoke' else [42,3407,2026]
jobs=[(m,s) for s in seeds for m in models];active={};finished=[];start=time.monotonic();logdir=root/'queue_logs';logdir.mkdir(exist_ok=True)
statuspath=root/(a.phase+'_queue_status.json')
if statuspath.exists():raise FileExistsError(statuspath)
while jobs or active:
 for gpu in a.gpus:
  if gpu in active and active[gpu]['p'].poll() is not None:
   item=active.pop(gpu);item['log'].close();r=dict(model=item['job'][0],seed=item['job'][1],gpu=gpu,returncode=item['p'].returncode);finished.append(r);print(json.dumps(r),flush=True)
  if gpu not in active and jobs:
   model,seed=job=jobs.pop(0)
   directory=root/('smoke' if a.phase=='smoke' else 'experiments')/model/'sqrt_inverse_frequency'/f'seed_{seed}'
   if directory.exists():raise FileExistsError(directory)
   logfile=logdir/f'{a.phase}_{model}_{seed}.log';log=logfile.open('x')
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONWARNINGS='ignore::FutureWarning')
   cmd=[py,'-m','BrainPromptPro.candidate_runner','train','--model',model,'--seed',str(seed),'--device','cuda:0']
   if a.phase=='smoke':cmd+=['--smoke-epochs','2']
   proc=subprocess.Popen(cmd,cwd='/home/panyao/AD',env=env,stdout=log,stderr=subprocess.STDOUT);active[gpu]=dict(p=proc,log=log,job=job);print(json.dumps(dict(start=job,gpu=gpu,pid=proc.pid)),flush=True)
 tmp=statuspath.with_suffix('.tmp');tmp.write_text(json.dumps(dict(phase=a.phase,finished=finished,running=[dict(gpu=g,job=v['job'],pid=v['p'].pid) for g,v in active.items()],pending=jobs,elapsed_seconds=time.monotonic()-start),indent=2));tmp.replace(statuspath)
 if active:time.sleep(5)
assert all(x['returncode']==0 for x in finished),finished
print(a.phase.upper()+'_COMPLETE',flush=True)
