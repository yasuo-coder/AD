"""Resumable development-only queue; never trains over existing experiment artifacts."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--models',nargs='+',required=True);p.add_argument('--seeds',nargs='+',type=int,default=[42,3407,2026]);p.add_argument('--gpus',nargs='+',type=int,default=list(range(8)));p.add_argument('--runner',default='BrainPromptPro.candidate_runner');a=p.parse_args()
root=Path('/home/panyao/AD/BrainPromptPro/outputs/research/final_optimization_20261007/stage2_20261008');py='/home/panyao/miniconda3/envs/brainprompt/bin/python';logdir=root/'queue_logs';logdir.mkdir(exist_ok=True);status=root/(a.phase+'_queue_status.json')
if status.exists():raise FileExistsError('Use a distinct phase identifier when resuming; preserve old status')
jobs=[];finished=[]
for seed in a.seeds:
 for model in a.models:
  out=root/'experiments'/model/'sqrt_inverse_frequency'/f'seed_{seed}'
  if (out/'result.json').exists():
   x=json.loads((out/'result.json').read_text());assert x['model']==model and x['seed']==seed and x['test']=='NOT_EVALUATED'
   assert hashlib.sha256((out/'best_validation.pt').read_bytes()).hexdigest()==x['checkpoint_sha256']
   finished.append(dict(model=model,seed=seed,state='preexisting_completed'));continue
  if out.exists():raise FileExistsError(f'Incomplete artifact needs explicit recovery audit: {out}')
  jobs.append((model,seed))
active={};start=time.monotonic();failed=False
while jobs or active:
 for gpu in a.gpus:
  if gpu in active and active[gpu]['p'].poll() is not None:
   item=active.pop(gpu);item['log'].close();record=dict(model=item['job'][0],seed=item['job'][1],gpu=gpu,returncode=item['p'].returncode);finished.append(record);print(json.dumps(record),flush=True);failed|=record['returncode']!=0
  if gpu not in active and jobs and not failed:
   model,seed=job=jobs.pop(0);log=(logdir/f'{a.phase}_{model}_{seed}.log').open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONWARNINGS='ignore::FutureWarning')
   proc=subprocess.Popen([py,'-m',a.runner,'train','--model',model,'--seed',str(seed),'--device','cuda:0'],cwd='/home/panyao/AD',env=env,stdout=log,stderr=subprocess.STDOUT)
   active[gpu]=dict(p=proc,log=log,job=job);print(json.dumps(dict(start=job,gpu=gpu,pid=proc.pid)),flush=True)
 report=dict(phase=a.phase,runner=a.runner,finished=finished,running=[dict(gpu=g,job=v['job'],pid=v['p'].pid) for g,v in active.items()],pending=jobs,elapsed_seconds=time.monotonic()-start,failed=failed)
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(status)
 if failed and not active:break
 if active:time.sleep(5)
if failed:raise RuntimeError('A job failed; pending jobs were not started. Inspect logs before resuming.')
print('QUEUE_COMPLETE',flush=True)
