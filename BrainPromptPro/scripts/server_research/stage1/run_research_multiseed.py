import json,os,subprocess,time
from pathlib import Path
root=Path('/home/panyao/AD/BrainPromptPro/outputs/research/final_optimization_20261007')
py='/home/panyao/miniconda3/envs/brainprompt/bin/python'
profile='sqrt_inverse_frequency'
choice=root/'supporting_profile_selection.json'
if choice.exists(): raise FileExistsError(choice)
selection={'profile':profile,'basis':'BrainPrompt seed=42 validation-only comparison; all five primary metrics improved; current Pro scores not used to select profile','unweighted_result':str(root/'experiments/BrainPrompt/unweighted/seed_42/result.json'),'weighted_result':str(root/'experiments/BrainPrompt'/profile/'seed_42/result.json'),'test_accessed':False,'remaining_seeds':[3407,2026],'apply_to_all_existing_models':True,'training_trick_is_not_architectural_innovation':True}
choice.write_text(json.dumps(selection,indent=2)+'\n')
while True:
 s=json.loads((root/'baseline_queue_status.json').read_text())
 if not s['running'] and not s['pending']:
  assert all(x.get('returncode',0)==0 for x in s['finished']);break
 time.sleep(5)
jobs=[(name,profile,seed) for seed in (3407,2026) for name in ('BrainPrompt','CurrentPro_legacy','CurrentPro_stable')]
active={};finished=[];start=time.monotonic();logroot=root/'queue_logs'
while jobs or active:
 for gpu in (4,5):
  if gpu in active and active[gpu]['p'].poll() is not None:
   item=active.pop(gpu);item['log'].close();res=dict(model=item['job'][0],profile=item['job'][1],seed=item['job'][2],gpu=gpu,returncode=item['p'].returncode)
   finished.append(res);print(json.dumps(res),flush=True)
  if gpu not in active and jobs:
   name,profile,seed=job=jobs.pop(0)
   out=root/'experiments'/name/profile/f'seed_{seed}'
   if out.exists(): raise FileExistsError(out)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONWARNINGS='ignore::FutureWarning')
   log=(logroot/f'{name}_{profile}_{seed}.log').open('x')
   p=subprocess.Popen([py,'-m','BrainPromptPro.research_runner','train','--model',name,'--profile',profile,'--seed',str(seed),'--device','cuda:0'],cwd='/home/panyao/AD',env=env,stdout=log,stderr=subprocess.STDOUT)
   active[gpu]=dict(p=p,log=log,job=job);print(json.dumps(dict(started=job,gpu=gpu,pid=p.pid)),flush=True)
 status=dict(finished=finished,running=[dict(gpu=g,job=v['job'],pid=v['p'].pid) for g,v in active.items()],pending=jobs,elapsed_seconds=time.monotonic()-start)
 temp=root/'multiseed_queue_status.tmp.json';temp.write_text(json.dumps(status,indent=2));temp.replace(root/'multiseed_queue_status.json')
 if active:time.sleep(5)
assert all(x['returncode']==0 for x in finished),finished
print('MULTISEED_DEVELOPMENT_COMPLETE',flush=True)
