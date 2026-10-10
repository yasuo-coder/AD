"""Evaluate only both explicitly locked experiment sets; never train or choose models."""
import json,os,subprocess,time
from pathlib import Path
r=Path('/home/panyao/AD/BrainPromptPro/outputs/research/final_optimization_20261007');s=r/'stage2_20261008'
py='/home/panyao/miniconda3/envs/brainprompt/bin/python'
pending=[]
for base in (r,s):
 lock=json.loads((base/'selection_lock.json').read_text())
 for entry in lock['experiments']:
  p=Path(entry['experiment'])
  assert not (p/'test_result.json').exists(), 'Final test is single-pass; existing results must not be overwritten'
  spec=json.loads((p/'spec.json').read_text())
  runner='BrainPromptPro.research_runner' if base==r else 'BrainPromptPro.candidate_v3_initialized_runner' if spec.get('population_version')==3 else 'BrainPromptPro.candidate_runner'
  pending.append(dict(**entry,runner=runner))
status=s/'final_test_queue_status.json';assert not status.exists()
gpus=[0,5,7];active={};finished=[];failed=False
while pending or active:
 for gpu in gpus:
  if gpu in active and active[gpu]['process'].poll() is not None:
   x=active.pop(gpu);x['log'].close();entry=x['entry'];entry.update(gpu=gpu,returncode=x['process'].returncode);finished.append(entry);failed|=entry['returncode']!=0
   print(json.dumps(entry),flush=True)
  if gpu not in active and pending and not failed:
   entry=pending.pop(0);out=Path(entry['experiment']);log=(s/'queue_logs'/f"final_test_{entry['model']}_{entry['seed']}.log").open('x')
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   proc=subprocess.Popen([py,'-m',entry['runner'],'test','--experiment',str(out),'--device','cuda:0'],cwd='/home/panyao/AD',env=env,stdout=log,stderr=subprocess.STDOUT)
   active[gpu]=dict(process=proc,entry=entry,log=log)
 report=dict(finished=finished,running=[dict(gpu=g,pid=x['process'].pid,**x['entry']) for g,x in active.items()],pending=pending,failed=failed)
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(status)
 if failed and not active:break
 if active:time.sleep(2)
if failed:raise RuntimeError('Final test export failed; inspect before any recovery')
print('FINAL_TEST_EXPORT_COMPLETE',flush=True)
