"""Real namespace denial checks and bounded real-data smoke tests, not performance runs."""
import subprocess,time
from .common import *
from .isolation import command
def main():
    root=ROOT/'private/preflight_final';root.mkdir(parents=True,exist_ok=True)
    taskfile=root/'task.json';write(taskfile,dict(stage='A',name='original',kind='original',plan=0))
    code="""
import json,os,socket,torch
from pathlib import Path
targets=['/home/panyao/AD/BrainPromptNComms2025/out/adni7_ncomms_subject10_20261021/cohort.private.pt','/home/panyao/AD/BrainPromptNComms2025/out/adni7_ncomms_subject10_20261021/units/seed_42/fold_00/predictions.private.pt','/data2','/home/panyao/AD/BrainPromptR2/private','/proc/1/root/home/panyao/AD/BrainPromptNComms2025/out']
denied={}
for name in targets:
 try:
  with open(name,'rb') as f:f.read(1)
 except OSError:denied[name]=True
 else:denied[name]=False
assert all(denied.values())
data=torch.load('/work/data/data.private.pt',weights_only=True,map_location='cpu')
assert len(data['fc'])==1051 and len(set(data['groups']))==715
assert torch.cuda.is_available()
a=torch.ones(2,device='cuda');assert float(a.sum())==2
network_interfaces=os.listdir('/sys/class/net')
try:socket.create_connection(('1.1.1.1',443),timeout=1)
except OSError:network_blocked=True
else:network_blocked=False
assert network_blocked
print(json.dumps(dict(denied=denied,fit_scans=len(data['fc']),fit_subjects=len(set(data['groups'])),cuda=True,network_blocked=network_blocked)))
"""
    p=subprocess.run(command(taskfile,root/'probe','4',code=code),capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr+p.stdout)
    probe=json.loads(p.stdout.strip().splitlines()[-1]);results=[]
    for kind,stage,condition in [('original','A','no_prompt'),('soft_topk','A','no_prompt'),('g_off','B','no_prompt'),('mlp','B','no_prompt'),('logistic','B','no_prompt'),('g_off','D','roi_age_sex'),('logistic','D','full')]:
        task=dict(stage=stage,name=stage+'_'+kind,kind=kind,plan=0,condition=condition)
        if kind=='logistic':task['C']=.1
        path=root/(stage+'_'+kind);write(taskfile,task)
        with (root/(stage+'_'+kind+'.log')).open('w') as log:
            ret=subprocess.run(command(taskfile,path,'4',extra=['--smoke-epochs','2']),stdout=log,stderr=subprocess.STDOUT)
        if ret.returncode:raise RuntimeError(f'Smoke failed {task}; see {log.name}')
        r=read(path/'result.json');assert r['status']=='SMOKE_COMPLETE';results.append(dict(task=task,status=r['status'],parameters=r['parameters']))
        if kind=='soft_topk':
            expected=torch.load(path/'predictions.private.pt',weights_only=False);(path/'result.json').unlink()
            with (root/'resume.log').open('w') as log:
                ret=subprocess.run(command(taskfile,path,'4',extra=['--smoke-epochs','2']),stdout=log,stderr=subprocess.STDOUT)
            assert ret.returncode==0
            resumed=read(path/'result.json');actual=torch.load(path/'predictions.private.pt',weights_only=False)
            assert resumed['epochs']==2
            resume_diff=max(float(np.max(np.abs(expected[k]['prob']-actual[k]['prob']))) for k in expected)
            assert resume_diff<=1e-6, resume_diff
    write(ROOT/'public/preflight.public.json',dict(status='PASS',namespace=probe,smoke=results,resume_terminal_checkpoint='PASS; same epoch and subject probabilities within 1e-6 (DGL GPU reductions)',resume_max_probability_delta=resume_diff,smoke_weights_reused=False))
    print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':main()
