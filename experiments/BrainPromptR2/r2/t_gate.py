"""Bounded FIT-only capacity and information-flow checks, never performance evidence."""
import argparse,copy,time
import numpy as np
import torch
from torch.nn import functional as F
from .common import *
from .models import StagedT,BrainPromptT
def grad_report(model):
 groups={}
 for n,p in model.named_parameters():
  if p.grad is not None:groups[n.split('.')[0]]=groups.get(n.split('.')[0],0.)+float(p.grad.detach().square().sum())
 return {k:v**.5 for k,v in groups.items()}
def diagnostics(model,data,ids,device):
 model.eval();att=[];hooks=[]
 def hook(layer,args):
  tokens,bias=args;b,n,h=tokens.shape
  with torch.no_grad():
   q,k,v=layer.qkv(layer.norm1(tokens)).reshape(b,n,3,layer.heads,h//layer.heads).permute(2,0,3,1,4).unbind(0)
   scores=q@k.transpose(-1,-2)/(h//layer.heads)**.5
   if layer.use_bias:scores=scores+layer.beta[None,:,None,None]*bias[:,None]
   a=scores.softmax(-1);att.append(float(-(a*a.clamp_min(1e-30).log()).sum(-1).mean()))
 for layer in model.encoder:hooks.append(layer.register_forward_pre_hook(hook))
 with torch.no_grad():base=model.forward_batch(data,ids,device)['logits'];second=model.forward_batch(data,ids,device)['logits']
 for h in hooks:h.remove()
 model.zero_grad(set_to_none=True);out=model.forward_batch(data,ids,device)['logits'];loss=F.cross_entropy(out,data['y'][ids].to(device));loss.backward()
 gradients=grad_report(model);betas=[dict(value=l.beta.detach().cpu().tolist(),gradient=None if l.beta.grad is None else l.beta.grad.detach().cpu().tolist()) for l in model.encoder]
 changes={}
 for mode in ['patient_zero','patient_permuted','FC_mean']:
  altered=dict(data)
  if mode.startswith('patient'):
   altered['subject_prompt']=data['subject_prompt'].clone()
   altered['subject_prompt'][ids]=0 if mode=='patient_zero' else data['subject_prompt'][list(reversed(ids))]
  else:
   altered['fc']=data['fc'].clone();altered['fc'][ids]=data['fc'][ids].mean(0)
  with torch.no_grad():other=model.forward_batch(altered,ids,device)['logits'];diff=(other-other.mean(1,keepdim=True))-(base-base.mean(1,keepdim=True))
  changes[mode]=float(diff.square().mean().sqrt())
 if hasattr(model,'disease_prompt'):
  saved=model.disease_prompt.clone();permutation=torch.tensor([1,2,3,4,0],device=device)
  model.disease_prompt.copy_(saved[permutation])
  with torch.no_grad():other=model.forward_batch(data,ids,device)['logits']
  changes['query_permuted_fixed_columns_rms']=float((other-base).square().mean().sqrt())
  changes['query_permuted_aligned_columns_rms']=float((other-base[:,permutation]).square().mean().sqrt())
  model.disease_prompt.copy_(saved)
 return dict(eval_repeat_max_abs=float((base-second).abs().max()),logits_variance_across_patients=base.detach().var(0,unbiased=False).cpu().tolist(),
  centered_logits_variance=float((base-base.mean(1,keepdim=True)).detach().var(0,unbiased=False).mean()),
  gradient_norms=gradients,attention_entropy=att,beta=betas,counterfactual_centered_logit_rms=changes)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');ap.add_argument('--steps',type=int,default=1000);a=ap.parse_args()
 device=torch.device(a.device);data=torch.load(ROOT/'prepared/data.private.pt',weights_only=True,map_location='cpu');seed_all(5729)
 train=[];val=[]
 for c in range(5):
  groups=sorted({g for i,g in enumerate(data['groups']) if int(data['y'][i])==c})
  for g in groups[:5]:train.append(data['groups'].index(g))
  for g in groups[5:8]:val.append(data['groups'].index(g))
 assert len(train)==25 and len(val)==15
 old=read(OLD/'out/adni7_ncomms_subject10_20261021/units/seed_42/fold_00/components/E4/result.json')['config']
 results=[];earlier=True
 for name,stage in [('T_roi_mean',0),('T_plus_fc_bias',1),('T_plus_patient_token',2),('T_plus_disease_query',3),('E4_exact_architecture',None)]:
  if stage is not None and not earlier:
   results.append(dict(name=name,status='SKIPPED_AFTER_GATE_FAILURE'));continue
  seed_all(5729);model=(BrainPromptT(old,data) if stage is None else StagedT(data,stage)).to(device)
  optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.)
  first=diagnostics(model,data,train,device);history=[];started=time.time()
  for step in range(1,a.steps+1):
   model.train();optimizer.zero_grad(set_to_none=True)
   logits=model.forward_batch(data,train,device)['logits'];loss=F.cross_entropy(logits,data['y'][train].to(device));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
   if step==1 or step%25==0:
    model.eval()
    with torch.no_grad():
     tl=model.forward_batch(data,train,device)['logits'];vl=model.forward_batch(data,val,device)['logits']
     ce=float(F.cross_entropy(tl,data['y'][train].to(device)));acc=float((tl.argmax(1)==data['y'][train].to(device)).float().mean())
     vm=metric(data['y'][val].numpy(),vl.double().softmax(-1).cpu().numpy())
    history.append(dict(update=step,train_ce=ce,train_accuracy=acc,internal_validation=vm))
    if acc>=.96 and ce<=.15:break
  final=diagnostics(model,data,train,device)
  passed=acc>=.96 and ce<=.15 and final['eval_repeat_max_abs']<=1e-6 and first['gradient_norms'].get('fc_projection',0)>1e-8 and first['gradient_norms'].get('encoder',0)>1e-8 and final['counterfactual_centered_logit_rms']['FC_mean']>1e-4
  result=dict(name=name,status='PASS_CAPACITY' if passed else 'FAIL_CAPACITY',updates=step,seconds=time.time()-started,
   initial=first,final=final,history=history,passed=passed,
   caveat='25 FIT subjects for fitting + 15 disjoint FIT subjects for diagnostic validation; no outer VAL/TEST. CE-only capacity check, no historical auxiliary objective. E4 preserves architecture/dropout, but this is a bounded diagnostic with weight_decay=0, not formal training.')
  results.append(result);write(ROOT/'private/t_gate_progress.json',{'results':results})
  print(name,result['status'],step,ce,acc,flush=True)
  if stage is not None:earlier=earlier and passed
  del model,optimizer
  if device.type=='cuda':torch.cuda.empty_cache()
 eligible=all(x.get('passed',False) for x in results)
 final=dict(status='COMPLETE',fit_subjects=25,diagnostic_validation_subjects=15,steps_cap=a.steps,
  gate=dict(train_accuracy_min=.96,train_ce_max=.15,fc_and_encoder_initial_gradient_min=1e-8,eval_repeat_max=1e-6,fc_perturbation_rms_min=1e-4),
  results=results,T_capacity_gate_passed=eligible,
  T_internal_cv_eligibility='pending repeated internal validation and stopping-pattern gate' if eligible else 'STOP_T_ROUTE',
  use_for_model_ranking=False,weights_reused=False)
 write(ROOT/'public/diagnostics/T_capacity_gate.public.json',final)
if __name__=='__main__':main()
