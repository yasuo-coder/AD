import copy,json,math,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
sys.dont_write_bytecode=True;sys.path.insert(0,'/home/panyao/AD')
from BrainPromptPro import candidate_v3_initialized_runner
from BrainPromptPro import candidate_runner as r
from BrainPromptPro.data.metadata import MetadataEncoder
from BrainPromptPro.data.brainnet_pro import FoldDataset,sha256
cfg,protocol,data,split=r.load_context();torch.set_num_threads(2)
for name in ['M3_v3','M3_v3_fixed','M3_v3_no_metadata','M3_v3_direct']:
 for seed in [42,3407,2026]:
  path=r.experiment_path(name,'sqrt_inverse_frequency',seed)
  if not (path/'result.json').exists():continue
  if (path/'diagnostics/m3v3_mechanism.json').exists():continue
  ck=torch.load(path/'best_validation.pt',map_location='cpu',weights_only=True);enc=MetadataEncoder.from_state_dict(ck['metadata_encoder']);r.seed_all(seed);model,c=r.build_model(name,ck['config'],data,enc,ck['spec']);model.load_state_dict(ck['model']);model.eval();pop=model.population
  store={k:[] for k in ['age','sex','site']};messages=[];resnorm=[];fusednorm=[];meta={k:[] for k in store}
  loader=DataLoader(FoldDataset(data,split['val'],enc),batch_size=20,shuffle=False)
  with torch.no_grad():
   for batch in loader:
    out=r.forward_batch(model,batch,'cpu');diag=pop.last_diagnostics
    for key,a in out['reference_attention'].items():store[key].append(a.cpu());meta[key].append(diag['metadata_corrections'][key].cpu())
    messages.append(diag['messages'].cpu());resnorm.append(out['module_residuals']['M3'].norm(dim=-1));fusednorm.append(out['fused_repr'].norm(dim=-1))
  store={k:torch.cat(v) for k,v in store.items()};meta={k:torch.cat(v) for k,v in meta.items()};msg=torch.cat(messages);names=list(store);rel={}
  for key,a in store.items():
   entropy=-(a*a.clamp_min(1e-30).log()).sum(-1)
   rel[key]=dict(nnz_min=int((a>0).sum(-1).min()),nnz_max=int((a>0).sum(-1).max()),entropy_over_log32=float(entropy.mean()/math.log(32)),entropy_over_log798=float(entropy.mean()/math.log(798)),effective_neighbors_mean=float((1/a.square().sum(-1)).mean()),top1_mean=float(a.max(-1).values.mean()),metadata_correction_mean=float(meta[key].mean()),metadata_correction_std=float(meta[key].std(unbiased=False)))
  cross={}
  for i in range(3):
   for j in range(i+1,3):
    same=(store[names[i]]>0)&(store[names[j]]>0);cross[names[i]+'_'+names[j]]=dict(support_overlap_fraction=float(same.sum(-1).float().mean()/32),attention_cosine=float(F.cosine_similarity(store[names[i]],store[names[j]],dim=-1).mean()),message_cosine=float(F.cosine_similarity(msg[:,i],msg[:,j],dim=-1).mean()))
  # Main classification gradients use actual TRAIN class weights; no stochastic dropout in diagnostic.
  b=next(iter(DataLoader(FoldDataset(data,split['train'][:20],enc),batch_size=20)));model.zero_grad();out=r.forward_batch(model,b,'cpu');loss=F.cross_entropy(out['logits'],b['label'],weight=ck['class_weights']);loss.backward();grad={}
  for group in ['metric','temperature','scorers.age','scorers.sex','scorers.site','fusion_logits','bottleneck','out','gain_raw']:
   ps=[p for k,p in pop.named_parameters() if k.startswith(group)];grad[group]=dict(classification_norm=sum(float(p.grad.square().sum()) for p in ps if p.grad is not None)**.5,weight_decay_norm=.005*sum(float(p.detach().square().sum()) for p in ps if p.requires_grad)**.5)
  result=json.loads((path/'result.json').read_text());answer=dict(model=name,seed=seed,checkpoint_sha256=sha256(path/'best_validation.pt'),role='TRAIN gradient / VALIDATION mechanism only; no test',selected_epoch=ck['epoch']+1,train_f1=result['train']['f1_macro'],validation_f1=result['val']['f1_macro'],relations=rel,cross_relations=cross,mixture=pop.fusion_logits.softmax(-1).tolist(),gain=float(2*pop.gain_raw.sigmoid()),metric_weights=(1+.5*pop.metric.tanh()).tolist(),temperatures=(.25+.75*pop.temperature.sigmoid()).tolist(),residual_norm_mean=float(torch.cat(resnorm).mean()),residual_to_fused_norm_ratio=float((torch.cat(resnorm)/torch.cat(fusednorm)).mean()),gradients=grad,trainable_population_parameters=sum(p.numel() for p in pop.parameters() if p.requires_grad),interpretation='Forced top32 entropy reduction is not evidence that learned relations improve disease prediction; controls must be compared')
  r.dump(path/'diagnostics/m3v3_mechanism.json',answer);print(name,seed,'complete',flush=True)
