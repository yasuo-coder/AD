"""Post-training train/validation mechanism probes; never instantiate test loader."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
sys.dont_write_bytecode=True;sys.path.insert(0,'/home/panyao/AD')
from BrainPromptPro import candidate_runner as r
from BrainPromptPro.data.metadata import MetadataEncoder
from BrainPromptPro.data.brainnet_pro import FoldDataset,sha256
p=argparse.ArgumentParser();p.add_argument('--models',nargs='+');p.add_argument('--seeds',nargs='+',type=int,default=[42,3407,2026]);a=p.parse_args()
torch.set_num_threads(2);cfg,protocol,data,split=r.load_context()
def stats(x):
 x=x.detach().float().cpu();return dict(mean=float(x.mean()),std=float(x.std(unbiased=False)),min=float(x.min()),max=float(x.max()))
def similarity(x,y):
 x=F.normalize(x.float().flatten(1),dim=-1);s=x@x.T;n=len(s);mask=~torch.eye(n,dtype=torch.bool);same=y[:,None]==y[None];return dict(within_class_mean=float(s[mask&same].mean()),between_class_mean=float(s[mask&~same].mean()),across_sample_feature_std=float(x.std(0,unbiased=False).mean()))
def entropy(x):return float((-(x*x.clamp_min(1e-30).log()).sum(-1)/math.log(x.shape[-1])).mean())
for name in a.models or list(json.loads((r.RUN/'registry.json').read_text())):
 for seed in a.seeds:
  out=r.experiment_path(name,'sqrt_inverse_frequency',seed)
  if not (out/'result.json').exists():continue
  ck=torch.load(out/'best_validation.pt',map_location='cpu',weights_only=True)
  assert all(sha256(f)==h for f,h in ck['source_sha256'].items())
  enc=MetadataEncoder.from_state_dict(ck['metadata_encoder']);r.seed_all(seed);model,c=r.build_model(name,ck['config'],data,enc,ck['spec']);model.load_state_dict(ck['model']);model.eval()
  vals=DataLoader(FoldDataset(data,split['val'],enc),batch_size=20,shuffle=False)
  collect={};ys=[];ids=[];logits=[];residuals={};refatt={};rawrel={}
  with torch.no_grad():
   for batch in vals:
    o=r.forward_batch(model,batch,'cpu');ys.append(batch['label']);ids+=batch['subject_id'];logits.append(o['logits'])
    for key in ['roi_features','roi_attention','disease_tokens','subject_repr','fused_repr','edge_gates','relation_weights']:
     if o.get(key) is not None:collect.setdefault(key,[]).append(o[key].cpu())
    if o.get('signed_adjacencies') is not None:
     collect.setdefault('signed_graph',[]).append((o['signed_adjacencies']['positive']-o['signed_adjacencies']['negative']).cpu())
    for key,x in o.get('module_residuals',{}).items():residuals.setdefault(key,[]).append(x.cpu())
    for key,x in (o.get('reference_attention') or {}).items():refatt.setdefault(key,[]).append(x.cpu())
    for key,x in o.get('population_adjacencies',{}).items():rawrel.setdefault(key,[]).append(x.cpu())
  collect={k:torch.cat(v) for k,v in collect.items()};y=torch.cat(ys);logit=torch.cat(logits);diag=dict(model=name,seed=seed,checkpoint_epoch=ck['epoch']+1,checkpoint_sha256=sha256(out/'best_validation.pt'),role='validation',sample_count=len(y),test_accessed=False,interpretation='Descriptive association, not causal disease biomarkers or clinical validation')
  if 'roi_features' in collect:
   h=F.normalize(collect['roi_features'],dim=-1);cos=h@h.transpose(1,2);mask=~torch.eye(h.shape[1],dtype=torch.bool)
   diag['roi_cosine_offdiagonal']=stats(cos[:,mask])
  for key in ['subject_repr','fused_repr']:
   if key in collect:diag[key]=dict(norm=stats(collect[key].norm(dim=-1)),similarity=similarity(collect[key],y))
  for key,values in residuals.items():diag['residual_'+key]=stats(torch.cat(values).norm(dim=-1))
  if 'edge_gates' in collect:
   gates=collect['edge_gates'];graph=collect['signed_graph'];mask=~torch.eye(graph.shape[-1],dtype=torch.bool)
   diag['M1']=dict(gates=stats(gates[:,mask]),fraction_gate_below_001=float((gates[:,mask]<.01).float().mean()),fraction_gate_above_099=float((gates[:,mask]>.99).float().mean()),across_patient_gate_std=float(gates[:,mask].std(0,unbiased=False).mean()),nonzero_graph_fraction=float((graph[:,mask]!=0).float().mean()),graph_similarity=similarity(graph[:,mask],y),per_class_graph_mean_norm=[float(graph[y==i].mean(0).norm()) for i in range(5)])
  if 'roi_attention' in collect:
   att=collect['roi_attention'];norm=F.normalize(att,dim=-1);cross=norm@norm.transpose(1,2);mask=~torch.eye(5,dtype=torch.bool)
   diag['M2']=dict(normalized_entropy=entropy(att),disease_attention_cosine=stats(cross[:,mask]),per_query_entropy=[entropy(att[:,i]) for i in range(5)],attention_sample_similarity=similarity(att,y),top10_roi_indices_by_query=att.mean(0).topk(10,-1).indices.tolist(),trueclass_query_roi_attention=[att[y==i].mean(0).tolist() for i in range(5)])
  if refatt:
   refatt={k:torch.cat(v) for k,v in refatt.items()};rawrel={k:torch.cat(v) for k,v in rawrel.items()};names=list(refatt)
   diag['M3']=dict(reference_count=len(model.population.reference_ids),reference_indices_exact_train=ck['training_indices_used']==split['train'],attention={k:dict(normalized_entropy=entropy(v),max_weight=stats(v.max(-1).values),sample_similarity=similarity(v,y)) for k,v in refatt.items()},raw_sigmoid={k:stats(v) for k,v in rawrel.items()},relation_cosines={names[i]+'_'+names[j]:float(F.cosine_similarity(refatt[names[i]],refatt[names[j]],dim=-1).mean()) for i in range(3) for j in range(i+1,3)},fusion_weights=stats(collect['relation_weights']))
   # Repartition exact validation queries: no query-query population edges.
   with torch.no_grad():
    alternative=torch.cat([r.forward_batch(model,b,'cpu')['logits'] for b in DataLoader(FoldDataset(data,split['val'],enc),batch_size=7,shuffle=False)])
   diag['M3']['batch20_vs7_max_logit_delta']=float((alternative-logit).abs().max())
  # Fixed TRAIN probe only. Both losses use actual training class weights, eval dropout disabled.
  batch=next(iter(DataLoader(FoldDataset(data,split['train'][:20],enc),batch_size=20)))
  model.zero_grad();o=r.forward_batch(model,batch,'cpu');main=F.cross_entropy(o['logits'],batch['label'],weight=ck['class_weights']);aux=F.cross_entropy(o['disease_logits'],batch['label'],weight=ck['class_weights'])
  named=[(k,v) for k,v in model.named_parameters() if v.requires_grad];params=[v for k,v in named]
  g=torch.autograd.grad(main,params,retain_graph=True,allow_unused=True);ag=torch.autograd.grad(aux,params,allow_unused=True)
  groups={'trunk':'base.original.layers','M1':'adaptive.','M2':'disease_readout.','M3':'population.','M4':'slots.','M5':'covariance.'};gd={}
  for group,prefix in groups.items():
   idx=[i for i,(k,v) in enumerate(named) if k.startswith(prefix)]
   if not idx:continue
   v=torch.cat([(torch.zeros_like(params[i]) if g[i] is None else g[i]).flatten() for i in idx]);w=torch.cat([(torch.zeros_like(params[i]) if ag[i] is None else ag[i]).flatten() for i in idx])
   gd[group]=dict(weighted_classification_gradient_norm=float(v.norm()),weighted_auxiliary_gradient_norm=float(w.norm()),cosine=float(F.cosine_similarity(v[None],w[None],dim=-1)))
  diag['gradient_probe']=dict(role='first 20 fixed training samples',training_dropout=False,class_weighting='sqrt_inverse_frequency',classification_and_auxiliary=gd)
  if refatt:
   diag['M3']['training_self_reference_attention_max']=max(float(v[i,model.population.reference_lookup[s]]) for v in o['reference_attention'].values() for i,s in enumerate(batch['subject_id']))
  dst=out/'diagnostics';dst.mkdir(exist_ok=True);r.dump(dst/'module_diagnostics.json',diag)
  # Compact relevant numerical structures for reproducible plotting.
  keep={k:v for k,v in collect.items() if k not in ['roi_features','edge_gates']};keep.update(truth=y,sample_ids=ids)
  torch.save(keep,dst/'validation_structure.pt')
  print(json.dumps(dict(model=name,seed=seed,diagnostics='complete')),flush=True)
