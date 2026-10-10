import copy,json,sys
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
sys.dont_write_bytecode=True;sys.path.insert(0,'/home/panyao/AD')
from BrainPromptPro import candidate_v3_runner as factory
from BrainPromptPro import candidate_runner as r
from BrainPromptPro.data.metadata import MetadataEncoder,record_id
from BrainPromptPro.data.brainnet_pro import FoldDataset
cfg,protocol,data,split=r.load_context();torch.set_num_threads(2)
enc=MetadataEncoder().fit([data.records[i] for i in split['train']],cfg['data']['site_column'],cfg['data']['missing_site'])
train=next(iter(DataLoader(FoldDataset(data,split['train'][:20],enc),batch_size=20)))
val=next(iter(DataLoader(FoldDataset(data,split['val'],enc),batch_size=100)))
models={};details={};base=None
for name in ['M3_v3','M3_v3_fixed','M3_v3_no_metadata','M3_v3_direct']:
 r.seed_all(42);model,c=r.build_model(name,cfg,data,enc,r.allowed_spec(name));models[name]=model
 if base is None:base=copy.deepcopy(model.base.state_dict())
 for k,v in base.items():
  if torch.is_tensor(v):torch.testing.assert_close(v,model.base.state_dict()[k],atol=0,rtol=0)
 model.train();o=r.forward_batch(model,train,'cpu');pop=model.population
 loss=F.cross_entropy(o['logits'],train['label']);loss.backward()
 gradients={}
 for group in ['metric','temperature','scorers.age','scorers.sex','scorers.site','fusion_logits','bottleneck','out','gain_raw']:
  gradients[group]=sum(float(p.grad.norm()) for k,p in pop.named_parameters() if k.startswith(group) and p.grad is not None)
 if name=='M3_v3':assert all(v>0 for v in gradients.values()),gradients
 assert pop.reference_ids==[record_id(data.records[i]) for i in split['train']]
 assert not any(p.requires_grad for k,p in pop.named_buffers())
 assert all(p.ndim<2 or p.shape[0]!=6670 for p in pop.parameters())
 for a in o['reference_attention'].values():
  torch.testing.assert_close(a.sum(-1),torch.ones(20),atol=1e-6,rtol=1e-6);assert ((a>0).sum(-1)==32).all()
  for i,s in enumerate(train['subject_id']):assert a[i,pop.reference_lookup[s]]==0
 assert o['module_residuals']['M3'].norm(dim=-1).max()<2
 model.eval()
 with torch.no_grad():
  expected=r.forward_batch(model,val,'cpu')['logits'];errors={}
  for bs in [1,7,20]:
   chunks=[]
   for start in range(0,100,bs):
    batch={k:v[start:start+bs] for k,v in val.items()};chunks.append(r.forward_batch(model,batch,'cpu')['logits'])
   diff=float((torch.cat(chunks)-expected).abs().max());assert diff<=1e-5,(name,bs,diff);errors[str(bs)]=diff
  reverse={k:(v.flip(0) if torch.is_tensor(v) else list(reversed(v))) for k,v in val.items()}
  diff=float((r.forward_batch(model,reverse,'cpu')['logits'].flip(0)-expected).abs().max());assert diff<=1e-5
  if name!='M3_v3_direct':
   # With zero reference values, changing current FC cannot produce any population value output.
   probe=copy.deepcopy(model);probe.population.reference_keys.zero_();z=r.forward_batch(probe,train,'cpu')['module_residuals']['M3'];assert torch.equal(z,torch.zeros_like(z))
 details[name]=dict(trainable_population_parameters=sum(p.numel() for p in pop.parameters() if p.requires_grad),all_population_parameters=sum(p.numel() for p in pop.parameters()),classification_gradients=gradients,batch_size_vs100_max_delta=errors,reverse_order_max_delta=diff,reference_count=len(pop.reference_ids),self_masked=True,actual_neighbors=32,zero_reference_values_give_zero_output=name!='M3_v3_direct')
# No-metadata and fixed share the same effective initial retrieval, output head and gain.
for name in ['M3_v3_fixed','M3_v3_no_metadata','M3_v3_direct']:
 for k in ['bottleneck.weight','out.weight','gain_raw']:
  torch.testing.assert_close(models['M3_v3'].population.state_dict()[k],models[name].population.state_dict()[k],atol=0,rtol=0)
r.dump(r.RUN/'verification/m3v3_contract_tests.json',dict(passed=True,models=details,test_loader_created=False,initialization_note='V3 nonzero small residual; common BP trunk and matched heads identical at fixed seed, not a claim of active-module zero residual'))
print(json.dumps(details,indent=2))
