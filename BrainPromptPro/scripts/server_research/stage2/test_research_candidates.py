import copy,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
sys.dont_write_bytecode=True;sys.path.insert(0,'/home/panyao/AD')
from BrainPromptPro.candidate_runner import load_context,build_model,forward_batch,seed_all,RUN,dump
from BrainPromptPro.nets.original_adapter import OriginalBrainPromptAdapter
from BrainPromptPro.data.metadata import MetadataEncoder,record_id
from BrainPromptPro.data.brainnet_pro import FoldDataset
cfg,protocol,data,split=load_context();torch.set_num_threads(2)
enc=MetadataEncoder().fit([data.records[i] for i in split['train']],cfg['data']['site_column'],cfg['data']['missing_site'])
b=next(iter(DataLoader(FoldDataset(data,split['train'][:20],enc),batch_size=20)))
seed_all(42);official=OriginalBrainPromptAdapter(cfg,data.roi_prompt,data.disease_prompt)
seed_all(42);control,_=build_model('BP_control',cfg,data,enc,{'modules':[]})
seed_all(42);candidate,ccfg=build_model('all',cfg,data,enc,{'modules':['M1','M2','M3','M4','M5']})
for name,v in official.state_dict().items():
 if torch.is_tensor(v):
  torch.testing.assert_close(v,control.base.state_dict()[name],atol=0,rtol=0)
  torch.testing.assert_close(v,candidate.base.state_dict()[name],atol=0,rtol=0)
from BrainPromptPro.train_TUs_graph_classification_llm import model_inputs
for training in [True,False]:
 official.train(training);control.train(training);candidate.train(training)
 torch.manual_seed(812);o=official(**model_inputs(b,'cpu'))
 torch.manual_seed(812);c=forward_batch(control,b,'cpu')
 torch.manual_seed(812);n=forward_batch(candidate,b,'cpu')
 for key in ['logits','disease_logits','fused_repr']:
  torch.testing.assert_close(o[key],c[key],atol=0,rtol=0)
  torch.testing.assert_close(o[key],n[key],atol=1e-6,rtol=1e-6)
 # Zero-residual candidates preserve common classification gradients.
 official.zero_grad();candidate.zero_grad()
 F.cross_entropy(o['logits'],b['label']).backward()
 F.cross_entropy(n['logits'],b['label']).backward()
 for name,p in official.named_parameters():
  q=dict(candidate.base.named_parameters())[name]
  if p.grad is None:assert q.grad is None,name
  else:torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=1e-5)
print('BP all-off exact and initial residual outputs/common gradients match in train/eval',flush=True)
assert candidate.population.reference_ids==[record_id(data.records[i]) for i in split['train']]
assert not set(candidate.population.reference_ids)&{record_id(data.records[i]) for i in split['val']+split['test']}
opt=torch.optim.Adam(candidate.parameters(),lr=5e-5)
candidate.train();grad={}
for step in range(3):
 opt.zero_grad();o=forward_batch(candidate,b,'cpu');loss=F.cross_entropy(o['logits'],b['label']);loss.backward()
 for family,token in [('M1','adaptive.edge'),('M2','disease_readout.q'),('M3','population.scorers'),('M4','slots.proj'),('M5','covariance.proj')]:
  grad[family]=max(grad.get(family,0.),sum(float(p.grad.abs().sum()) for name,p in candidate.named_parameters() if token in name and p.grad is not None))
 opt.step()
assert all(v>0 for v in grad.values()),grad
for a in o['reference_attention'].values():
 torch.testing.assert_close(a.sum(-1),torch.ones(20),atol=1e-6,rtol=1e-6)
 for i,s in enumerate(b['subject_id']):assert a[i,candidate.population.reference_lookup[s]]==0
candidate.eval()
with torch.no_grad():
 p=forward_batch(candidate,b,'cpu')['logits']
 pieces=[]
 for start in range(0,20,7):
  part={k:(v[start:start+7] if torch.is_tensor(v) or isinstance(v,(list,tuple)) else v) for k,v in b.items()}
  pieces.append(forward_batch(candidate,part,'cpu')['logits'])
 q=torch.cat(pieces);torch.testing.assert_close(p,q,atol=2e-5,rtol=1e-5)
print('Reference membership/self-mask, module gradients, finite outputs, batch-independent inference PASS',flush=True)
out=dict(passed=True,all_off_exact_train_eval=True,zero_residual_initial_output_and_common_gradient_equivalence=True,train_reference_only=True,self_reference_masked=True,classification_gradient_after_three_steps=grad,max_batch_size_logit_difference=float((p-q).abs().max()),test_loader_created=False)
dump(RUN/'verification/candidate_contract_tests.json',out)
print(json.dumps(out,indent=2))
