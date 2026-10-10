import copy,sys,json
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
sys.dont_write_bytecode=True;sys.path.insert(0,'/home/panyao/AD')
from BrainPromptPro import candidate_v3_runner
from BrainPromptPro import candidate_runner as r
from BrainPromptPro.data.metadata import MetadataEncoder
from BrainPromptPro.data.brainnet_pro import FoldDataset
cfg,p,data,split=r.load_context();torch.set_num_threads(2);enc=MetadataEncoder().fit([data.records[i] for i in split['train']],cfg['data']['site_column'],cfg['data']['missing_site']);r.seed_all(42);model,c=r.build_model('M3_v3',cfg,data,enc,r.allowed_spec('M3_v3'));b=next(iter(DataLoader(FoldDataset(data,split['train'][:20],enc),batch_size=20)));report={}
for variant in ['review_initial_small','xavier_gain1']:
 m=copy.deepcopy(model)
 if variant=='xavier_gain1':
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(42)
   for sub in m.population.scorers.values():
    for layer in sub:
     if isinstance(layer,nn.Linear):nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
   nn.init.xavier_uniform_(m.population.bottleneck.weight);nn.init.xavier_uniform_(m.population.out.weight);nn.init.zeros_(m.population.gain_raw)
 m.train();r.seed_all(42);o=r.forward_batch(m,b,'cpu');loss=F.cross_entropy(o['logits'],b['label']);loss.backward();groups={}
 for group in ['scorers','metric','temperature','fusion_logits','bottleneck','out','gain_raw']:
  params=[p for k,p in m.population.named_parameters() if k.startswith(group)];g=sum(float(p.grad.square().sum()) for p in params if p.grad is not None)**.5;decay=.005*sum(float(p.detach().square().sum()) for p in params)**.5;groups[group]=dict(classification_gradient_l2=g,weight_decay_gradient_l2=decay)
 report[variant]=dict(population_residual_norm_mean=float(o['module_residuals']['M3'].detach().norm(dim=-1).mean()),groups=groups)
r.dump(r.RUN/'verification/m3v3_initialization_scale_probe.json',dict(role='TRAIN first20 only, before any V3 training, no validation selection',variants=report));print(json.dumps(report,indent=2))
