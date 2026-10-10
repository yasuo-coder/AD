"""Reuse actual historical controls under the new seven-batch TRAIN protocol."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from .research_candidates_v1 import ResearchBrainPrompt
from .research_population_v3 import SparseReferencePopulation
from ..research_runner import build_model as historical_baseline
from ..data.metadata import record_id


def build_control(kind,cfg,data,encoder,train_indices):
    if kind=='legacy':
        model,_=historical_baseline('CurrentPro_legacy',cfg,data,encoder,{})
    elif kind in ('M2','M3v3','M3v3_no_metadata'):
        model=ResearchBrainPrompt(cfg,data,encoder,dict(modules=['M2'] if kind=='M2' else ['M3']),train_indices)
        if kind!='M2':
            # Same deterministic TRAIN-only PCA and initialized V3 operator as
            # historical study, recomputed on the NEW TRAIN population.
            upper=torch.triu_indices(data.node_num,data.node_num,1)
            raw=data.fc[train_indices][:,upper[0],upper[1]].contiguous()
            x=raw.double();mu=x.mean(0);center=x-mu
            values,left=torch.linalg.eigh(center@center.T/(len(x)-1));idx=torch.argsort(values,descending=True)[:32]
            eigen=values[idx];left=left[:,idx]
            if not torch.isfinite(eigen).all() or float(eigen[-1])<=1e-8:raise ValueError('Degenerate TRAIN PCA')
            components=center.T@left/((len(x)-1)*eigen).sqrt()[None]
            pivots=components.abs().argmax(0);components*=components[pivots,torch.arange(32)].sign()[None]
            scale=(eigen+max(1e-8,1e-4*float(eigen[0]))).rsqrt()
            pca=dict(upper=upper,pca_mean=mu.float(),pca_components=components.float(),pca_scale=scale.float(),reference_keys=F.normalize((center@components)*scale,dim=-1,eps=1e-6).float())
            digest=hashlib.sha256(b''.join(pca[k].numpy().tobytes() for k in sorted(pca))).hexdigest()
            records=[data.records[i] for i in train_indices];ids=[record_id(r) for r in records]
            with torch.random.fork_rng(devices=[]):
                model.population=SparseReferencePopulation(cfg['net_params']['hidden_dim'],pca,encoder.transform(records),ids,digest,'no_metadata' if kind.endswith('no_metadata') else 'learned')
                for scorer in model.population.scorers.values():
                    for layer in scorer:
                        if isinstance(layer,nn.Linear):nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
                nn.init.xavier_uniform_(model.population.bottleneck.weight);nn.init.xavier_uniform_(model.population.out.weight);nn.init.zeros_(model.population.gain_raw)
                if kind.endswith('no_metadata'):
                    for p in model.population.scorers.parameters():nn.init.zeros_(p)
    else:raise ValueError('Unregistered historical control')
    model.use_p2=False;model.use_p3=False;model.use_p5=False
    return model
