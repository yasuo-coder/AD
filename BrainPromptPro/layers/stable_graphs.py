"""Opt-in continuous graph priors and normalized semantic readout.

The original modules and checkpoint contract remain available unchanged.
All disease queries are applied to all subjects; no labels enter this module.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .adaptive_brain_graph import SemanticGuidedAdaptiveBrainGraph
from .disease_aware_readout import DiseaseAwareSemanticReadout
from .multirelation_population_graph import MultiRelationPopulationGraph, symmetric_pair
from .gcn_layer import symmetric_normalize, unit_diagonal


class LayerNormalizedGCN(nn.Module):
    def __init__(self, dim, dropout=0., residual=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.weight)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, h, adjacency):
        message = symmetric_normalize(adjacency) @ (h @ self.weight) + self.bias
        message = self.dropout(F.gelu(self.norm(message)))
        return h + message if self.residual else message


class StableAdaptiveBrainGraph(SemanticGuidedAdaptiveBrainGraph):
    def __init__(self, *args, hidden_dim, **kwargs):
        super().__init__(*args, hidden_dim=hidden_dim, **kwargs)
        self.feature_norm = nn.LayerNorm(hidden_dim)
        self.semantic_norm = nn.LayerNorm(hidden_dim)
        self.semantic_gate = nn.Parameter(torch.tensor(math.log(.1/.9)))

    def forward(self, fc, roi_prompt, features=None):
        # fc: [B,N,N]; the full FC row remains on an explicit residual path.
        x = self.feature_norm(self.feature_projection(fc if features is None else features))
        r = self.semantic_norm(self.roi_projection(roi_prompt)).unsqueeze(0).expand(fc.shape[0],-1,-1)
        h = x + self.semantic_gate.sigmoid() * self.node_mlp(torch.cat((x,r),dim=-1))
        chunks=[]
        for start in range(0,fc.shape[-1],self.pair_chunk_size):
            left=h[:,start:start+self.pair_chunk_size,None,:];right=h[:,None,:,:]
            c=fc[:,start:start+self.pair_chunk_size,:]
            score=(checkpoint(self._score_chunk,left,right,c,use_reentrant=False)
                   if self.training and torch.is_grad_enabled() else self._score_chunk(left,right,c))
            # A continuous subject-specific FC prior, never a hard threshold.
            chunks.append(c.abs().square() * (.25 + .75 * score))
        a=torch.cat(chunks,dim=1)
        return unit_diagonal((a+a.transpose(-1,-2))*.5),h


class StableDiseaseReadout(DiseaseAwareSemanticReadout):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.log_scale=nn.Parameter(torch.tensor(math.log(5.)))
        self.readout_gate=nn.Parameter(torch.tensor(math.log(.1/.9)))

    def forward(self,h,disease_prompt):
        # h: [B,N,D]; disease_prompt: [K,P]; every K query sees every subject.
        q=self.q_projection(disease_prompt);k=self.k_projection(h);v=self.v_projection(h)
        scale=self.log_scale.exp().clamp(1.,20.)
        attention=(torch.einsum('kd,bnd->bkn',F.normalize(q,dim=-1),F.normalize(k,dim=-1))*scale).softmax(-1)
        representations=attention@v
        r=F.normalize(self.representation_projection(representations),dim=-1,eps=1e-8)
        d=F.normalize(self.disease_projection(q),dim=-1,eps=1e-8)
        logits=(r*d.unsqueeze(0)).sum(-1)*scale
        semantic=(logits.softmax(-1).unsqueeze(-1)*representations).sum(1)
        # Preserve individual ROI features when disease queries are similar.
        subject=h.mean(1)+self.readout_gate.sigmoid()*semantic
        return subject,logits,attention


class StablePopulationGraph(MultiRelationPopulationGraph):
    def __init__(self,hidden_dim,*args,**kwargs):
        super().__init__(hidden_dim,*args,**kwargs)
        # Return messages only: the subject residual is added once by fusion.
        self.gnns=nn.ModuleDict({r:LayerNormalizedGCN(hidden_dim,residual=False) for r in self.relations})

    def forward(self,subject_repr,age,sex,site):
        pair=symmetric_pair(subject_repr)
        delta=(age[:,None]-age[None,:]).abs()
        metadata={'age':torch.stack((delta,age[:,None]*age[None,:],(age[:,None]+age[None,:])*.5),-1)}
        priors={'age':.25+.75*(-delta).exp()}
        for name,ids,embedding in [('sex',sex,self.sex_embedding),('site',site,self.site_embedding)]:
            if name in self.relations:
                same=(ids[:,None]==ids[None,:])&(ids[:,None]!=0)&(ids[None,:]!=0)
                metadata[name]=torch.cat((symmetric_pair(embedding(ids)),same.to(subject_repr.dtype).unsqueeze(-1)),-1)
                priors[name]=.25+.75*same.to(subject_repr.dtype)
        adjacencies={};outputs=[]
        for name in self.relations:
            score=self.scorers[name](torch.cat((pair,metadata[name]),-1)).squeeze(-1).sigmoid()
            a=unit_diagonal((score+score.T)*.5*priors[name])
            adjacencies[name]=a;outputs.append(self.gnns[name](subject_repr,a))
        weights=(self.relation_gate.softmax(-1).expand(len(subject_repr),-1) if self.gate_mode=='global'
                 else self.relation_gate(torch.cat((subject_repr,*outputs),-1)).softmax(-1))
        return (torch.stack(outputs,1)*weights.unsqueeze(-1)).sum(1),adjacencies,weights
