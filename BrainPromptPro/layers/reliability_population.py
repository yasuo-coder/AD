"""P5: versioned TRAIN-only references with continuous reliability-aware edges.

Representation generation is deliberately separate and must bind its immutable
encoder version. Query IDs are used only for removing self references.
"""
import torch
from torch import nn
from torch.nn import functional as F


def stability_reliability(views,temperature=.05):
    if temperature<=0 or views.ndim!=3 or views.shape[1]<2:raise ValueError('Need [B,K>=2,D] and positive temperature')
    if not torch.isfinite(views).all():raise ValueError('Nonfinite views')
    x=F.normalize(views,dim=-1,eps=1e-8)
    variance=(x-x.mean(1,keepdim=True)).square().sum(-1).mean(1)
    return torch.exp(-variance/temperature),variance


class ReliabilityReferencePopulation(nn.Module):
    def __init__(self, reference, reliability, reference_ids, train_ids, encoder_version,
                 hidden_dim, topk=16, use_reliability=True, learn_edges=True):
        super().__init__()
        if not encoder_version:raise ValueError('Representation version is required')
        if len(set(reference_ids))!=len(reference_ids) or set(reference_ids)!=set(train_ids):
            raise ValueError('Reference membership must equal TRAIN exactly')
        if reference.ndim!=2 or len(reference)!=len(reference_ids) or reliability.shape!=(len(reference),):raise ValueError('Bank shapes mismatch')
        if not torch.isfinite(reference).all() or not torch.isfinite(reliability).all() or (reliability<0).any() or (reliability>1).any():raise ValueError('Invalid reference features/reliability')
        if not 1<=topk<len(reference):raise ValueError('Not enough train references for top-k and self mask')
        self.ids=list(reference_ids);self.lookup={s:i for i,s in enumerate(self.ids)};self.encoder_version=encoder_version
        self.topk=topk;self.use_reliability=use_reliability;self.learn_edges=learn_edges
        self.register_buffer('reference',reference.detach().clone());self.register_buffer('reliability',reliability.detach().clone())
        d=reference.shape[-1]
        self.metric=nn.Parameter(torch.zeros(d),requires_grad=learn_edges)
        self.scorer=nn.Sequential(nn.Linear(3,16),nn.GELU(),nn.Linear(16,1))
        self.value=nn.Linear(d,hidden_dim,bias=False)
        self.gain=nn.Parameter(torch.tensor(-2.))
        if not learn_edges:
            for p in self.scorer.parameters():p.requires_grad_(False)
    def get_extra_state(self):
        return dict(reference_ids=self.ids,encoder_version=self.encoder_version,topk=self.topk,use_reliability=self.use_reliability,learn_edges=self.learn_edges)
    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('Reference/version/config changed during reload')
    def forward(self,query,reliability,sample_ids,encoder_version):
        if encoder_version!=self.encoder_version:raise ValueError('Query/reference encoder versions differ')
        if query.shape[-1]!=self.reference.shape[-1] or reliability.shape!=(len(query),) or len(sample_ids)!=len(query):raise ValueError('Query shapes mismatch')
        if not torch.isfinite(query).all() or not torch.isfinite(reliability).all() or (reliability<0).any() or (reliability>1).any():raise ValueError('Invalid query/reliability')
        metric=1+.5*self.metric.tanh()
        q=F.normalize(query*metric,dim=-1)
        k=F.normalize(self.reference*metric,dim=-1)
        sim=q@k.T
        rq=reliability[:,None].expand_as(sim);rr=self.reliability[None].expand_as(sim)
        if not self.use_reliability:rq=torch.ones_like(rq);rr=torch.ones_like(rr)
        correction=self.scorer(torch.stack((sim,rq,rr),-1)).squeeze(-1).tanh() if self.learn_edges else torch.zeros_like(sim)
        # Reference reliability remains pair-specific under normalization; a
        # query-only scale would cancel and is therefore not used as a weight.
        score=sim/.5+correction
        if self.use_reliability:score=score+torch.log(rr.clamp_min(.05))
        allowed=torch.ones_like(score,dtype=torch.bool)
        for i,s in enumerate(sample_ids):
            j=self.lookup.get(s)
            if j is not None:allowed[i,j]=False
        score=score.masked_fill(~allowed,float('-inf'))
        nearest=torch.argsort(score,dim=-1,descending=True,stable=True)[:,:self.topk]
        support=torch.zeros_like(allowed).scatter_(1,nearest,True)
        attention=score.masked_fill(~support,float('-inf')).softmax(-1)
        uniform=allowed.to(query.dtype)/allowed.sum(-1,keepdim=True)
        message=(attention-uniform)@self.reference
        delta=self.value(message)
        delta=self.gain.sigmoid()*delta/(1+delta.square().sum(-1,keepdim=True)).sqrt()
        return dict(residual=delta,attention=attention,reliability=reliability,
                    entropy=-(attention*attention.clamp_min(1e-8).log()).sum(-1),
                    effective_neighbors=(attention>0).sum(-1),message=message,similarity=sim)
