"""Pre-reviewed M3 revision: fixed TRAIN values, learned sparse relations only."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class SparseReferencePopulation(nn.Module):
    def __init__(self, d, pca, metadata, ids, artifact_sha256, control='learned'):
        super().__init__()
        if control not in ('learned','fixed','no_metadata','direct'): raise ValueError(control)
        self.control=control;self.reference_ids=list(ids);self.reference_lookup={s:i for i,s in enumerate(ids)}
        self.artifact_sha256=artifact_sha256;self.n_neighbors=32
        for key in ('upper','pca_mean','pca_components','pca_scale','reference_keys'):
            self.register_buffer(key,pca[key].clone())
        for key in ('age','sex','site'):self.register_buffer('reference_'+key,metadata[key].clone())
        self.metric=nn.Parameter(torch.zeros(3,32));self.temperature=nn.Parameter(torch.full((3,),math.log(.5)))
        self.scorers=nn.ModuleDict({r:nn.Sequential(nn.Linear(4 if r=='sex' else 3,8),nn.GELU(),nn.Linear(8,1)) for r in ('age','sex','site')})
        for scorer in self.scorers.values():
            for layer in scorer:
                if isinstance(layer,nn.Linear):nn.init.normal_(layer.weight,std=.02);nn.init.zeros_(layer.bias)
        self.fusion_logits=nn.Parameter(torch.zeros(3))
        self.bottleneck=nn.Linear(32,8,bias=False);self.out=nn.Linear(8,d,bias=False);self.dropout=nn.Dropout(.2)
        nn.init.normal_(self.bottleneck.weight,std=.02);nn.init.normal_(self.out.weight,std=.02)
        self.gain_raw=nn.Parameter(torch.tensor(math.log(.05/.95)))
        if control in ('fixed','no_metadata','direct'):
            for p in self.scorers.parameters():nn.init.zeros_(p);p.requires_grad_(False)
        if control in ('fixed','direct'):
            self.metric.requires_grad_(False);self.temperature.requires_grad_(False);self.fusion_logits.requires_grad_(False)
    def get_extra_state(self):
        return dict(reference_ids=self.reference_ids,artifact_sha256=self.artifact_sha256,control=self.control)
    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('TRAIN PCA/reference/control provenance changed')
    def project(self,fc):
        x=fc[:,self.upper[0],self.upper[1]]
        return F.normalize(((x-self.pca_mean)@self.pca_components)*self.pca_scale,dim=-1,eps=1e-6).detach()
    def forward(self,h,fc,age,sex,site,sample_ids):
        if sample_ids is None:raise ValueError('Acquisition IDs required only for self mask')
        q=self.project(fc);k=self.reference_keys
        permitted=torch.ones(len(q),len(k),device=q.device,dtype=torch.bool)
        for i,s in enumerate(sample_ids):
            j=self.reference_lookup.get(s)
            if j is not None:permitted[i,j]=False
        if int(permitted.sum(-1).min())<self.n_neighbors:raise ValueError('Insufficient TRAIN references')
        uniform=permitted.to(q.dtype)/permitted.sum(-1,keepdim=True)
        age_delta=(age[:,None]-self.reference_age[None]).abs().clamp(0,4)
        age_mean=((age[:,None]+self.reference_age[None])*.5).clamp(-3,3)
        meta={'age':torch.stack((age_delta,age_delta.square()/4,age_mean),-1)}
        x=sex[:,None];y=self.reference_sex[None]
        meta['sex']=torch.stack(((x==1)&(y==1),(x==2)&(y==2),((x==1)&(y==2))|((x==2)&(y==1)),(x==0)|(y==0)),-1).to(q.dtype)
        x=site[:,None];y=self.reference_site[None];known=(x!=0)&(y!=0)
        meta['site']=torch.stack((known&(x==y),known&(x!=y),~known),-1).to(q.dtype)
        difference=(q[:,None]-k[None]).square();attention={};messages=[];corrections={}
        for index,name in enumerate(('age','sex','site')):
            weights=1+.5*self.metric[index].tanh();tau=.25+.75*self.temperature[index].sigmoid()
            correction=.5*self.scorers[name](meta[name]).squeeze(-1).tanh();corrections[name]=correction
            score=-(difference*weights).sum(-1)/tau+correction
            allowed=score.masked_fill(~permitted,float('-inf'))
            # Stable support tie handling follows frozen reference order, never numerical ID features.
            nearest=torch.argsort(allowed,dim=-1,descending=True,stable=True)[:,:self.n_neighbors]
            support=torch.zeros_like(permitted).scatter_(1,nearest,True)
            a=allowed.masked_fill(~support,float('-inf')).softmax(-1)
            attention[name]=a;messages.append((a-uniform)@k)
        mixture=self.fusion_logits.softmax(-1);message=(torch.stack(messages,1)*mixture[None,:,None]).sum(1)
        if self.control=='direct':message=q
        u=self.out(self.dropout(self.bottleneck(message)));gain=2*self.gain_raw.sigmoid()
        delta=gain*u/(1+u.square().sum(-1,keepdim=True)).sqrt()
        self.last_diagnostics=dict(metadata_corrections=corrections,messages=torch.stack(messages,1),uniform=uniform,gain=gain,message=message)
        # L1 of this simplex adjacency is constant: top-k, not the shared L1 coefficient, supplies sparsity.
        return delta,attention,mixture[None].expand(len(q),-1),attention
