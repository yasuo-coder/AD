"""Five explicitly reviewed numerical candidates around the unchanged BP trunk.

No spectral/spatial text embeddings; no labels or source partitions enter forward.
M4/M5 are generic readout controls, not automatically publishable innovations.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from .original_adapter import OriginalBrainPromptAdapter
from ..layers.gcn_layer import legacy_brain_adjacency
from ..data.metadata import record_id


def zero_linear(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None: nn.init.zeros_(module.bias)
    return module


def normalize_nonnegative(a):
    inv=a.sum(-1).clamp_min(1e-8).rsqrt()
    return inv.unsqueeze(-1)*a*inv.unsqueeze(-2)


class SignedConditionalMessages(nn.Module):
    def __init__(self,d,p,n,layers):
        super().__init__();self.fc_encoder=nn.Linear(n,24);self.roi_encoder=nn.Linear(p,24)
        self.edge=nn.Sequential(nn.Linear(50,32),nn.GELU(),nn.Linear(32,2))
        self.pos=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(layers)])
        self.neg=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(layers)])
        self.norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.out=nn.ModuleList([zero_linear(nn.Linear(d,d,bias=False)) for _ in range(layers)])
    def graph(self,fc,prompt):
        z=F.gelu(self.fc_encoder(fc)+self.roi_encoder(prompt)[None])
        z=F.normalize(z,dim=-1)
        pair=torch.cat(((z[:,:,None]-z[:,None,:]).abs(),z[:,:,None]*z[:,None,:],fc[...,None],fc.abs()[...,None]),-1)
        score=self.edge(pair);score=(score+score.transpose(1,2))*.5
        gates=.25+.75*score.sigmoid()
        mask=1-torch.eye(fc.shape[-1],device=fc.device,dtype=fc.dtype)
        positive=fc.clamp_min(0)*gates[...,0]*mask
        negative=(-fc).clamp_min(0)*gates[...,1]*mask
        return positive,negative,gates
    def message(self,h,pos,neg,i):
        # Separate nonnegative degree systems; anti-correlations never form negative degrees.
        m=normalize_nonnegative(pos)@self.pos[i](h)-normalize_nonnegative(neg)@self.neg[i](h)
        return self.out[i](F.gelu(self.norm[i](m)))


class DiseaseTokenReadout(nn.Module):
    def __init__(self,d,p,k):
        super().__init__();self.q=nn.Linear(p,32,bias=False);self.key=nn.Linear(d,32,bias=False)
        self.value=nn.Linear(d,d,bias=False);self.out=zero_linear(nn.Linear(k*d,d,bias=False))
        self.norm=nn.LayerNorm(d)
    def forward(self,h,prompt):
        q=F.normalize(self.q(prompt-prompt.mean(0,keepdim=True)),dim=-1)
        centered=h-h.mean(1,keepdim=True)
        keys=F.normalize(self.key(centered),dim=-1)
        att=(4.*torch.einsum('kd,bnd->bkn',q,keys)).softmax(-1)
        tokens=att@self.value(centered)
        return self.out(self.norm(tokens).flatten(1)),att,tokens


class OrderedSlots(nn.Module):
    def __init__(self,d,n):
        super().__init__();self.proj=nn.Linear(d,8);self.norm=nn.LayerNorm(8)
        self.out=zero_linear(nn.Linear(n*8,d,bias=False))
    def forward(self,h):return self.out(self.norm(self.proj(h)).flatten(1))


class CovarianceReadout(nn.Module):
    def __init__(self,d):
        super().__init__();self.proj=nn.Linear(d,16,bias=False)
        self.register_buffer('upper',torch.triu_indices(16,16))
        self.out=zero_linear(nn.Linear(136,d,bias=False))
    def forward(self,h):
        x=self.proj(h);x=x-x.mean(1,keepdim=True)
        covariance=x.transpose(1,2)@x/max(h.shape[1]-1,1)
        # Per-patient scale, never a statistic fitted across validation/test people.
        covariance=covariance/(covariance.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]+1e-4)
        return self.out(covariance[:,self.upper[0],self.upper[1]])


class ReferencePopulation(nn.Module):
    def __init__(self,d,n,num_sites,fc,metadata,ids):
        super().__init__();self.reference_ids=list(ids);self.reference_lookup={s:i for i,s in enumerate(ids)}
        self.register_buffer('upper',torch.triu_indices(n,n,offset=1))
        self.register_buffer('reference_fc',fc[:,self.upper[0],self.upper[1]].clone())
        for name in ['age','sex','site']:self.register_buffer('reference_'+name,metadata[name].clone())
        self.fc_projection=nn.Linear(self.upper.shape[1],32,bias=False)
        self.query_projection=nn.Linear(d,32,bias=False)
        self.sex_embedding=nn.Embedding(3,4,padding_idx=0)
        self.site_embedding=nn.Embedding(num_sites,4,padding_idx=0)
        self.scorers=nn.ModuleDict({r:nn.Sequential(nn.Linear(64+(3 if r=='age' else 9),32),nn.GELU(),nn.Linear(32,1)) for r in ['age','sex','site']})
        self.values=nn.ModuleDict({r:nn.Linear(32,d,bias=False) for r in ['age','sex','site']})
        self.gate=nn.Linear(d,3);self.norm=nn.LayerNorm(d)
        self.out=zero_linear(nn.Linear(d,d,bias=False))
    def get_extra_state(self):return {'reference_ids':self.reference_ids}
    def set_extra_state(self,state):
        if state['reference_ids']!=self.reference_ids:raise ValueError('Reference membership changed')
    def forward(self,h,fc,age,sex,site,sample_ids):
        if sample_ids is None:raise ValueError('Acquisition IDs required only for self-reference masking')
        k=F.normalize(self.fc_projection(self.reference_fc),dim=-1)
        current=F.normalize(self.fc_projection(fc[:,self.upper[0],self.upper[1]]),dim=-1)
        q=F.normalize(self.query_projection(h),dim=-1)
        pair=torch.cat(((q[:,None]-k[None]).abs(),q[:,None]*k[None]),-1)
        ra=self.reference_age
        meta={'age':torch.stack(((age[:,None]-ra[None]).abs(),age[:,None]*ra[None],(age[:,None]+ra[None])*.5),-1)}
        for name,x,embedding in [('sex',sex,self.sex_embedding),('site',site,self.site_embedding)]:
            ref=getattr(self,'reference_'+name);a=embedding(x)[:,None];b=embedding(ref)[None]
            same=(x[:,None]==ref[None])&(x[:,None]!=0)&(ref[None]!=0)
            meta[name]=torch.cat(((a-b).abs(),a*b,same.to(h.dtype)[...,None]),-1)
        permitted=torch.ones(len(h),len(k),device=h.device,dtype=torch.bool)
        for i,s in enumerate(sample_ids):
            j=self.reference_lookup.get(s)
            if j is not None:permitted[i,j]=False
        raw={};attention={};messages=[]
        for name in ['age','sex','site']:
            score=self.scorers[name](torch.cat((pair,meta[name]),-1)).squeeze(-1)
            a=score.masked_fill(~permitted,float('-inf')).softmax(-1)
            raw[name]=score.sigmoid()*permitted;attention[name]=a
            messages.append(a@self.values[name](k)-self.values[name](current))
        weights=self.gate(h).softmax(-1)
        message=(torch.stack(messages,1)*weights[...,None]).sum(1)
        return self.out(self.norm(message)),raw,weights,attention


class ResearchBrainPrompt(nn.Module):
    """Candidate-off delegates to the exact official adapter."""
    accepts_sample_ids=True
    def __init__(self,cfg,data,encoder,spec,train_indices):
        super().__init__();self.active=tuple(spec['modules']);self.num_classes=data.num_classes
        basecfg=copy.deepcopy(cfg)
        self.base=OriginalBrainPromptAdapter(basecfg,data.roi_prompt,data.disease_prompt)
        self.register_buffer('roi_prompt',data.roi_prompt.clone());self.register_buffer('disease_prompt',data.disease_prompt.clone())
        d=cfg['net_params']['hidden_dim'];n=data.node_num;p=data.roi_prompt.shape[-1]
        # Candidate initialization must not move the shared trunk/dropout RNG stream.
        with torch.random.fork_rng(devices=[]):
            if 'M1' in self.active:self.adaptive=SignedConditionalMessages(d,p,n,cfg['net_params']['L'])
            if 'M2' in self.active:self.disease_readout=DiseaseTokenReadout(d,p,self.num_classes)
            if 'M3' in self.active:
                rec=[data.records[i] for i in train_indices]
                self.population=ReferencePopulation(d,n,len(encoder.site_mapping)+1,data.fc[train_indices],encoder.transform(rec),[record_id(r) for r in rec])
            if 'M4' in self.active:self.slots=OrderedSlots(d,n)
            if 'M5' in self.active:self.covariance=CovarianceReadout(d)
    def forward(self,fc,subject_prompt,age,sex,site,sample_ids=None):
        if not self.active:return self.base(fc,subject_prompt,age,sex,site)
        import dgl
        ori=self.base.original;b,n,_=fc.shape;d=ori.embedding_h.out_features
        adj=legacy_brain_adjacency(fc,self.base.config['legacy_fc_threshold'],self.base.config['legacy_edge_ratio'])
        graphs=[]
        for x,a in zip(fc,adj):
            src,dst=a.nonzero(as_tuple=True);g=dgl.graph((src,dst),num_nodes=n,device=fc.device)
            g.ndata['feat']=x;g.edata['feat']=x[src,dst,None];graphs.append(g)
        g=dgl.batch(graphs);h=ori.embedding_h(g.ndata['feat']);e=g.edata['feat']
        h=ori.pos_emb(h.reshape(b,n,d)).reshape(-1,d)
        pos=neg=gates=None
        if 'M1' in self.active:pos,neg,gates=self.adaptive.graph(fc,self.roi_prompt)
        for i,conv in enumerate(ori.layers):
            before=h.reshape(b,n,d);h,e=conv(g,h,e)
            if pos is not None:h=h+self.adaptive.message(before,pos,neg,i).reshape(-1,d)
        g.ndata['h']=h;roi=h.reshape(b,n,d);subject=dgl.mean_nodes(g,'h')
        attention=tokens=None;parts={}
        if 'M2' in self.active:
            parts['M2'],attention,tokens=self.disease_readout(roi,self.disease_prompt)
        if 'M4' in self.active:parts['M4']=self.slots(roi)
        if 'M5' in self.active:parts['M5']=self.covariance(roi)
        for value in parts.values():subject=subject+value
        meta=ori.linear_layer1(subject_prompt.squeeze(1))
        legacy=ori.dropout(((ori.sim(subject)*ori.sim(meta))>.8).to(fc.dtype))
        fused=ori.global_gcs(subject,legacy)+subject+meta
        relations={};relation_weights=reference_attention=None
        if 'M3' in self.active:
            parts['M3'],relations,relation_weights,reference_attention=self.population(subject,fc,age,sex,site,sample_ids)
            fused=fused+parts['M3']
        disease_logits=F.normalize(fused,dim=-1)@F.normalize(ori.label_transform(self.disease_prompt),dim=-1).T
        return dict(logits=ori.MLP_layer(fused),disease_logits=disease_logits,A_brain=adj if pos is None else pos+neg,
                    roi_features=roi,roi_attention=attention,disease_tokens=tokens,subject_repr=subject,
                    population_adjacencies=relations,relation_weights=relation_weights,reference_attention=reference_attention,
                    legacy_population_adjacency=legacy,fused_repr=fused,module_residuals=parts,
                    signed_adjacencies=None if pos is None else {'positive':pos,'negative':neg},edge_gates=gates)
