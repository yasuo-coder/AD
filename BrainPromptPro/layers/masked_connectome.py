"""P2: numerical symmetric masking with explicit visibility and isolated targets."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def symmetric_fc_mask(fc, ratio, generator=None, valid=None):
    if fc.ndim != 3 or fc.shape[-1] != fc.shape[-2]:
        raise ValueError('Expected [B,N,N] FC')
    if not 0 <= ratio < 1 or not torch.isfinite(fc).all():
        raise ValueError('Invalid mask ratio or nonfinite FC')
    if not torch.allclose(fc, fc.transpose(-1,-2), atol=1e-6):
        raise ValueError('FC must be symmetric')
    valid = torch.ones_like(fc,dtype=torch.bool) if valid is None else valid
    if valid.shape != fc.shape or valid.dtype != torch.bool or not torch.equal(valid,valid.transpose(-1,-2)):
        raise ValueError('Validity must be a symmetric Boolean matrix')
    upper=torch.triu_indices(fc.shape[-1],fc.shape[-1],1,device=fc.device)
    draw=torch.rand((len(fc),upper.shape[1]),device=fc.device,generator=generator)<ratio
    hidden=torch.zeros_like(valid)
    hidden[:,upper[0],upper[1]]=draw & valid[:,upper[0],upper[1]]
    hidden=hidden | hidden.transpose(-1,-2)
    visible=valid & ~hidden
    masked=torch.where(visible,fc,torch.zeros_like(fc))
    return masked,visible


def masked_topology(masked_fc, visible, edge_ratio=.2):
    """Choose edges from visible values only; hidden original values never enter."""
    if not 0 < edge_ratio <= 1:raise ValueError('edge_ratio must be in (0,1]')
    if masked_fc.shape != visible.shape:raise ValueError('Shape mismatch')
    n=masked_fc.shape[-1];eye=torch.eye(n,device=visible.device,dtype=torch.bool)
    candidates=visible & ~eye
    # Match the baseline signed-quantile/absolute comparison convention on the
    # visible subset. Masked positions cannot influence the quantile or edges.
    adj=[]
    for x,v in zip(masked_fc,candidates):
        values=x[v]
        if not len(values):a=eye.clone()
        else:
            index=min(int(len(values)*(1-edge_ratio)),len(values)-1)
            threshold=values.sort().values[index]
            a=(x.abs()>=threshold)&v
            a=a|eye
        adj.append(a)
    return torch.stack(adj).to(masked_fc.dtype)


class SymmetricConnectionDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder='dot', rank=32):
        super().__init__();self.decoder=decoder
        if decoder not in ('dot','pair'):raise ValueError(decoder)
        self.project=nn.Linear(hidden_dim,rank,bias=False)
        if decoder=='pair':self.pair=nn.Sequential(nn.Linear(rank*3,rank),nn.GELU(),nn.Linear(rank,1))
        else:self.bias=nn.Parameter(torch.zeros(()))
    def forward(self,h):
        x=self.project(h)
        if self.decoder=='dot':score=x@x.transpose(-1,-2)/math.sqrt(x.shape[-1])+self.bias
        else:
            a=x[:,:,None];b=x[:,None,:]
            score=self.pair(torch.cat((a+b,(a-b).abs(),a*b),-1)).squeeze(-1)
        return score.tanh()


def reconstruction_loss(prediction,target,visible,valid=None):
    valid=torch.ones_like(visible) if valid is None else valid
    selected=(~visible)&valid&torch.ones_like(visible).triu(1)
    if not selected.any():return prediction.sum()*0
    return (prediction[selected]-target.detach()[selected]).square().mean()


def reconstruction_metrics(prediction,target,visible,valid=None):
    valid=torch.ones_like(visible) if valid is None else valid
    take=(~visible)&valid&torch.ones_like(visible).triu(1)
    x=prediction.detach()[take];y=target.detach()[take]
    def mse(m):return float((x[m]-y[m]).square().mean()) if m.any() else None
    correlation=None
    if len(x)>1 and x.std()>0 and y.std()>0:
        correlation=float(torch.corrcoef(torch.stack((x,y)))[0,1])
    return dict(masked_edges=len(x),masked_mse=mse(torch.ones_like(x,dtype=torch.bool)),positive_mse=mse(y>0),negative_mse=mse(y<0),correlation=correlation)


def symmetric_kl(logits1,logits2):
    a=F.log_softmax(logits1.float(),-1);b=F.log_softmax(logits2.float(),-1)
    return .5*((a.exp()*(a-b)).sum(-1)+(b.exp()*(b-a)).sum(-1)).mean()
