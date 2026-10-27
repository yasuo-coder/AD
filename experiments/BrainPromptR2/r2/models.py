import math,sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import OLD,ROOT
sys.path.insert(0,str(ROOT/'legacy'))
from bpstudy.models import BrainPromptG,BrainPromptT,FCBiasedLayer,fisher_transform

class StagedT(nn.Module):
 """FIT capacity diagnostic. No ROI text in the pure numeric ROI-token stages."""
 def __init__(self,data,stage,hidden=64):
  super().__init__();self.stage=stage;self.fc_projection=nn.Linear(116,hidden)
  self.identity=nn.Parameter(torch.randn(116,hidden)/math.sqrt(hidden))
  self.subject_projection=nn.Linear(2048,hidden)
  self.disease_projection=nn.Linear(2048,hidden)
  self.register_buffer('disease_prompt',data['disease'].clone())
  self.encoder=nn.ModuleList([FCBiasedLayer(hidden,4,128,0.,.5,stage>=1) for _ in range(2)])
  self.final_norm=nn.LayerNorm(hidden)
  self.cross_attention=nn.MultiheadAttention(hidden,4,dropout=0.,batch_first=True)
  self.class_score=nn.Linear(hidden,1);self.head=nn.Linear(hidden,5)
 def forward_batch(self,data,ids,device,**kwargs):
  fc=data['fc'][ids].to(device);tokens=self.fc_projection(fc)+self.identity[None]
  bias=fisher_transform(fc)
  if self.stage>=2:
   prompt=data['subject_prompt'][ids].to(device)
   if kwargs.get('mask_subject'):prompt=torch.zeros_like(prompt)
   tokens=torch.cat([tokens,self.subject_projection(prompt)[:,None]],1);bias=F.pad(bias,(0,1,0,1))
  for layer in self.encoder:tokens=layer(tokens,bias)
  tokens=self.final_norm(tokens)
  if self.stage>=3:
   queries=self.disease_projection(self.disease_prompt).expand(len(ids),-1,-1)
   features,_=self.cross_attention(queries,tokens[:,:116],tokens[:,:116],need_weights=False)
   logits=self.class_score(features+queries).squeeze(-1)
  else:logits=self.head(tokens[:,:116].mean(1))
  return dict(logits=logits,features=tokens[:,:116].mean(1))

class NumericModel(nn.Module):
 def __init__(self,dim,kind):
  super().__init__();self.kind=kind
  self.net=nn.Linear(dim,5) if kind=='logistic' else nn.Sequential(nn.Linear(dim,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,5))
 def forward_batch(self,data,ids,device,**kwargs):return {'logits':self.net(data['x'][ids].to(device))}
