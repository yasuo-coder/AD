"""P3 reuses historical M2 V2 attention; adds explicit class-token logits."""
import torch
from torch import nn
from torch.nn import functional as F
from ..nets.research_candidates_v1 import DiseaseTokenReadout
from ..data.labels import CLASS_NAMES


class TaskSpecificROIFusion(nn.Module):
    def __init__(self, hidden_dim, prompt_dim, class_names=CLASS_NAMES, mode="semantic"):
        super().__init__()
        if tuple(class_names)!=tuple(CLASS_NAMES):raise ValueError('Disease token order mismatch')
        if mode not in ('semantic','random','learned_query'):raise ValueError(mode)
        self.mode=mode
        self.class_names=tuple(class_names)
        self.readout=DiseaseTokenReadout(hidden_dim,prompt_dim,len(class_names))
        self.class_weight=nn.Parameter(torch.empty(len(class_names),hidden_dim))
        self.class_bias=nn.Parameter(torch.zeros(len(class_names)))
        nn.init.xavier_uniform_(self.class_weight)
        if mode=='random':
            g=torch.Generator().manual_seed(314159)
            self.register_buffer('random_prompt',torch.randn(len(class_names),prompt_dim,generator=g))
        elif mode=='learned_query':
            self.readout.q=nn.Identity()
            self.queries=nn.Parameter(torch.randn(len(class_names),32)/32**.5)
    def class_logits(self,tokens):
        if tokens.shape[1]!=len(self.class_names):raise ValueError('Expected all five class tokens')
        return (tokens*self.class_weight[None]).sum(-1)+self.class_bias
    def forward(self,roi,disease_prompt):
        if disease_prompt.shape[0]!=len(self.class_names):raise ValueError('Expected all disease prompts')
        prompt=self.random_prompt if self.mode=='random' else self.queries if self.mode=='learned_query' else disease_prompt
        residual,attention,tokens=self.readout(roi,prompt)
        return dict(residual=residual,logits=self.class_logits(tokens),tokens=tokens,attention=attention)


class ParameterMatchedROIFusion(nn.Module):
    """Mean-pool MLP control; all parameters are used, no disease queries."""
    def __init__(self, hidden_dim, prompt_dim, classes=5):
        super().__init__()
        d=hidden_dim;k=classes
        target=prompt_dim*32+d*32+d*d+k*d*d+2*d+k*d+k
        width=round((target-(d+k))/(2*d+k+1))
        self.hidden=nn.Linear(d,width)
        self.output=nn.Linear(width,d+k)
        self.hidden_dim=d
        # Match the semantic module's initially zero residual path.
        with torch.no_grad():
            self.output.weight[:d].zero_();self.output.bias[:d].zero_()
        actual=sum(p.numel() for p in self.parameters())
        if abs(actual-target)/target>.01:raise ValueError('Control parameter mismatch exceeds 1%')
        self.parameter_counts={'semantic_target':target,'actual':actual}
    def forward(self,roi,disease_prompt):
        value=self.output(F.gelu(self.hidden(roi.mean(1))))
        return dict(residual=value[:,:self.hidden_dim],logits=value[:,self.hidden_dim:],tokens=None,attention=None)
