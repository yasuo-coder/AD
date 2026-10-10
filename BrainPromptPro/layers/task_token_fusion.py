"""P3 reuses historical M2 V2 attention; adds explicit class-token logits."""
import torch
from torch import nn
from ..nets.research_candidates_v1 import DiseaseTokenReadout
from ..data.labels import CLASS_NAMES


class TaskSpecificROIFusion(nn.Module):
    def __init__(self, hidden_dim, prompt_dim, class_names=CLASS_NAMES):
        super().__init__()
        if tuple(class_names)!=tuple(CLASS_NAMES):raise ValueError('Disease token order mismatch')
        self.class_names=tuple(class_names)
        self.readout=DiseaseTokenReadout(hidden_dim,prompt_dim,len(class_names))
        self.class_weight=nn.Parameter(torch.empty(len(class_names),hidden_dim))
        self.class_bias=nn.Parameter(torch.zeros(len(class_names)))
        nn.init.xavier_uniform_(self.class_weight)
    def class_logits(self,tokens):
        if tokens.shape[1]!=len(self.class_names):raise ValueError('Expected all five class tokens')
        return (tokens*self.class_weight[None]).sum(-1)+self.class_bias
    def forward(self,roi,disease_prompt):
        if disease_prompt.shape[0]!=len(self.class_names):raise ValueError('Expected all disease prompts')
        residual,attention,tokens=self.readout(roi,disease_prompt)
        return dict(residual=residual,logits=self.class_logits(tokens),tokens=tokens,attention=attention)
