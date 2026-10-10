"""All disease prompts query every subject; labels are not model inputs."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class DiseaseAwareSemanticReadout(nn.Module):
    def __init__(self, hidden_dim, prompt_dim, attention_dim=None):
        super().__init__()
        attention_dim = attention_dim or hidden_dim
        self.scale = math.sqrt(attention_dim)
        self.q_projection = nn.Linear(prompt_dim, attention_dim)
        self.k_projection = nn.Linear(hidden_dim, attention_dim)
        self.v_projection = nn.Linear(hidden_dim, hidden_dim)
        self.representation_projection = nn.Linear(hidden_dim, attention_dim)
        self.disease_projection = nn.Linear(attention_dim, attention_dim)

    def forward(self, h, disease_prompt):
        # h: [B,N,D]; disease_prompt: [K,P]; roi_attention: [B,K,N].
        q = self.q_projection(disease_prompt)
        k = self.k_projection(h)
        v = self.v_projection(h)
        attention = torch.softmax(torch.einsum('kd,bnd->bkn', q, k) / self.scale, dim=-1)
        representations = attention @ v  # [B,K,D]
        r = F.normalize(self.representation_projection(representations), dim=-1, eps=1e-8)
        d = F.normalize(self.disease_projection(q), dim=-1, eps=1e-8)
        logits = (r * d.unsqueeze(0)).sum(-1)  # [B,K]
        probability = torch.softmax(logits, dim=-1)
        subject = (probability.unsqueeze(-1) * representations).sum(1)
        return subject, logits, attention
