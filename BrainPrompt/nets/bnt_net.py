"""BNT: two transformer blocks and orthonormal clustering readout.

Adapted from Wayfear/BrainNetworkTransformer, MIT; see third_party/licenses.
"""
import torch
from torch import nn
from torch.nn import functional as F
from layers.baseline_utils import dense_batch


class BNTNet(nn.Module):
    def __init__(self, p):
        super().__init__()
        n, d = p['node_num'], p['in_dim']
        heads = p.get('bnt_heads',4)
        clusters = p.get('bnt_clusters',min(100,n))
        if d % heads or not 1 <= clusters <= d:
            raise ValueError('BNT requires feature width divisible by heads and 1 <= clusters <= width')
        self.node_num = n
        self.transformers = nn.ModuleList([nn.TransformerEncoderLayer(d,heads,
            dim_feedforward=p.get('bnt_ffn_dim',1024),dropout=p.get('dropout',0.1),batch_first=True)
            for _ in range(2)])
        self.encoder = nn.Sequential(nn.Linear(n*d,32),nn.LeakyReLU(),nn.Linear(32,32),
                                     nn.LeakyReLU(),nn.Linear(32,n*d))
        centers = torch.empty(clusters,d)
        nn.init.orthogonal_(centers)
        self.cluster_centers = nn.Parameter(centers,requires_grad=not p.get('bnt_freeze_centers',True))
        self.reduction = nn.Sequential(nn.Linear(d,8),nn.LeakyReLU())
        self.classifier = nn.Sequential(nn.Linear(8*clusters,256),nn.LeakyReLU(),
            nn.Linear(256,32),nn.LeakyReLU(),nn.Linear(32,p['n_classes']))
        self.kl_weight = p.get('bnt_kl_weight',0.0)
        self.assignments = None

    def forward(self, g, h, e, batch_llms=None):
        x,_ = dense_batch(g,h)
        if x.shape[1] != self.node_num:
            raise ValueError('BNT graph does not match configured atlas')
        for block in self.transformers:
            x = block(x)
        encoded = self.encoder(x.flatten(1)).reshape_as(x)
        logits = (encoded @ self.cluster_centers.T).square()/self.cluster_centers.norm(dim=-1).clamp_min(1e-8)
        self.assignments = logits.softmax(-1)
        pooled = self.assignments.transpose(1,2) @ encoded
        return self.classifier(self.reduction(pooled).flatten(1))

    def loss(self, pred, label):
        loss = F.cross_entropy(pred,label)
        if self.kl_weight:
            q = self.assignments.flatten(0,1).clamp_min(1e-8)
            target = q.square()/q.sum(0).clamp_min(1e-8)
            target = (target/target.sum(-1,keepdim=True)).detach()
            loss = loss+self.kl_weight*F.kl_div(q.log(),target,reduction='batchmean')
        return loss
