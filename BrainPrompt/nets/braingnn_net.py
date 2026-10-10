"""BrainGNN dense PyTorch port; ROI-conditioned convolution + two R-pools.

Reference: xxlya/BrainGNN_Pytorch (see docs/BASELINES.md for port differences).
The original ROI identity is retained after pooling; labels enter loss only.
"""
import torch
from torch import nn
from torch.nn import functional as F
from layers.baseline_utils import dense_batch, select_nodes, pool_count, SafeBatchNorm1d


class ROIConv(nn.Module):
    def __init__(self, roi_num, in_dim, out_dim, communities):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.kernel = nn.Sequential(nn.Linear(roi_num, communities, bias=False),
                                    nn.ReLU(), nn.Linear(communities, in_dim*out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, a, identity):
        kernel = self.kernel(identity).reshape(*x.shape[:2], self.in_dim, self.out_dim)
        transformed = (x.unsqueeze(-2) @ kernel).squeeze(-2)
        eye = torch.eye(a.shape[-1], device=a.device, dtype=torch.bool)
        mask = (a != 0) | eye
        a = torch.where(eye & (a == 0), torch.ones_like(a), a)
        attn = a.masked_fill(~mask, -torch.inf).softmax(-1)
        # Upstream MyNNConv applies destination softmax then a mean aggregator.
        return (attn @ transformed) / mask.sum(-1, keepdim=True) + self.bias


class ROIPool(nn.Module):
    def __init__(self, dim, ratio):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim) / dim**0.5)
        self.ratio = ratio

    def forward(self, x, a, identity):
        score = (x @ F.normalize(self.weight, dim=0)).sigmoid()
        idx = score.topk(pool_count(x.shape[1], self.ratio), dim=1).indices
        x, a = select_nodes(x, a, idx)
        identity = identity.gather(1, idx.unsqueeze(-1).expand(-1, -1, identity.shape[-1]))
        return x*score.gather(1, idx).unsqueeze(-1), a, identity, score, idx


class BrainGNNNet(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.node_num = p['node_num']
        dim = p.get('braingnn_dim', 32)
        self.ratio = p.get('pool_ratio', 0.5)
        k = p.get('braingnn_communities', 8)
        self.conv1 = ROIConv(self.node_num, p['in_dim'], dim, k)
        self.conv2 = ROIConv(self.node_num, dim, dim, k)
        self.pool1, self.pool2 = ROIPool(dim, self.ratio), ROIPool(dim, self.ratio)
        self.classifier = nn.Sequential(nn.Linear(4*dim, dim), nn.ReLU(), SafeBatchNorm1d(dim),
            nn.Dropout(0.5), nn.Linear(dim, 512), nn.ReLU(), SafeBatchNorm1d(512),
            nn.Dropout(0.5), nn.Linear(512, p['n_classes']))
        self.unit_weight = p.get('braingnn_unit_weight', 0.0)
        self.topk_weight = p.get('braingnn_topk_weight', 0.1)
        self.consistency_weight = p.get('braingnn_consistency_weight', 0.1)
        self.last_scores = None

    def forward(self, g, h, e, batch_llms=None):
        x, a = dense_batch(g, h, e)
        if x.shape[1] != self.node_num:
            raise ValueError('BrainGNN graph does not match configured atlas')
        identity = torch.eye(self.node_num, device=h.device, dtype=h.dtype).expand(x.shape[0],-1,-1)
        x = self.conv1(x, a, identity)
        x, a, identity, s1, idx1 = self.pool1(x, a, identity)
        first = torch.cat((x.amax(1), x.mean(1)), -1)
        # Upstream squares adjacency after adding loops, then removes diagonal.
        eye = torch.eye(a.shape[-1], device=h.device, dtype=h.dtype)
        a = (a+eye) @ (a+eye)
        a = a*(1-eye)
        x = self.conv2(x, a, identity)
        x, a, identity, s2, idx2 = self.pool2(x, a, identity)
        second = torch.cat((x.amax(1), x.mean(1)), -1)
        self.last_scores = (s1, s2)
        self.last_roi_indices = (idx1.detach(), idx1.gather(1,idx2).detach())
        return self.classifier(torch.cat((first,second), -1))

    def loss(self, pred, label):
        loss = F.cross_entropy(pred, label)
        if self.unit_weight:
            loss = loss+self.unit_weight*sum((pool.weight.norm()-1).square() for pool in (self.pool1,self.pool2))
        if self.topk_weight:
            for scores in self.last_scores:
                scores = scores.sort(dim=1).values.clamp(1e-7,1-1e-7)
                count = max(1,int(scores.shape[1]*min(self.ratio,1-self.ratio)))
                loss = loss-self.topk_weight*(scores[:,-count:].log().mean()+(1-scores[:,:count]).log().mean())
        if self.consistency_weight:
            # First-layer scores are in original atlas order, so compare matching ROIs.
            for c in label.unique():
                scores = self.last_scores[0][label == c]
                loss = loss+self.consistency_weight*(scores-scores.mean(0)).square().sum()/scores.shape[0]
        return loss
