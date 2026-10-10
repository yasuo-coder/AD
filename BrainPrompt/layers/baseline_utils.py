"""Fixed-atlas tensor adapters used by the Table 2 baseline ports."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def dense_batch(g, h, e=None):
    sizes = g.batch_num_nodes().tolist()
    if not sizes or len(set(sizes)) != 1:
        raise ValueError('These baselines require equal, consistently ordered atlas ROIs per graph')
    n = sizes[0]
    if h.shape[0] != sum(sizes):
        raise ValueError('Node feature count does not match graph batch')
    x = torch.stack(h.split(sizes))
    src, dst = g.edges(order='eid')
    offsets = torch.arange(len(sizes), device=h.device).repeat_interleave(n) * n
    if not torch.equal(offsets[src], offsets[dst]):
        raise ValueError('ROI graph edges cannot cross subject boundaries')
    a = h.new_zeros(len(sizes), n, n)
    values = h.new_ones(src.numel()) if e is None else e.reshape(-1).to(h)
    if values.numel() != src.numel():
        raise ValueError('Expected one scalar edge weight per edge')
    # Matrix multiplication uses receiver rows and sender columns.
    a.index_put_((offsets[src] // n, dst-offsets[dst], src-offsets[src]), values, accumulate=True)
    return x, a


def select_nodes(x, a, indices):
    x = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    a = a.gather(1, indices.unsqueeze(-1).expand(-1, -1, a.shape[-1]))
    a = a.gather(2, indices.unsqueeze(1).expand(-1, a.shape[1], -1))
    return x, a


def normalized_adjacency(a, self_loops=True):
    if self_loops:
        eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
        a = a * (1-eye) + eye
    inv = a.sum(-1).clamp_min(1e-8).rsqrt()
    return inv.unsqueeze(-1)*a*inv.unsqueeze(-2)


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Use running statistics for a singleton final batch, without dropping subjects."""
    def forward(self, x):
        if self.training and x.numel() // x.shape[1] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight,
                                self.bias, False, self.momentum, self.eps)
        return super().forward(x)


class DenseGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, a):
        return normalized_adjacency(a) @ self.linear(x) + self.bias


def pool_count(n, ratio):
    if not 0 < ratio <= 1:
        raise ValueError('pool ratio must be in (0,1]')
    return max(1, math.ceil(n*ratio))
