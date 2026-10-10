"""Dense weighted message passing; adjacency is [B,N,N] or [N,N]."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def symmetric_normalize(adjacency, eps=1e-8):
    degree = adjacency.sum(-1).clamp_min(eps)
    inv = degree.rsqrt()
    return inv.unsqueeze(-1) * adjacency * inv.unsqueeze(-2)


def unit_diagonal(adjacency):
    eye = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype)
    return adjacency * (1 - eye) + eye


def off_diagonal(adjacency):
    n = adjacency.shape[-1]
    mask = ~torch.eye(n, device=adjacency.device, dtype=torch.bool)
    return adjacency[..., mask]


class WeightedGCNLayer(nn.Module):
    """D^-1/2 A D^-1/2 H W + b, activation, optional residual/dropout.

    The B axis is never folded into N; there is no cross-subject ROI message.
    BatchNorm (when enabled for baseline parity) shares training statistics,
    but does not introduce cross-subject edges.
    """
    def __init__(self, in_dim, out_dim, dropout=0., batch_norm=True, residual=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)
        self.norm = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
        self.residual = residual and in_dim == out_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adjacency):
        # h: [B,N,D]; adjacency: [B,N,N], differentiable edge weights.
        out = symmetric_normalize(adjacency) @ (h @ self.weight) + self.bias
        out = self.norm(out.reshape(-1, out.shape[-1])).reshape_as(out)
        out = F.relu(out)
        if self.residual:
            out = out + h
        return self.dropout(out)


class LegacyPopulationGCN(nn.Module):
    """Exact unnormalized simpleGCNLayer equations from official BrainPrompt."""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.empty(1, dim))
        bound = 1 / math.sqrt(dim)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, h, adjacency):
        return h + F.relu(adjacency @ (h @ self.weight) + self.bias)


def legacy_brain_adjacency(fc, threshold=.3, edge_ratio=.2):
    """Preserve repository's signed quantile followed by absolute comparison."""
    if not 0 <= edge_ratio <= 1:
        raise ValueError('edge_ratio must be in [0,1]')
    if edge_ratio:
        flat = fc.flatten(-2)
        index = min(int(flat.shape[-1] * (1 - edge_ratio)), flat.shape[-1] - 1)
        threshold = flat.sort(-1).values[..., index, None, None]
    return (fc.abs() >= threshold).to(fc.dtype)


def legacy_population_adjacency(h, meta, threshold=.8):
    h = F.normalize(h, dim=-1, eps=1e-8)
    meta = F.normalize(meta, dim=-1, eps=1e-8)
    return ((h @ h.T).sigmoid() * (meta @ meta.T).sigmoid() > threshold).to(h.dtype)
