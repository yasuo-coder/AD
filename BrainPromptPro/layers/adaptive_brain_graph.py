"""FC and semantic guided, subject-specific continuous brain graphs."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .gcn_layer import unit_diagonal


class SemanticGuidedAdaptiveBrainGraph(nn.Module):
    def __init__(self, feature_dim, prompt_dim, hidden_dim, edge_hidden_dim=64,
                 edge_dropout=0., pair_chunk_size=16):
        super().__init__()
        if pair_chunk_size < 1:
            raise ValueError('pair_chunk_size must be positive')
        self.pair_chunk_size = pair_chunk_size
        self.feature_projection = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU())
        self.roi_projection = nn.Sequential(nn.Linear(prompt_dim, hidden_dim), nn.GELU())
        self.node_mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU())
        self.edge_mlp = nn.Sequential(nn.Linear(2 + 2 * hidden_dim, edge_hidden_dim),
                                      nn.GELU(), nn.Dropout(edge_dropout), nn.Linear(edge_hidden_dim, 1))

    def _score_chunk(self, left, right, fc):
        pair = torch.cat((fc.unsqueeze(-1), fc.abs().unsqueeze(-1),
                          (left - right).abs(), left * right), dim=-1)
        return self.edge_mlp(pair).squeeze(-1).sigmoid()

    def forward(self, fc, roi_prompt, features=None):
        # fc: [B,N,N]; roi_prompt: [N,P]; features: [B,N,F].
        features = fc if features is None else features
        x = self.feature_projection(features)
        r = self.roi_projection(roi_prompt).unsqueeze(0).expand(fc.shape[0], -1, -1)
        h = self.node_mlp(torch.cat((x, r), dim=-1))
        chunks = []
        for start in range(0, fc.shape[-1], self.pair_chunk_size):
            left = h[:, start:start + self.pair_chunk_size, None, :]
            right = h[:, None, :, :]
            c = fc[:, start:start + self.pair_chunk_size, :]
            # Recompute pair features in backward instead of retaining B*N*N*D.
            if self.training and torch.is_grad_enabled():
                score = checkpoint(self._score_chunk, left, right, c, use_reentrant=False)
            else:
                score = self._score_chunk(left, right, c)
            chunks.append(score)
        adjacency = torch.cat(chunks, dim=1)
        # Also enforces symmetry when independent dropout masks are active.
        adjacency = unit_diagonal((adjacency + adjacency.transpose(-1, -2)) * .5)
        return adjacency, h
