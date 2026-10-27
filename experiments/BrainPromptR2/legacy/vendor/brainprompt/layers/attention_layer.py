import torch
from torch import nn


class LMPE(nn.Module):
    def __init__(self, emb_dim, d_model, node_num, dropout, init_emb):
        super(LMPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        if init_emb.shape != (node_num, emb_dim):
            raise ValueError('ROI prompt shape mismatch')
        self.register_buffer('init_emb', init_emb.detach().float().clone())

        # Compute the positional encodings once in log space.
        self.mlp = nn.Linear(emb_dim, d_model)

    def forward(self, x):
        bz, _, _, = x.shape
        device = x.device
        pos_emb = self.mlp(self.init_emb.to(device)).repeat(bz, 1, 1)
        emb = x + pos_emb
        return self.dropout(emb)


