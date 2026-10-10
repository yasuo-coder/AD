"""Local-to-global GNN with SABP and phenotype-conditioned population edges.

Adapted from cnuzh/LG-GNN (MIT). Uses one split at a time, never train+test
in a single population; see docs/BASELINES.md for the protocol change.
"""
import torch
from torch import nn
from torch.nn import functional as F
from layers.baseline_utils import (dense_batch,DenseGCN,normalized_adjacency,
                                   select_nodes,pool_count,SafeBatchNorm1d)


class Chebyshev(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.filters = nn.ModuleList([nn.Linear(in_dim,out_dim,bias=False) for _ in range(3)])
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self,x,a):
        # Symmetric normalized Laplacian with lambda_max=2: L_scaled=-D^-1/2 A D^-1/2.
        lap = -normalized_adjacency(a,self_loops=False)
        t1 = lap @ x
        t2 = 2*(lap @ t1)-x
        return self.filters[0](x)+self.filters[1](t1)+self.filters[2](t2)+self.bias


class LGGNNNet(nn.Module):
    def __init__(self,p,trainset):
        super().__init__()
        n = p['node_num']
        self.node_num = n
        self.local1 = DenseGCN(p['in_dim'],64)
        self.local2 = DenseGCN(64,20)
        self.score = DenseGCN(20,1)
        self.mi_embed = DenseGCN(20,20)
        self.mi_score = nn.Linear(40,1)
        self.local3 = DenseGCN(20,20)
        self.ratio = p.get('lg_pool_ratio',0.9)
        self.phenotype_encoder = nn.Sequential(nn.Linear(2,128),nn.ReLU(),SafeBatchNorm1d(128),
            nn.Dropout(0.2),nn.Linear(128,128))
        width = 20*pool_count(n,self.ratio)
        self.global_layers = nn.ModuleList([Chebyshev(width if i==0 else 20,20) for i in range(4)])
        self.norms = nn.ModuleList([SafeBatchNorm1d(20) for _ in range(4)])
        self.layer_weights = nn.Parameter(torch.randn(4))
        self.classifier = nn.Linear(20,p['n_classes'])
        self.mi_weight = p.get('lg_mi_weight',0.1)
        self.affinity_threshold = p.get('lg_affinity_threshold',1.1)
        self.age_threshold = p.get('lg_age_threshold',2.0)
        if trainset is None:
            raise ValueError('LG-GNN needs training-fold demographics to fit its normalization')
        try:
            demographics = torch.stack([s[0].ndata['demographics'][0].detach().cpu() for s in trainset])
        except KeyError as exc:
            raise ValueError('LG-GNN requires --baseline_metadata_csv with graph_index,age,sex,site') from exc
        self.register_buffer('meta_mean',demographics.mean(0))
        self.register_buffer('meta_std',demographics.std(0,unbiased=False).clamp_min(1e-6))
        self.mi = None
        self.last_population_weights = None

    def forward(self,g,h,e,batch_llms=None):
        x,a = dense_batch(g,h,e)
        if x.shape[1] != self.node_num:
            raise ValueError('LG-GNN atlas mismatch')
        if 'demographics' not in g.ndata:
            raise ValueError('LG-GNN needs numeric demographics, not LLM subject embeddings')
        metadata = torch.stack(g.ndata['demographics'].split(g.batch_num_nodes().tolist()))[:,0].to(h)
        a = a.abs()
        x = F.relu(self.local1(x,a))
        x = F.relu(self.local2(x,a))
        embed = self.mi_embed(x,a)
        neg = x[:,torch.randperm(x.shape[1],device=x.device)] if self.training else x.roll(1,1)
        joint = self.mi_score(torch.cat((embed,x),-1)).flatten()
        marginal = self.mi_score(torch.cat((embed,neg),-1)).flatten()
        self.mi = joint.mean()-torch.logsumexp(marginal,0)+marginal.new_tensor(marginal.numel()).log()
        scores = self.score(x,a).squeeze(-1)
        idx = scores.topk(pool_count(x.shape[1],self.ratio),dim=1).indices
        pooled,pooled_a = select_nodes(x,a,idx)
        pooled = pooled*scores.gather(1,idx).tanh().unsqueeze(-1)
        embedding = (pooled+F.relu(self.local3(pooled,pooled_a))).flatten(1)
        # Static candidates: Pearson-distance similarity times age/sex/site affinity.
        centered = F.normalize(embedding.detach()-embedding.detach().mean(-1,keepdim=True),dim=-1)
        dist = (1-centered @ centered.T).clamp_min(0)
        dist.fill_diagonal_(0)
        sigma = dist.mean().clamp_min(1e-6)
        phenotype = ((metadata[:,None,0]-metadata[None,:,0]).abs()<self.age_threshold).to(h)
        phenotype = phenotype+(metadata[:,None,1]==metadata[None,:,1]).to(h)
        phenotype = phenotype+(metadata[:,None,2]==metadata[None,:,2]).to(h)
        candidates = (phenotype*torch.exp(-dist.square()/(2*sigma.square())))>self.affinity_threshold
        candidates.fill_diagonal_(False)
        # Site is categorical and only used for equality above; its arbitrary
        # integer encoding must not enter a numerical distance or learned MLP.
        normalized_meta = (metadata-self.meta_mean)/self.meta_std
        encoded = F.normalize(self.phenotype_encoder(normalized_meta[:,:2]),dim=-1)
        population = ((encoded @ encoded.T+1)*0.5)*candidates
        self.last_population_weights = population.detach()
        self.last_roi_indices = idx.detach()
        layer_out = []
        z = embedding
        for i,(conv,norm) in enumerate(zip(self.global_layers,self.norms)):
            z = F.relu(norm(conv(z,population)))
            if i:
                z = z+0.7*layer_out[-1]
            layer_out.append(z)
            if i<3:
                z = F.dropout(z,0.3,training=self.training)
        out = (torch.stack(layer_out)*self.layer_weights.softmax(0)[:,None,None]).sum(0)
        return self.classifier(out)

    def loss(self,pred,label):
        return F.cross_entropy(pred,label)-self.mi_weight*self.mi
