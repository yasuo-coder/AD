"""ContrastPool port: train-fold class summaries guide a shared soft assignment.

Based on AngusMonroe/ContrastPool (GPL-3.0); see third_party/licenses.
No evaluation labels or evaluation graphs enter the reference summaries.
"""
import torch
from torch import nn
from torch.nn import functional as F
from layers.baseline_utils import dense_batch, SafeBatchNorm1d, pool_count


class SummaryAttention(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.attention = nn.MultiheadAttention(n,1,dropout=0,batch_first=True)
        self.norm = nn.LayerNorm(n)

    def forward(self, x):
        return self.norm(x+self.attention(x,x,x,need_weights=False)[0])


class MeanSAGE(nn.Module):
    def __init__(self, dim, dropout, activation=True, batch_norm=True):
        super().__init__()
        self.linear = nn.Linear(2*dim,dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.bn = SafeBatchNorm1d(dim) if batch_norm else nn.Identity()

    def forward(self, x, a):
        x = self.dropout(x)
        neighbors = (a @ x)/a.sum(-1,keepdim=True).clamp_min(1)
        y = F.normalize(self.linear(torch.cat((x,neighbors),-1)),dim=-1)
        if self.activation:
            y = F.relu(y)
        return self.bn(y.flatten(0,1)).reshape_as(y)


class PooledSAGE(nn.Module):
    def __init__(self, dim, clusters, residual):
        super().__init__()
        self.linear = nn.Linear(dim,dim)
        self.bn = SafeBatchNorm1d(clusters)
        self.residual = residual

    def forward(self,x,a):
        y = F.relu(F.normalize(self.linear(a @ x),dim=-1))
        return self.bn(y+x if self.residual else y)


class ContrastPoolNet(nn.Module):
    def __init__(self,p,trainset):
        super().__init__()
        n,dim = p['node_num'],p['hidden_dim']
        if p['in_dim'] != n:
            raise ValueError('ContrastPool requires --node_feat_transform pearson')
        if trainset is None:
            raise ValueError('ContrastPool requires the current training fold to build class references')
        self.node_num = n
        self.classes = p['n_classes']
        max_refs = p.get('contrast_max_references',0)
        for c in range(self.classes):
            refs = [sample[0].ndata['feat'].detach().cpu().float().clone()
                    for sample in trainset if int(sample[1]) == c]
            if not refs:
                raise ValueError(f'ContrastPool training fold has no samples for class {c}')
            if max_refs:
                refs = refs[:max_refs]
            self.register_buffer(f'reference_{c}',torch.stack(refs))
        self.adj_roi, self.adj_subject = SummaryAttention(n),SummaryAttention(n)
        self.feat_roi, self.feat_subject = SummaryAttention(n),SummaryAttention(n)
        self.embedding = nn.Linear(n,dim)
        layers = p.get('L',3)
        self.before = nn.ModuleList([MeanSAGE(dim,p.get('dropout',0),activation=i<layers-1,
                                            batch_norm=p.get('batch_norm',True)) for i in range(layers)])
        self.feat_pool = MeanSAGE(dim,p.get('dropout',0))
        clusters = pool_count(n,p.get('pool_ratio',0.5))
        self.assignment_weight = nn.Parameter(torch.randn(n,clusters)/n**0.5)
        self.assignment_bias = nn.Parameter(torch.zeros(clusters))
        self.after = nn.ModuleList([PooledSAGE(dim,clusters,p.get('residual',True)) for _ in range(layers)])
        self.classifier = nn.Linear(dim,self.classes)
        self.entropy_weight = p.get('contrast_entropy_weight',0.001)
        self.last_contrast = self.last_assignment = None
        self.attention_entropy = None

    def forward(self,g,h,e,batch_llms=None):
        x,a = dense_batch(g,h)
        if x.shape[1] != self.node_num:
            raise ValueError('ContrastPool atlas mismatch')
        adj_summaries,feat_summaries = [],[]
        for c in range(self.classes):
            ref = getattr(self,f'reference_{c}')
            adj_summaries.append(self.adj_subject(self.adj_roi(ref).transpose(0,1)).mean(1))
            feat_summaries.append(self.feat_subject(self.feat_roi(ref).transpose(0,1)).mean(1))
        contrast = torch.stack(adj_summaries).std(0,unbiased=True)
        difference = torch.stack(feat_summaries).std(0,unbiased=True)
        logits = contrast @ (difference @ self.assignment_weight)+self.assignment_bias
        assignment = logits.softmax(-1)
        prob = contrast.softmax(-1)
        self.attention_entropy = -(prob*prob.clamp_min(1e-8).log()).sum(-1).mean()
        self.last_contrast,self.last_assignment = contrast.detach(),assignment.detach()
        x = self.embedding(x)
        for layer in self.before:
            x = layer(x,a)
        x = self.feat_pool(x,a)
        pooled = (assignment.T @ x)/(assignment.shape[1]**0.5)
        pooled_a = assignment.T @ a @ assignment
        for layer in self.after:
            pooled = layer(pooled,pooled_a)
        return self.classifier(pooled.sum(1))

    def loss(self,pred,label):
        return F.cross_entropy(pred,label)+self.entropy_weight*self.attention_entropy
