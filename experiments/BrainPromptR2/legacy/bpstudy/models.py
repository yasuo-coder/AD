"""Auditable BrainPrompt variants; canonical classes are CN, AD, SMC, EMCI, LMCI.

Only the ordinal head uses the natural CN/SMC/EMCI/LMCI/AD ordering. No label
enters a sample's forward pass. Soft population inference references prototypes
rebuilt exclusively from the current fold's fitting subjects.
"""
from collections import Counter
import math

import dgl
import torch
from torch import nn
from torch.nn import functional as F

from vendor.brainprompt.nets.brainpromptg_net import BrainPromptGNet
from vendor.brainprompt.nets.brainpromptc_net import BrainPromptCNet
from vendor.brainprompt.layers.gcn_layer import simpleGCNLayer


DEFAULTS = dict(
    model='BrainPromptG', pop_fusion='original', fisher_z=False,
    multiscale=False, signed=False, ordinal=False,
    hidden_dim=116, layers=4, dropout=0.5, batch_norm=True, residual=True,
    readout='mean', edge_ratio=0.2, edge_ratios=[0.1, 0.2, 0.3],
    fc_threshold=0.3, population_threshold=0.8, population_k=5,
    fisher_clip=0.999, normalization_eps=1e-8,
    transformer_hidden=64, transformer_layers=2, transformer_heads=4,
    transformer_feedforward=128, transformer_dropout=0.2,
    prompt_dropout=[0.2, 0.5], fc_bias=True, fc_bias_initial=0.5,
    ordinal_threshold_start=-1.5, ordinal_threshold_gap=1.0,
)


def resolved_config(config):
    out = dict(DEFAULTS)
    out.update(config)
    if out['pop_fusion'] not in {'original', 'soft_topk', 'off'}:
        raise ValueError('Unsupported population fusion')
    if not 0 < out['fisher_clip'] < 1:
        raise ValueError('Fisher clipping must lie in (0,1)')
    if len(out['prompt_dropout']) != 2 or any(not 0 <= p <= 1 for p in out['prompt_dropout']):
        raise ValueError('Expected ROI and subject dropout probabilities')
    return out


def fisher_transform(fc, clip=0.999):
    z = fc.clamp(-clip, clip).atanh()
    eye = torch.eye(fc.shape[-1], dtype=torch.bool, device=fc.device)
    return z.masked_fill(eye, 0.)


def symmetric_edge_keep(fc, drop):
    """Symmetric undirected perturbation; diagonal always retained."""
    if not 0 <= drop < 1:
        raise ValueError('Edge dropout must lie in [0,1)')
    if drop == 0:
        return torch.ones_like(fc, dtype=torch.bool)
    keep = (torch.rand_like(fc) >= drop).triu(1)
    keep = keep | keep.transpose(-1, -2)
    return keep | torch.eye(fc.shape[-1], dtype=torch.bool, device=fc.device)


def legacy_adjacency(fc, ratio=.2, threshold=.3):
    """Exact original signed quantile then absolute comparison (not abs top-k)."""
    if not 0 <= ratio <= 1:
        raise ValueError('Invalid edge ratio')
    if ratio:
        flat = fc.flatten(-2)
        index = min(int(flat.shape[-1] * (1 - ratio)), flat.shape[-1] - 1)
        threshold = flat.sort(-1).values[..., index, None, None]
    return fc.abs() >= threshold


def density_adjacency(fc, ratio):
    """Keep an exact fraction of undirected off-diagonal |FC| edges.

    The original signed-quantile rule remains confined to the single-view
    baseline. Multi-view densities refer to actual undirected edge counts.
    """
    if not 0 < ratio <= 1:
        raise ValueError('Density must lie in (0,1]')
    n = fc.shape[-1]
    row, col = torch.triu_indices(n, n, 1, device=fc.device)
    count = max(1, int(len(row) * ratio))
    selected = fc[:, row, col].abs().argsort(dim=-1, descending=True, stable=True)[:, :count]
    adjacency = torch.eye(n, device=fc.device, dtype=torch.bool).expand(len(fc), -1, -1).clone()
    batch = torch.arange(len(fc), device=fc.device)[:, None]
    adjacency[batch, row[selected], col[selected]] = True
    adjacency[batch, col[selected], row[selected]] = True
    return adjacency


def graph_batch(features, adjacency):
    graphs = []
    for x, a in zip(features, adjacency):
        src, dst = a.nonzero(as_tuple=True)
        g = dgl.graph((src, dst), num_nodes=len(x), device=x.device)
        g.ndata['feat'] = x
        g.edata['feat'] = x[src, dst, None]
        graphs.append(g)
    return dgl.batch(graphs)


class OrderedCumulativeHead(nn.Module):
    """Shared coefficient with ordered cut-points: logits decrease by boundary."""
    def __init__(self, hidden, start=-1.5, gap=1.):
        super().__init__()
        if gap <= 0:
            raise ValueError('Ordinal threshold gaps must be positive')
        self.weight = nn.Linear(hidden, 1, bias=False)
        self.first = nn.Parameter(torch.tensor(float(start)))
        self.raw_gaps = nn.Parameter(torch.full((3,), math.log(math.expm1(gap))))

    def forward(self, h):
        thresholds = torch.cat([self.first[None], self.first + F.softplus(self.raw_gaps).cumsum(0)])
        return self.weight(h) - thresholds[None]


class PopulationReferenceMixin:
    def _initialize_reference(self, hidden):
        self.register_buffer('reference_h', torch.zeros(5, hidden))
        self.register_buffer('reference_meta', torch.zeros(5, hidden))
        self.register_buffer('reference_ready', torch.tensor(False))
        self.reference_fit_ids = None
        self.last_population_adjacency = None

    def _soft_adjacency(self, h, meta, ref_h, ref_meta):
        eps = self.config['normalization_eps']
        s = .5 * (F.normalize(h, dim=-1, eps=eps) @ F.normalize(ref_h, dim=-1, eps=eps).T)
        s = s + .5 * (F.normalize(meta, dim=-1, eps=eps) @ F.normalize(ref_meta, dim=-1, eps=eps).T)
        k = min(int(self.config['population_k']), len(ref_h))
        if k < 1:
            raise ValueError('Population neighborhood must be nonempty')
        values, indices = s.topk(k, dim=-1)
        return torch.zeros_like(s).scatter(-1, indices, values.softmax(-1))

    def _population_fuse(self, h, meta, layer, legacy_dropout=None):
        mode = self.config['pop_fusion']
        if mode == 'off':
            self.last_population_adjacency = None
            return h + meta
        if mode == 'original':
            hn = F.normalize(h, dim=-1)
            mn = F.normalize(meta, dim=-1)
            a = (((hn @ hn.T).sigmoid() * (mn @ mn.T).sigmoid()) > self.config['population_threshold']).float()
            if legacy_dropout is not None:
                a = legacy_dropout(a)
            self.last_population_adjacency = a.detach()
            return layer(h, a) + h + meta
        if self.training:
            ref_h, ref_meta = h, meta
        else:
            if not bool(self.reference_ready):
                raise RuntimeError('Refresh TRAIN-only population reference before evaluation')
            ref_h, ref_meta = self.reference_h, self.reference_meta
        a = self._soft_adjacency(h, meta, ref_h, ref_meta)
        self.last_population_adjacency = a.detach()
        # Same simpleGCNLayer equation with a rectangular TRAIN reference bank.
        message = a @ (ref_h @ layer.weight) + layer.bias
        message = layer.activation(message) if layer.activation else message
        if layer.residual:
            message = message + h
        return layer.dropout(message) + h + meta

    @torch.no_grad()
    def refresh_reference(self, data, fit_ids, device, batch_size):
        if self.config['pop_fusion'] != 'soft_topk':
            return
        ids = [int(i) for i in fit_ids]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Reference IDs must be nonempty and unique')
        labels = data['y'][ids].to(device)
        if set(labels.cpu().tolist()) != set(range(5)):
            raise ValueError('TRAIN reference must contain five classes')
        groups = [str(data['groups'][i]) if 'groups' in data else str(i) for i in ids]
        count = Counter(groups)
        weights = torch.tensor([1 / count[g] for g in groups], device=device)
        was_training = self.training
        self.eval()
        hs, metas = [], []
        try:
            for j in range(0, len(ids), batch_size):
                h, meta = self._reference_embeddings(data, ids[j:j+batch_size], device)
                hs.append(h)
                metas.append(meta)
            h, meta = torch.cat(hs), torch.cat(metas)
            for c in range(5):
                selected = labels == c
                w = weights[selected, None]
                self.reference_h[c].copy_((h[selected] * w).sum(0) / w.sum())
                self.reference_meta[c].copy_((meta[selected] * w).sum(0) / w.sum())
            self.reference_ready.fill_(True)
            self.reference_fit_ids = tuple(ids)
        finally:
            self.train(was_training)


class BrainPromptG(PopulationReferenceMixin, nn.Module):
    def __init__(self, config, data):
        super().__init__()
        self.config = resolved_config(config)
        c = self.config
        p = dict(in_dim=116, edge_dim=1, hidden_dim=c['hidden_dim'], out_dim=c['hidden_dim'],
                 n_classes=5, in_feat_dropout=0., dropout=c['dropout'], L=c['layers'],
                 readout=c['readout'], batch_norm=c['batch_norm'], residual=c['residual'],
                 edge_feat=False, node_num=116, label_embs=data['disease'], lambda1=1.,
                 roi_prompt=data['semantic'])
        self.core = BrainPromptGNet(p)
        self.core.register_buffer('disease_prompts', data['disease'].detach().float().clone())
        # The original plain tensor is replaced by an explicit buffer reference
        # before every forward, so moving model devices is safe.
        self._initialize_reference(c['hidden_dim'])
        self.ordinal_head = OrderedCumulativeHead(c['hidden_dim'], c['ordinal_threshold_start'], c['ordinal_threshold_gap']) if c['ordinal'] else None
        self.signed_fusion = nn.Linear(2*c['hidden_dim'], c['hidden_dim'], bias=False) if c['signed'] else None
        if self.signed_fusion is not None:
            with torch.no_grad():
                self.signed_fusion.weight.copy_(torch.cat([torch.eye(c['hidden_dim']), -torch.eye(c['hidden_dim'])], 1) / 2)

    def _one_graph(self, features, adjacency):
        g = graph_batch(features, adjacency)
        h = self.core.embedding_h(g.ndata['feat'])
        hidden = h.shape[-1]
        h = self.core.pos_emb(h.reshape(-1, 116, hidden)).reshape(-1, hidden)
        e = g.edata['feat']
        for layer in self.core.layers:
            h, e = layer(g, h, e)
        g.ndata['h'] = h
        return {'sum':dgl.sum_nodes, 'max':dgl.max_nodes}.get(self.config['readout'], dgl.mean_nodes)(g, 'h')

    def _encode(self, fc, tta_drop=0.):
        c = self.config
        features = fisher_transform(fc, c['fisher_clip']) if c['fisher_z'] else fc
        ratios = c['edge_ratios'] if c['multiscale'] else [c['edge_ratio']]
        keep = symmetric_edge_keep(fc, tta_drop)
        outputs = []
        for ratio in ratios:
            adjacency = (density_adjacency(fc, ratio) if c['multiscale']
                         else legacy_adjacency(fc, ratio, c['fc_threshold'])) & keep
            if c['signed']:
                positive = self._one_graph(features, adjacency & (fc >= 0))
                negative = self._one_graph(features, adjacency & (fc < 0))
                outputs.append(self.signed_fusion(torch.cat([positive, negative], -1)))
            else:
                outputs.append(self._one_graph(features, adjacency))
        return torch.stack(outputs).mean(0) if len(outputs) > 1 else outputs[0]

    def _reference_embeddings(self, data, ids, device):
        return (self._encode(data['fc'][ids].to(device)),
                self.core.linear_layer1(data['subject_prompt'][ids].to(device).squeeze(1)))

    def forward_batch(self, data, ids, device, tta_drop=0., mask_subject=False):
        h = self._encode(data['fc'][ids].to(device), tta_drop)
        prompt = data['subject_prompt'][ids].to(device).squeeze(1)
        # Zero input is the documented metadata-missing control for original G.
        meta = self.core.linear_layer1(torch.zeros_like(prompt) if mask_subject else prompt)
        features = self._population_fuse(h, meta, self.core.global_gcs, self.core.dropout)
        self.core.fused_repr = features
        self.core.label_embs = self.core.disease_prompts
        self.core.label_reprs = self.core.label_transform(self.core.disease_prompts)
        disease = F.normalize(features, dim=-1, eps=self.config['normalization_eps']) @ F.normalize(self.core.label_reprs, dim=-1, eps=self.config['normalization_eps']).T
        return dict(logits=self.core.MLP_layer(features), disease_logits=disease,
                    ordinal_logits=self.ordinal_head(features) if self.ordinal_head else None,
                    features=features)


class FCBiasedLayer(nn.Module):
    def __init__(self, hidden, heads, feedforward, dropout, bias_initial, use_bias):
        super().__init__()
        if hidden % heads:
            raise ValueError('Transformer width must divide head count')
        self.heads, self.hidden, self.use_bias = heads, hidden, use_bias
        self.qkv = nn.Linear(hidden, 3*hidden)
        self.output = nn.Linear(hidden, hidden)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(nn.Linear(hidden, feedforward), nn.GELU(), nn.Dropout(dropout), nn.Linear(feedforward, hidden))
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.beta = nn.Parameter(torch.full((heads,), float(bias_initial)), requires_grad=use_bias)

    def forward(self, tokens, fc_bias):
        b, n, _ = tokens.shape
        q, k, v = self.qkv(self.norm1(tokens)).reshape(b, n, 3, self.heads, self.hidden//self.heads).permute(2,0,3,1,4).unbind(0)
        scores = (q @ k.transpose(-1,-2)) / math.sqrt(self.hidden//self.heads)
        if self.use_bias:
            scores = scores + self.beta[None,:,None,None] * fc_bias[:,None]
        attention = self.attention_dropout(scores.softmax(-1))
        value = (attention @ v).transpose(1,2).reshape(b,n,self.hidden)
        tokens = tokens + self.dropout(self.output(value))
        return tokens + self.dropout(self.feedforward(self.norm2(tokens)))


class BrainPromptT(PopulationReferenceMixin, nn.Module):
    def __init__(self, config, data):
        super().__init__()
        self.config = resolved_config(config)
        c, h = self.config, self.config['transformer_hidden']
        self.register_buffer('roi_prompt', data['semantic'].detach().float().clone())
        self.register_buffer('disease_prompt', data['disease'].detach().float().clone())
        self.fc_projection = nn.Linear(116,h)
        self.roi_projection = nn.Linear(2048,h)
        self.subject_projection = nn.Linear(2048,h)
        self.disease_projection = nn.Linear(2048,h)
        self.identity = nn.Parameter(torch.randn(116,h) / math.sqrt(h))
        self.roi_null = nn.Parameter(torch.zeros(h))
        self.subject_null = nn.Parameter(torch.zeros(h))
        self.encoder = nn.ModuleList([FCBiasedLayer(h,c['transformer_heads'],c['transformer_feedforward'],c['transformer_dropout'],c['fc_bias_initial'],c['fc_bias']) for _ in range(c['transformer_layers'])])
        self.final_norm = nn.LayerNorm(h)
        self.cross_attention = nn.MultiheadAttention(h,c['transformer_heads'],dropout=c['transformer_dropout'],batch_first=True)
        self.class_score = nn.Linear(h,1)
        self.population = simpleGCNLayer(h,F.relu,0.,batch_norm=False,residual=True)
        self.population_class_correction = nn.Linear(h,5,bias=False)
        self.ordinal_head = OrderedCumulativeHead(h,c['ordinal_threshold_start'],c['ordinal_threshold_gap']) if c['ordinal'] else None
        self._initialize_reference(h)
        self.last_roi_mask, self.last_subject_mask = None, None

    def _encode(self, fc, prompt, tta_drop=0., mask_subject=False):
        c = self.config
        # T has no sparse graph; edge TTA perturbs both FC row inputs and bias.
        fc = fc * symmetric_edge_keep(fc, tta_drop)
        z = fisher_transform(fc,c['fisher_clip'])
        features = z if c['fisher_z'] else fc
        b = len(fc)
        roi = self.roi_projection(self.roi_prompt).expand(b,-1,-1)
        roi_mask = (torch.rand(b,116,1,device=fc.device) < c['prompt_dropout'][0]) if self.training else torch.zeros(b,116,1,dtype=torch.bool,device=fc.device)
        roi = torch.where(roi_mask,self.roi_null[None,None],roi)
        subject = self.subject_projection(prompt)
        subject_mask = (torch.rand(b,1,device=fc.device) < c['prompt_dropout'][1]) if self.training else torch.zeros(b,1,dtype=torch.bool,device=fc.device)
        if mask_subject:
            subject_mask = torch.ones_like(subject_mask)
        subject = torch.where(subject_mask,self.subject_null[None],subject)
        self.last_roi_mask, self.last_subject_mask = roi_mask.detach(), subject_mask.detach()
        tokens = torch.cat([self.fc_projection(features) + roi + self.identity[None],subject[:,None]],1)
        bias = F.pad(z,(0,1,0,1))
        for layer in self.encoder:
            tokens = layer(tokens,bias)
        tokens = self.final_norm(tokens)
        queries = self.disease_projection(self.disease_prompt).expand(b,-1,-1)
        class_features,_ = self.cross_attention(queries,tokens[:,:116],tokens[:,:116],need_weights=False)
        class_features = class_features + queries
        return class_features,subject

    def _reference_embeddings(self, data, ids, device):
        classes,meta = self._encode(data['fc'][ids].to(device),data['subject_prompt'][ids].to(device).squeeze(1))
        return classes.mean(1),meta

    def forward_batch(self, data, ids, device, tta_drop=0., mask_subject=False):
        classes,meta = self._encode(data['fc'][ids].to(device),data['subject_prompt'][ids].to(device).squeeze(1),tta_drop,mask_subject)
        h = classes.mean(1)
        logits = self.class_score(classes).squeeze(-1)
        if self.config['pop_fusion'] == 'off':
            features = h
        else:
            features = self._population_fuse(h,meta,self.population)
            logits = logits + self.population_class_correction(features-h)
        disease = F.normalize(features,dim=-1,eps=self.config['normalization_eps']) @ F.normalize(self.disease_projection(self.disease_prompt),dim=-1,eps=self.config['normalization_eps']).T
        return dict(logits=logits,disease_logits=disease,
                    ordinal_logits=self.ordinal_head(features) if self.ordinal_head else None,
                    features=features)


def build_model(config: dict, data: dict) -> nn.Module:
    if data['semantic'].shape != (116,2048) or data['disease'].shape != (5,2048):
        raise ValueError('Expected AAL116 ROI and canonical five-class prompts')
    name = config.get('model',DEFAULTS['model'])
    if name == 'BrainPromptG':
        return BrainPromptG(config,data)
    if name == 'BrainPromptT':
        return BrainPromptT(config,data)
    raise ValueError(f'Unknown model {name}')
