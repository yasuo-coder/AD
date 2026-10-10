"""BrainPromptPro: three independently switchable structural improvements."""
import torch
from torch import nn
from torch.nn import functional as F
from ..layers.adaptive_brain_graph import SemanticGuidedAdaptiveBrainGraph
from ..layers.disease_aware_readout import DiseaseAwareSemanticReadout
from ..layers.multirelation_population_graph import MultiRelationPopulationGraph
from ..layers.gcn_layer import (WeightedGCNLayer, LegacyPopulationGCN,
                               legacy_brain_adjacency, legacy_population_adjacency)
from ..layers.mlp_readout_layer import MLPReadout


class BrainPromptProNet(nn.Module):
    def __init__(self, config, roi_prompt, disease_prompt, subject_prompt_dim, num_sites):
        super().__init__()
        self.config = dict(config)
        self.name = 'BrainPromptPro'
        self.register_buffer('roi_prompt', roi_prompt.detach().clone())
        self.register_buffer('disease_prompt', disease_prompt.detach().clone())
        self.node_num = roi_prompt.shape[0]
        self.num_classes = disease_prompt.shape[0]
        dim = config['hidden_dim']
        self.use_adaptive = config['use_adaptive_brain_graph']
        self.use_disease = config['use_disease_aware_readout']
        self.use_multi = config['use_multirelation_population_graph']
        if config.get('population_scope', 'batch') != 'batch':
            raise ValueError('Only population_scope=batch is implemented')
        if config.get('candidate_graph_mode', 'full') != 'full':
            raise ValueError('Adaptive graphs require full continuous adjacency')
        if config['readout'] not in ('mean', 'sum', 'max'):
            raise ValueError('readout must be mean/sum/max')
        if config['L'] < 1 or dim < 4:
            raise ValueError('L >= 1 and hidden_dim >= 4 required')
        if self.use_adaptive:
            self.adaptive_brain = SemanticGuidedAdaptiveBrainGraph(
                self.node_num, roi_prompt.shape[-1], dim, config['brain_edge_hidden_dim'],
                config['brain_edge_dropout'], config['pair_chunk_size'])
        else:
            self.embedding_h = nn.Linear(self.node_num, dim)
            self.roi_projection = nn.Linear(roi_prompt.shape[-1], dim)
        self.dropout = nn.Dropout(config['dropout'])
        self.brain_gnn = nn.ModuleList([WeightedGCNLayer(dim, dim, config['dropout'],
                                         config['batch_norm'], config['residual']) for _ in range(config['L'])])
        if self.use_disease:
            self.disease_readout = DiseaseAwareSemanticReadout(dim, disease_prompt.shape[-1],
                                                               config['disease_attention_dim'])
        else:
            self.label_transform = nn.Linear(disease_prompt.shape[-1], dim)
        self.subject_projection = nn.Linear(subject_prompt_dim, dim)
        if self.use_multi:
            self.population = MultiRelationPopulationGraph(
                dim, config['relation_hidden_dim'], num_sites=num_sites,
                categorical_dim=config['categorical_dim'], relation_gate=config['relation_gate'],
                relations=config['population_relations'])
        else:
            self.legacy_population = LegacyPopulationGCN(dim)
        self.predictor = MLPReadout(dim, self.num_classes)

    def encode_brain(self, fc):
        # FC and A_brain: [B,N,N]; h: [B,N,D].
        if fc.ndim != 3 or fc.shape[-2:] != (self.node_num, self.node_num):
            raise ValueError('FC must be [B,N,N] in the configured ROI order')
        if self.use_adaptive:
            adjacency, h = self.adaptive_brain(fc, self.roi_prompt)
        else:
            adjacency = legacy_brain_adjacency(fc, self.config['legacy_fc_threshold'], self.config['legacy_edge_ratio'])
            h = self.dropout(self.embedding_h(fc) + self.roi_projection(self.roi_prompt).unsqueeze(0))
        for layer in self.brain_gnn:
            h = layer(h, adjacency)
        return h, adjacency

    def forward(self, fc, subject_prompt, age, sex, site):
        # Labels are deliberately absent. Every subject uses all K queries.
        h, adjacency = self.encode_brain(fc)
        attention = None
        if self.use_disease:
            subject, disease_logits, attention = self.disease_readout(h, self.disease_prompt)
        else:
            readout = self.config['readout']
            subject = h.mean(1) if readout == 'mean' else h.sum(1) if readout == 'sum' else h.max(1).values
        meta = self.subject_projection(subject_prompt)
        relations, weights, legacy_adjacency = {}, None, None
        if self.use_multi:
            population, relations, weights = self.population(subject, age, sex, site)
        else:
            legacy_adjacency = self.dropout(legacy_population_adjacency(
                subject, meta, self.config['legacy_population_threshold']))
            population = self.legacy_population(subject, legacy_adjacency)
        fused = subject + population + meta
        if not self.use_disease:
            disease_logits = (F.normalize(fused, dim=-1, eps=1e-8) @
                              F.normalize(self.label_transform(self.disease_prompt), dim=-1, eps=1e-8).T)
        return {'logits': self.predictor(fused), 'disease_logits': disease_logits,
                'A_brain': adjacency, 'roi_features': h, 'roi_attention': attention,
                'subject_repr': subject, 'population_adjacencies': relations,
                'legacy_population_adjacency': legacy_adjacency,
                'relation_weights': weights, 'fused_repr': fused}
