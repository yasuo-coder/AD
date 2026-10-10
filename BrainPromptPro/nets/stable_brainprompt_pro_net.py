"""BrainPromptPro stable_v1: opt-in, preserves all three learnable modules."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .brainprompt_pro_net import BrainPromptProNet
from ..layers.stable_graphs import (LayerNormalizedGCN,StableAdaptiveBrainGraph,
                                    StableDiseaseReadout,StablePopulationGraph)
from ..layers.gcn_layer import legacy_population_adjacency


class StableBrainPromptProNet(BrainPromptProNet):
    def __init__(self,config,roi_prompt,disease_prompt,subject_prompt_dim,num_sites):
        super().__init__(config,roi_prompt,disease_prompt,subject_prompt_dim,num_sites)
        dim=config['hidden_dim']
        if self.use_adaptive:
            self.adaptive_brain=StableAdaptiveBrainGraph(self.node_num,roi_prompt.shape[-1],hidden_dim=dim,
                edge_hidden_dim=config['brain_edge_hidden_dim'],edge_dropout=config['brain_edge_dropout'],pair_chunk_size=config['pair_chunk_size'])
        self.brain_gnn=nn.ModuleList([LayerNormalizedGCN(dim,config['dropout'],config['residual']) for _ in range(config['L'])])
        if self.use_disease:
            self.disease_readout=StableDiseaseReadout(dim,disease_prompt.shape[-1],config['disease_attention_dim'])
        if self.use_multi:
            self.population=StablePopulationGraph(dim,config['relation_hidden_dim'],num_sites=num_sites,
                categorical_dim=config['categorical_dim'],relation_gate=config['relation_gate'],relations=config['population_relations'])
        self.subject_norm=nn.LayerNorm(dim)
        self.population_norm=nn.LayerNorm(dim)
        self.metadata_norm=nn.LayerNorm(dim)
        self.fusion_norm=nn.LayerNorm(dim)
        self.population_gate=nn.Parameter(torch.tensor(math.log(.1/.9)))
        self.metadata_gate=nn.Parameter(torch.tensor(math.log(.1/.9)))

    def forward(self,fc,subject_prompt,age,sex,site):
        h,adjacency=self.encode_brain(fc)
        attention=None
        if self.use_disease:
            subject,disease_logits,attention=self.disease_readout(h,self.disease_prompt)
        else:
            mode=self.config['readout']
            subject=h.mean(1) if mode=='mean' else h.sum(1) if mode=='sum' else h.max(1).values
        subject=self.subject_norm(subject)
        meta=self.metadata_norm(self.subject_projection(subject_prompt))
        relations,weights,legacy={},None,None
        if self.use_multi:
            population,relations,weights=self.population(subject,age,sex,site)
        else:
            legacy=legacy_population_adjacency(subject,meta,self.config['legacy_population_threshold'])
            population=self.legacy_population(subject,legacy)-subject
        fused=self.fusion_norm(subject+self.population_gate.sigmoid()*self.population_norm(population)
                               +self.metadata_gate.sigmoid()*meta)
        if not self.use_disease:
            disease_logits=F.normalize(fused,dim=-1)@F.normalize(self.label_transform(self.disease_prompt),dim=-1).T
        return dict(logits=self.predictor(self.dropout(fused)),disease_logits=disease_logits,
                    A_brain=adjacency,roi_features=h,roi_attention=attention,subject_repr=subject,
                    population_adjacencies=relations,legacy_population_adjacency=legacy,
                    relation_weights=weights,fused_repr=fused)
