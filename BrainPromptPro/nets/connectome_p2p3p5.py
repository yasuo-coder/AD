"""P2/P3/P5 around the unchanged official BrainPrompt ROI GCN.

P5 uses a separately frozen encoder checkpoint for both query and TRAIN-bank
representations. The classifier encoder remains trainable. No stale live cache.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .original_adapter import OriginalBrainPromptAdapter
from ..layers.gcn_layer import legacy_brain_adjacency
from ..layers.masked_connectome import masked_topology,SymmetricConnectionDecoder
from ..layers.task_token_fusion import TaskSpecificROIFusion,ParameterMatchedROIFusion
from ..layers.reliability_population import ReliabilityReferencePopulation,stability_reliability


def reference_features(encoder,fc,temperature=.05,views=3,mask_ratio=.1):
    """Common deterministic perturbations, independent of query batch/IDs."""
    from ..layers.masked_connectome import symmetric_fc_mask
    if encoder.training:raise ValueError('Reference encoder must be eval/frozen')
    g=torch.Generator(device='cpu').manual_seed(31013)
    masks=[]
    # Generate only visibility using a synthetic symmetric template; no cohort statistics.
    template=torch.eye(fc.shape[-1]).unsqueeze(0)
    for _ in range(views):
        _,m=symmetric_fc_mask(template,mask_ratio,g);masks.append(m.to(fc.device))
    with torch.no_grad():
        clean,_=encoder.encode_roi(fc)
        rep=[]
        for m in masks:
            visible=m.expand_as(fc);masked=torch.where(visible,fc,torch.zeros_like(fc))
            h,_=encoder.encode_roi(masked,visible)
            rep.append(h.mean(1))
        reliability,var=stability_reliability(torch.stack(rep,1),temperature)
    return clean.mean(1).detach(),reliability.detach(),var.detach()


class ConnectomeBrainPrompt(nn.Module):
    def __init__(self,config,roi_prompt,disease_prompt,modules=None,bank=None,reference_encoder=None):
        super().__init__();self.options=dict(modules or {})
        for key,expected in [('task_token_dim',32),('num_attention_heads',1),('task_attention_layers',1)]:
            if self.options.get(key,expected)!=expected:raise ValueError(f'{key}: initial M2 reuse supports {expected}; register a separate attention variant')
        self.use_p2=bool(self.options.get('use_p2',False));self.use_p3=bool(self.options.get('use_p3',False));self.use_p5=bool(self.options.get('use_p5',False))
        self.base=OriginalBrainPromptAdapter(config,roi_prompt,disease_prompt)
        self.register_buffer('disease_prompt',disease_prompt.clone())
        self.register_buffer('roi_prompt',roi_prompt.clone())
        self.num_classes=5;self.accepts_sample_ids=True;d=config['net_params']['hidden_dim']
        with torch.random.fork_rng(devices=[]):
            if self.use_p2:
                self.visibility_projection=nn.Linear(116,d,bias=False)
                nn.init.zeros_(self.visibility_projection.weight)
                self.decoder=SymmetricConnectionDecoder(d,self.options.get('decoder','dot'),normalization=self.options.get('decoder_normalization','none'))
            if self.use_p3:
                mode=self.options.get('task_token_mode','semantic')
                self.task_fusion=ParameterMatchedROIFusion(d,disease_prompt.shape[-1]) if mode=='parameter_matched' else TaskSpecificROIFusion(d,disease_prompt.shape[-1],mode=mode)
            if self.use_p5:
                if bank is None or reference_encoder is None:raise ValueError('P5 needs sealed TRAIN bank and corresponding frozen encoder')
                self.reference_encoder=reference_encoder.eval()
                for p in self.reference_encoder.parameters():p.requires_grad_(False)
                self.encoder_version=bank['encoder_version']
                self.population=ReliabilityReferencePopulation(bank['representation'],bank['reliability'],bank['reference_ids'],bank['train_ids'],
                    self.encoder_version,d,topk=self.options.get('reference_topk',16),use_reliability=self.options.get('use_reliability',True),learn_edges=self.options.get('learn_edges',True))
    def train(self,mode=True):
        super().train(mode)
        if self.use_p5:self.reference_encoder.eval()
        return self
    def encode_roi(self,fc,visible=None):
        import dgl
        ori=self.base.original;b,n,_=fc.shape
        adjacency=legacy_brain_adjacency(fc,self.base.config['legacy_fc_threshold'],self.base.config['legacy_edge_ratio']) if visible is None else masked_topology(fc,visible,self.base.config['legacy_edge_ratio'])
        graphs=[]
        for x,a in zip(fc,adjacency):
            src,dst=a.nonzero(as_tuple=True);g=dgl.graph((src,dst),num_nodes=n,device=fc.device)
            g.ndata['feat']=x;g.edata['feat']=x[src,dst,None];graphs.append(g)
        g=dgl.batch(graphs);h=ori.embedding_h(g.ndata['feat']);d=h.shape[-1]
        if visible is not None and self.use_p2:
            h=h+self.visibility_projection((~visible).to(fc.dtype)).reshape(-1,d)
        h=ori.pos_emb(h.reshape(b,n,d)).reshape(-1,d);e=g.edata['feat']
        for conv in ori.layers:h,e=conv(g,h,e)
        return h.reshape(b,n,d),adjacency
    def forward(self,fc,subject_prompt,age,sex,site,sample_ids=None,visible=None,reconstruct=False):
        if not any([self.use_p2,self.use_p3,self.use_p5]) and visible is None:
            out=self.base(fc,subject_prompt,age,sex,site)
            out.update(patient_embedding=out['fused_repr'],auxiliary_losses={},diagnostics={})
            return out
        ori=self.base.original;roi,adj=self.encode_roi(fc,visible);subject=roi.mean(1)
        diagnostics={};token_logits=None;attention=None
        if self.use_p3:
            task=self.task_fusion(roi,self.disease_prompt);subject=subject+task['residual'];token_logits=task['logits'];attention=task['attention']
            diagnostics.update(disease_tokens=task['tokens'],roi_attention=attention,task_logits=token_logits)
        meta=ori.linear_layer1(subject_prompt.squeeze(1))
        legacy=ori.dropout(((ori.sim(subject)*ori.sim(meta))>.8).to(fc.dtype))
        fused=ori.global_gcs(subject,legacy)+subject+meta
        relations={}
        if self.use_p5:
            if sample_ids is None:raise ValueError('Sample IDs required only for self-reference masking')
            q,r,var=reference_features(self.reference_encoder,fc,self.options.get('reliability_temperature',.05))
            pop=self.population(q,r,sample_ids,self.encoder_version);fused=fused+pop['residual'];diagnostics['population']=pop;diagnostics['reliability_variance']=var
            relations={'reference':pop['attention']}
        logits=ori.MLP_layer(fused)
        if token_logits is not None:logits=logits+token_logits
        disease_logits=F.normalize(fused,dim=-1)@F.normalize(ori.label_transform(self.disease_prompt),dim=-1).T
        if reconstruct:
            if not self.use_p2:raise ValueError('Reconstruction requires P2')
            diagnostics['reconstruction']=self.decoder(roi)
        return dict(logits=logits,disease_logits=disease_logits,patient_embedding=fused,auxiliary_losses={},diagnostics=diagnostics,
            roi_features=roi,roi_attention=attention,A_brain=adj,population_adjacencies=relations,fused_repr=fused,legacy_population_adjacency=legacy)
