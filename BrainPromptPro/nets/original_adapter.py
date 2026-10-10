"""Optional read-only original BrainPrompt-G on the same audited Pro splits.

This imports the original model unchanged. Only its input/output contract is
adapted so comparisons use identical samples, folds and validation selection.
"""
import os
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from ..layers.gcn_layer import legacy_brain_adjacency
from ..runtime import ROOT, output_path


class OriginalBrainPromptAdapter(nn.Module):
    def __init__(self, config, roi_prompt, disease_prompt):
        super().__init__()
        self.name='OriginalBrainPromptG'
        self.config=dict(config['net_params'])
        self.config.update(use_adaptive_brain_graph=False,use_disease_aware_readout=False,
                           use_multirelation_population_graph=False)
        self.use_adaptive=self.use_disease=self.use_multi=False
        self.num_classes=disease_prompt.shape[0]
        if roi_prompt.shape != (116,2048) or disease_prompt.shape[-1]!=2048:
            raise ValueError('Original model hardcodes AAL116 and 2048-dimensional prompts')
        p=dict(self.config,in_dim=116,edge_dim=1,node_num=116,n_classes=self.num_classes,
               out_dim=self.config['hidden_dim'],edge_feat=False,in_feat_dropout=0.,
               lambda1=config['loss']['lambda_disease'],label_embs=disease_prompt)
        root=Path(config['baseline_root']);cwd=Path.cwd()
        os.environ['MPLCONFIGDIR']=str(output_path('cache','matplotlib'))
        previous_path=list(sys.path)
        # The original nets/layers directories are namespace packages. Remove
        # the CLI script directory so Pro's regular packages cannot shadow them.
        sys.path[:]=[str(root)]+[p for p in sys.path if Path(p).resolve()!=ROOT]
        try:
            os.chdir(root)
            from nets.brainpromptg_net import BrainPromptGNet
            self.original=BrainPromptGNet(p)
        finally:
            os.chdir(cwd);sys.path[:]=previous_path
        torch.testing.assert_close(self.original.pos_emb.init_emb,roi_prompt)

    def forward(self,fc,subject_prompt,age,sex,site):
        import dgl
        adjacency=legacy_brain_adjacency(fc,self.config['legacy_fc_threshold'],self.config['legacy_edge_ratio'])
        graphs=[]
        for x,a in zip(fc,adjacency):
            src,dst=a.nonzero(as_tuple=True)
            graph=dgl.graph((src,dst),num_nodes=fc.shape[-1],device=fc.device)
            graph.ndata['feat']=x;graph.edata['feat']=x[src,dst,None]
            graphs.append(graph)
        graph=dgl.batch(graphs)
        logits=self.original(graph,graph.ndata['feat'],graph.edata['feat'],subject_prompt)
        fused=self.original.fused_repr
        disease_logits=F.normalize(fused,dim=-1,eps=1e-8) @ F.normalize(self.original.label_reprs,dim=-1,eps=1e-8).T
        return dict(logits=logits,disease_logits=disease_logits,A_brain=adjacency,
                    roi_attention=None,relation_weights=None,population_adjacencies={},fused_repr=fused,
                    legacy_population_adjacency=None)
