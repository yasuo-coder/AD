"""fTSPL paper reconstruction: deep multimodal prompts, PMFCC, GCN adapter.

The authors' public repository contains no implementation. Unspecified choices
are explicit in docs/BASELINES.md; this is not certified Table 2 reproduction.
"""
import torch
from torch import nn
from torch.nn import functional as F
from layers.baseline_utils import dense_batch,DenseGCN
from layers.ftspl_backbone import BiomedCLIPBackbone


def pearson_rows(a,b):
    a = F.normalize(a-a.mean(-1,keepdim=True),dim=-1)
    b = F.normalize(b-b.mean(-1,keepdim=True),dim=-1)
    return a @ b.transpose(-1,-2)


class FTSPLNet(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.node_num = p['node_num']
        self.time_steps = p.get('ftspl_time_steps',128)
        self.backbone = p.get('_test_ftspl_backbone')
        if self.backbone is None:
            self.backbone = BiomedCLIPBackbone(p)
        self.backbone.requires_grad_(False)
        depth,length = p.get('ftspl_prompt_depth',4),p.get('ftspl_prompt_length',2)
        if depth<1 or length<1:
            raise ValueError('fTSPL prompt depth/length must be positive')
        self.time_projection = nn.Linear(self.time_steps,self.backbone.image_dim)
        self.roi_projection = nn.Linear(self.backbone.image_dim,self.backbone.output_dim)
        self.image_prompts = nn.Parameter(torch.randn(depth,length,self.backbone.image_dim)*0.02)
        self.text_prompts = nn.Parameter(torch.randn(depth,length,self.backbone.text_dim)*0.02)
        # N ROI nodes plus one image and one text node, with N-dimensional profiles.
        self.gcn1 = DenseGCN(self.node_num,p.get('hidden_dim',116))
        self.gcn2 = DenseGCN(p.get('hidden_dim',116),p.get('out_dim',116))
        self.classifier = nn.Linear(p.get('out_dim',116),p['n_classes'])
        self.temperature = p.get('ftspl_temperature',0.07)
        if self.temperature<=0:
            raise ValueError('fTSPL temperature must be positive')
        self.contrast_weight = p.get('ftspl_contrast_weight',0.5)
        self.task_weight = p.get('ftspl_task_weight',0.5)
        self.image_repr = self.text_repr = None
        self.last_multimodal_fc = None

    def forward(self,g,h,e,batch_llms=None):
        fc,_ = dense_batch(g,h)
        if fc.shape[1:] != (self.node_num,self.node_num):
            raise ValueError('fTSPL expects Pearson FC node features [N,N]')
        if 'timeseries' not in g.ndata or 'ftspl_tokens' not in g.ndata:
            raise ValueError('fTSPL requires retained ROI time series and ABTG tokens; use its config with llm-main.py')
        sizes = g.batch_num_nodes().tolist()
        ts = torch.stack(g.ndata['timeseries'].split(sizes)).to(h)
        # Fixed configurable temporal resampling; never reconstruct time series from FC.
        ts = F.interpolate(ts,size=self.time_steps,mode='linear',align_corners=False)
        tokens = torch.stack(g.ndata['ftspl_tokens'].split(sizes))[:,0].long()
        self.backbone.eval()
        image,roi = self.backbone.encode_image_tokens(self.time_projection(ts),self.image_prompts)
        text = self.backbone.encode_text_tokens(tokens,self.text_prompts)
        projected = self.roi_projection(roi)
        image_fc = pearson_rows(image.unsqueeze(1),projected)
        text_fc = pearson_rows(text.unsqueeze(1),projected)
        fm = torch.cat((fc,image_fc,text_fc),dim=1)
        # Eq.(8) is rectangular. Complete a symmetric graph by retaining FC and
        # connecting the two added nodes to ROIs using Eq.(6)/(7); image-text is
        # their Pearson correlation. Absolute edge strengths stabilize GCN degree.
        B,N,_ = fc.shape
        adj = fc.new_zeros(B,N+2,N+2)
        adj[:,:N,:N] = fc
        adj[:,N:,:N] = fm[:,N:]
        adj[:,:N,N:] = fm[:,N:].transpose(1,2)
        cross = pearson_rows(image.unsqueeze(1),text.unsqueeze(1)).reshape(B)
        adj[:,N,N+1] = cross
        adj[:,N+1,N] = cross
        adj = adj.abs()
        x = F.relu(self.gcn1(fm,adj))
        x = F.relu(self.gcn2(x,adj))
        self.image_repr,self.text_repr = image,text
        self.last_multimodal_fc = fm.detach()
        return self.classifier(x.mean(1))

    def loss(self,pred,label):
        logits = F.normalize(self.image_repr,dim=-1) @ F.normalize(self.text_repr,dim=-1).T/self.temperature
        target = torch.arange(logits.shape[0],device=logits.device)
        contrast = (F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2
        return self.task_weight*F.cross_entropy(pred,label)+self.contrast_weight*contrast
