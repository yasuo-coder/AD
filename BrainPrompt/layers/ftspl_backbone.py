"""BiomedCLIP adapter for fTSPL paper reconstruction (OpenCLIP 2.23).

Frozen encoders still run with autograd so prompt gradients are preserved.
Pretrained weights are mandatory in production; random backbones are test-only.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

BIOMED_MODEL = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'


def load_ftspl_tokenizer(p):
    try:
        from open_clip.tokenizer import HFTokenizer
    except ImportError as exc:
        raise ImportError('fTSPL requires requirements-ftspl.txt optional dependencies') from exc
    return HFTokenizer(p.get('ftspl_tokenizer_path','microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'),
                       context_length=p.get('ftspl_context_length',256))


class BiomedCLIPBackbone(nn.Module):
    def __init__(self,p):
        super().__init__()
        checkpoint = p.get('ftspl_pretrained_path')
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError('fTSPL requires --ftspl_pretrained_path to a BiomedCLIP open_clip_pytorch_model.bin; '
                                    'see docs/BASELINES.md. Random pretrained weights are never used implicitly.')
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError('Install requirements-ftspl.txt before running fTSPL') from exc
        # Register the documented BiomedCLIP architecture without a network fetch.
        open_clip.factory._MODEL_CONFIGS['ftspl_biomedclip'] = {
            'embed_dim':512,
            'vision_cfg':{'timm_model_name':'vit_base_patch16_224','timm_model_pretrained':False,
                          'timm_pool':'','timm_proj':'linear','image_size':224},
            'text_cfg':{'hf_model_name':p.get('ftspl_tokenizer_path','microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'),
                        'hf_tokenizer_name':p.get('ftspl_tokenizer_path','microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'),
                        'hf_model_pretrained':False,'hf_proj_type':'mlp',
                        'hf_pooler_type':'cls_last_hidden_state_pooler','context_length':256}}
        model = open_clip.create_model('ftspl_biomedclip')
        state = torch.load(checkpoint,map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state,strict=True)
        self.visual,self.text = model.visual,model.text
        self.image_dim = self.visual.trunk.num_features
        self.text_dim = self.text.transformer.config.hidden_size
        self.output_dim = 512
        self.requires_grad_(False)

    def encode_image_tokens(self, tokens, prompts):
        vit = self.visual.trunk
        B,N,_ = tokens.shape
        # ROI tokens replace image patches. Interpolate the frozen patch-position
        # sequence to N ROIs; atlas order is fixed. Paper leaves this unspecified.
        pos = F.interpolate(vit.pos_embed[:,1:].transpose(1,2),size=N,mode='linear',align_corners=False).transpose(1,2)
        x = torch.cat((vit.cls_token.expand(B,-1,-1),tokens),1)
        x = x+torch.cat((vit.pos_embed[:,:1],pos),1)
        x = vit.norm_pre(vit.pos_drop(x))
        for i,block in enumerate(vit.blocks):
            if i<len(prompts):
                x = block(torch.cat((x,prompts[i].expand(B,-1,-1)),1))[:,:N+1]
            else:
                x = block(x)
        x = vit.norm(x)
        return self.visual.head(vit.fc_norm(x[:,0])),x[:,1:]

    def encode_text_tokens(self,tokens,prompts):
        bert = self.text.transformer
        B,T = tokens.shape
        x = bert.embeddings(input_ids=tokens)
        mask = tokens.ne(0)
        for i,layer in enumerate(bert.encoder.layer):
            if i<len(prompts):
                incoming = torch.cat((x,prompts[i].expand(B,-1,-1)),1)
                incoming_mask = torch.cat((mask,mask.new_ones(B,prompts.shape[1])),1)
            else:
                incoming,incoming_mask = x,mask
            extended = (~incoming_mask)[:,None,None,:].to(x)*torch.finfo(x.dtype).min
            x = layer(incoming,attention_mask=extended)[0][:,:T]
        # Paper Eq.(3): project final non-padding word token, not a disease token.
        idx = mask.sum(-1).clamp_min(1)-1
        return self.text.proj(x[torch.arange(B,device=x.device),idx])

    def train(self,mode=True):
        super().train(False)
        return self
