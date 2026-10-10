import json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from BrainPromptPro.data.data import LoadData
from BrainPromptPro.data.brainnet_pro import FoldDataset
from BrainPromptPro.data.metadata import MetadataEncoder
from BrainPromptPro.nets.load_net import gnn_model
from BrainPromptPro.nets.original_adapter import OriginalBrainPromptAdapter
from BrainPromptPro.train_TUs_graph_classification_llm import model_inputs
torch.set_num_threads(2)
root=Path('/home/panyao/AD/BrainPromptPro/outputs/comparisons/adni123_official_scan10_last_seed41_20261005_v2')
out=Path('/data2/panyao_data/ADNI/model_audits/pro_optimization_20261006')
data=LoadData(json.loads((root/'config.json').read_text())['data']);rows=[]
for fold in range(10):
    item=dict(fold=fold)
    for name in ['BrainPrompt','BrainPromptPro']:
        ck=torch.load(root/name/f'fold_{fold}'/'last.pt',map_location='cpu',weights_only=True)
        encoder=MetadataEncoder.from_state_dict(ck['metadata_encoder'])
        indices=[]
        for y in range(5):indices.extend([i for i in ck['split']['train'] if int(data.labels[i])==y][:4])
        batch=next(iter(DataLoader(FoldDataset(data,indices,encoder),batch_size=20)))
        if name=='BrainPrompt':
            model=OriginalBrainPromptAdapter(ck['config'],data.roi_prompt,data.disease_prompt)
            captured=[]
            handle=model.original.global_gcs.register_forward_pre_hook(lambda m,x:captured.append(x[0]))
        else:model=gnn_model('BrainPromptPro',ck['config']['net_params'],data.roi_prompt,data.disease_prompt,2048,len(encoder.site_mapping)+1)
        model.load_state_dict(ck['model']);model.eval();o=model(**model_inputs(batch,'cpu'))
        cls=F.cross_entropy(o['logits'],batch['label']);disease=F.cross_entropy(o['disease_logits'],batch['label'])
        if name=='BrainPrompt':
            subject=captured[0];meta=model.original.linear_layer1(batch['subject_prompt']);handle.remove()
            item[name]=dict(subject_norm=subject.norm(dim=-1).mean().item(),meta_norm=meta.norm(dim=-1).mean().item(),classification_loss=cls.item(),disease_loss=disease.item())
        else:
            h=o['roi_features'];g1=torch.autograd.grad(cls,h,retain_graph=True)[0];g2=torch.autograd.grad(disease,h)[0]
            item[name]=dict(classification_loss=cls.item(),disease_loss=disease.item(),classification_h_gradient_norm=g1.norm().item(),disease_h_gradient_norm=g2.norm().item(),gradient_cosine=F.cosine_similarity(g1.flatten(),g2.flatten(),dim=0).item())
        del model,ck,o
    rows.append(item)
    (out/'branch_gradient_diagnostics.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
