import copy,itertools,json,math,os,sys
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
from BrainPromptPro.metrics import classification_metrics
from BrainPromptPro.train_TUs_graph_classification_llm import model_inputs
from BrainPromptPro.layers.gcn_layer import off_diagonal

torch.set_num_threads(2)
root=Path('/home/panyao/AD/BrainPromptPro/outputs/comparisons/adni123_official_scan10_last_seed41_20261005_v2')
target=Path('/data2/panyao_data/ADNI/model_audits/pro_optimization_20261006')
config=json.loads((root/'config.json').read_text());data=LoadData(config['data']);results=[]
device='cuda:0'

@torch.no_grad()
def evaluate(model,indices,encoder):
    model.eval();probs=[];ys=[]
    for batch in DataLoader(FoldDataset(data,indices,encoder),batch_size=20):
        probs.append(model(**model_inputs(batch,device))['logits'].softmax(-1).cpu());ys.append(batch['label'])
    p=torch.cat(probs).numpy();y=torch.cat(ys).numpy()
    return classification_metrics(y,p,5),p

for fold in range(10):
    ck=torch.load(root/'BrainPromptPro'/f'fold_{fold}'/'last.pt',weights_only=True,map_location='cpu')
    encoder=MetadataEncoder.from_state_dict(ck['metadata_encoder']);split=ck['split']
    model=gnn_model('BrainPromptPro',ck['config']['net_params'],data.roi_prompt,data.disease_prompt,2048,len(encoder.site_mapping)+1).to(device)
    model.load_state_dict(ck['model']);model.eval()
    val,p=evaluate(model,split['val'],encoder)
    # Structural statistics on a balanced diagnostic batch from TRAIN only.
    chosen=[]
    for label in range(5):
        chosen.extend([i for i in split['train'] if int(data.labels[i])==label][:4])
    batch=next(iter(DataLoader(FoldDataset(data,chosen,encoder),batch_size=20)))
    inp=model_inputs(batch,device)
    with torch.no_grad():
        o=model(**inp);a=off_diagonal(o['A_brain']);att=o['roi_attention'];h=o['roi_features']
        relation=list(o['population_adjacencies'].values())
        subject=o['subject_repr'];meta=model.subject_projection(inp['subject_prompt']);pop=o['fused_repr']-subject-meta
        stat=dict(brain_edge_mean=a.mean().item(),brain_edge_std=a.std().item(),brain_subject_edge_std=a.std(0).mean().item(),
                  attention_entropy=(-(att*att.clamp_min(1e-12).log()).sum(-1)/math.log(116)).mean().item(),
                  class_attention_mean_l1=float(torch.stack([(att[:,i]-att[:,j]).abs().sum(-1).mean() for i,j in itertools.combinations(range(5),2)]).mean()),
                  roi_cosine=float(off_diagonal(F.normalize(h,dim=-1)@F.normalize(h,dim=-1).transpose(-1,-2)).mean()),
                  relation_cosines=[F.cosine_similarity(off_diagonal(x),off_diagonal(y),dim=0).item() for x,y in itertools.combinations(relation,2)],
                  subject_norm=subject.norm(dim=-1).mean().item(),meta_norm=meta.norm(dim=-1).mean().item(),population_norm=pop.norm(dim=-1).mean().item())
    recal=copy.deepcopy(model);bn=[m for m in recal.modules() if isinstance(m,torch.nn.BatchNorm1d)]
    for m in bn:m.reset_running_stats();m.momentum=None;m.train()
    with torch.no_grad():
        for b in DataLoader(FoldDataset(data,split['train'],encoder),batch_size=20):recal(**model_inputs(b,device))
    calibrated,_=evaluate(recal,split['val'],encoder)
    del recal
    perm=np.random.RandomState(71).permutation(len(split['val']));shuffle=[split['val'][i] for i in perm]
    _,q=evaluate(model,shuffle,encoder);q=q[np.argsort(perm)]
    row=dict(fold=fold,validation=val,bn_recalibrated_validation=calibrated,structure=stat,
             validation_batch_shuffle_probability_l1=float(np.abs(q-p).sum(1).mean()),
             validation_batch_shuffle_changed_labels=int((q.argmax(1)!=p.argmax(1)).sum()))
    results.append(row);print(json.dumps(row),flush=True)
    (target/'last_checkpoint_diagnostics.json').write_text(json.dumps(results,indent=2)+'\n')
    del model,ck
