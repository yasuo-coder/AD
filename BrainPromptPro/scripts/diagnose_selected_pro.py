"""Training-only structural diagnostics of the frozen selected checkpoints."""
import argparse,json,math,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from BrainPromptPro.optimization_cv import experiment
from BrainPromptPro.data.data import LoadData
from BrainPromptPro.data.brainnet_pro import FoldDataset
from BrainPromptPro.data.metadata import MetadataEncoder
from BrainPromptPro.nets.load_net import gnn_model
from BrainPromptPro.train_TUs_graph_classification_llm import model_inputs
from BrainPromptPro.layers.gcn_layer import off_diagonal

p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();root=experiment(a.run_id)
torch.set_num_threads(2);data=LoadData(json.loads((root/'config.json').read_text())['data']);rows=[]
for choice in json.loads((root/'selection.json').read_text())['folds']:
    ck=torch.load(choice['checkpoint'],weights_only=True,map_location='cpu');encoder=MetadataEncoder.from_state_dict(ck['metadata_encoder'])
    model=gnn_model('BrainPromptPro',ck['config']['net_params'],data.roi_prompt,data.disease_prompt,2048,len(encoder.site_mapping)+1).to('cuda:0');model.load_state_dict(ck['model']);model.eval()
    indices=[]
    for label in range(5):indices.extend([i for i in ck['split']['train'] if int(data.labels[i])==label][:4])
    batch=next(iter(DataLoader(FoldDataset(data,indices,encoder),batch_size=20)))
    with torch.no_grad():
        o=model(**model_inputs(batch,'cuda:0'));edges=off_diagonal(o['A_brain']);attention=o['roi_attention'];h=F.normalize(o['roi_features'],dim=-1)
        row=dict(fold=choice['fold'],candidate=choice['candidate'],brain_edge_mean=edges.mean().item(),
            brain_edge_std=edges.std().item(),brain_subject_edge_std=edges.std(0).mean().item(),
            attention_entropy=(-(attention*attention.clamp_min(1e-12).log()).sum(-1)/math.log(116)).mean().item(),
            roi_cosine=off_diagonal(h@h.transpose(-1,-2)).mean().item())
        if choice['candidate']!='legacy':row.update(metadata_gate=model.metadata_gate.sigmoid().item(),population_gate=model.population_gate.sigmoid().item(),disease_readout_gate=model.disease_readout.readout_gate.sigmoid().item())
        torch.save(dict(scan_indices=indices,subject_ids=batch['subject_id'],A_brain=o['A_brain'].cpu(),roi_attention=attention.cpu(),
                        population_adjacencies={k:v.cpu() for k,v in o['population_adjacencies'].items()},relation_weights=o['relation_weights'].cpu()),
                   root/'selected'/f"fold_{choice['fold']}"/'training_structure.pt')
        rows.append(row)
    del model,ck
(root/'selected_structure.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
