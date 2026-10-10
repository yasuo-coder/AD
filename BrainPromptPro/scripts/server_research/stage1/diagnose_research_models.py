"""Train/validation diagnostics only; never constructs a test loader."""
import argparse,itertools,json,math,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
sys.dont_write_bytecode=True
sys.path.insert(0,'/home/panyao/AD')
from BrainPromptPro.research_runner import RUN,load_context,build_model,evaluate,export_payload,dump
from BrainPromptPro.data.brainnet_pro import FoldDataset,sha256
from BrainPromptPro.data.metadata import MetadataEncoder
from BrainPromptPro.train_TUs_graph_classification_llm import model_inputs

parser=argparse.ArgumentParser();parser.add_argument('--experiment',required=True);args=parser.parse_args()
path=Path(args.experiment);out=path/'diagnostics';out.mkdir(exist_ok=False)
torch.set_num_threads(2)
cfg,protocol,data,split=load_context()
ck=torch.load(path/'best_validation.pt',weights_only=True,map_location='cpu')
enc=MetadataEncoder.from_state_dict(ck['metadata_encoder'])
model,cfg=build_model(ck['model_name'],ck['config'],data,enc,ck['spec']);model.load_state_dict(ck['model']);model.eval()
selected=[]
for y in range(5):selected.extend([i for i in split['train'] if int(data.labels[i])==y][:4])
batch=next(iter(DataLoader(FoldDataset(data,selected,enc),batch_size=20)));inputs=model_inputs(batch,'cpu')
capture={};handles=[]
def tap(module,key):
    def hook(_,a,value):capture[key]=value
    handles.append(module.register_forward_hook(hook))
if ck['model_name']=='BrainPrompt':
    def gnn_hook(_,a,value):capture['roi_raw']=value[0]
    handles.append(model.original.layers[-1].register_forward_hook(gnn_hook))
    def pop_hook(_,a,value):capture.update(subject=a[0],population_adj=a[1],population=value)
    handles.append(model.original.global_gcs.register_forward_hook(pop_hook))
    tap(model.original.linear_layer1,'metadata')
else:
    tap(model.subject_projection,'metadata')
    tap(model.population,'population_full')
o=model(**inputs)
for h in handles:h.remove()
roi_raw=capture['roi_raw'] if 'roi_raw' in capture else o['roi_features']
roi=roi_raw.reshape(20,116,-1)
subject=capture['subject'] if 'subject' in capture else o['subject_repr']
off=~torch.eye(116,dtype=torch.bool);popoff=~torch.eye(len(selected),dtype=torch.bool)
a=o['A_brain'];edges=a[:,off];fc=inputs['fc'][:,off]
report=dict(checkpoint_sha256=sha256(path/'best_validation.pt'),selection='Training class-balanced diagnostic batch: four examples/class',
            sample_indices=selected,model=ck['model_name'],profile=ck['supporting_profile'],seed=ck['seed'],
            graph=dict(mean=float(edges.mean()),std=float(edges.std()),across_sample_edge_std=float(edges.std(0).mean()),
                       density=float((edges!=0).float().mean()),near_zero=float((edges<.05).float().mean()),near_one=float((edges>.95).float().mean()),
                       mean_correlation_with_signed_fc=float(np.mean([np.corrcoef(x.detach().numpy(),y.numpy())[0,1] for x,y in zip(edges,fc)]))),
            roi_mean_cosine=float((F.normalize(roi,dim=-1)@F.normalize(roi,dim=-1).transpose(-1,-2))[:,off].mean()),
            branch_norms=dict(subject=float(subject.norm(dim=-1).mean()),metadata=float(capture['metadata'].norm(dim=-1).mean()),fused=float(o['fused_repr'].norm(dim=-1).mean())))
cls=F.cross_entropy(o['logits'],batch['label']);aux=F.cross_entropy(o['disease_logits'],batch['label'])
g1=torch.autograd.grad(cls,roi_raw,retain_graph=True)[0];g2=torch.autograd.grad(aux,roi_raw,retain_graph=True)[0]
report['roi_loss_gradient']=dict(classification_norm=float(g1.norm()),disease_auxiliary_norm=float(g2.norm()),
                                cosine=float(F.cosine_similarity(g1.flatten(),g2.flatten(),dim=0)))
if o['roi_attention'] is not None:
    att=o['roi_attention'];att.retain_grad()
    similarities=F.normalize(att,dim=-1)@F.normalize(att,dim=-1).transpose(-1,-2)
    report['attention']=dict(normalized_entropy=float((-(att*att.clamp_min(1e-12).log()).sum(-1)/math.log(116)).mean()),
                             disease_query_cosine=similarities.mean(0).detach().tolist(),
                             disease_pair_l1=float(torch.stack([(att[:,i]-att[:,j]).abs().sum(-1).mean() for i,j in itertools.combinations(range(5),2)]).mean()),
                             top_roi_by_query={c:[data.roi_names[i] for i in att[:,k].mean(0).topk(10).indices] for k,c in enumerate(data.class_names)})
    if hasattr(model,'metadata_gate'):
        report['fusion_gates']=dict(metadata=float(model.metadata_gate.sigmoid()),population=float(model.population_gate.sigmoid()),readout=float(model.disease_readout.readout_gate.sigmoid()))
    relations=o['population_adjacencies'];weights=o['relation_weights']
    report['population']=dict(relation_weights_mean=weights.mean(0).detach().tolist(),relation_names=list(relations),
        relation_entropy=float((-(weights*weights.clamp_min(1e-12).log()).sum(-1)/math.log(3)).mean()),
        edges={k:dict(mean=float(v[popoff].mean()),std=float(v[popoff].std()),density=float((v[popoff]!=0).float().mean())) for k,v in relations.items()},
        pair_cosine={a+'_'+b:float(F.cosine_similarity(relations[a][popoff],relations[b][popoff],dim=0)) for a,b in itertools.combinations(relations,2)})
else:report['population']=dict(binary_nonzero=int(capture['population_adj'].count_nonzero()))
model.zero_grad();cls.backward()
grad={}
for name,p in model.named_parameters():
    if any(token in name for token in ('edge_mlp','q_projection','k_projection','v_projection','population.scorers','relation_gate','population_gate','metadata_gate')):
        grad[name]=None if p.grad is None else float(p.grad.norm())
report['classification_only_gradient_norms']=grad
report['disease_prompt_embedding_grad']='Frozen LLM embedding buffer, intentionally not trainable; inspect query/key/value projection gradients instead'
if o['roi_attention'] is not None:report['attention_gradient_norm']=None if o['roi_attention'].grad is None else float(o['roi_attention'].grad.norm())
tensor_out=dict(sample_indices=selected,diagnoses=batch['label'],A_brain=a.detach(),roi_attention=None if o['roi_attention'] is None else o['roi_attention'].detach(),
                population_adjacencies={k:v.detach() for k,v in o['population_adjacencies'].items()})
torch.save(tensor_out,out/'training_structure.pt')
weights=ck['class_weights']
loader=DataLoader(FoldDataset(data,split['val'],enc),batch_size=20,shuffle=False)
_,p=evaluate(model,loader,'cpu',cfg,weights,capture=True)
perm=np.random.RandomState(123).permutation(len(split['val']));changed=[split['val'][i] for i in perm]
_,q=evaluate(model,DataLoader(FoldDataset(data,changed,enc),batch_size=20,shuffle=False),'cpu',cfg,weights)
q=q['probability'][np.argsort(perm)];p=p['probability']
report['validation_batch_composition_sensitivity']=dict(mean_probability_l1=float((p-q).abs().sum(-1).mean()),changed_predictions=int((p.argmax(-1)!=q.argmax(-1)).sum()))
dump(out/'module_diagnostics.json',report)
print(json.dumps(report,indent=2))
