import json
import torch
from BrainPromptPro.runtime import ROOT
from BrainPromptPro.nets.brainprompt_pro_net import BrainPromptProNet


def fixture(batch=4, nodes=116, classes=5, prompt_dim=32, gate='subject_conditioned', flags=(True,True,True)):
    torch.manual_seed(41)
    config = json.loads((ROOT/'configs/adni_AAL116/BrainPromptPro_adni_AAL116.json').read_text())
    net = config['net_params']
    net.update(hidden_dim=16, disease_attention_dim=12, brain_edge_hidden_dim=16,
               relation_hidden_dim=16, dropout=0., relation_gate=gate)
    for name, value in zip(('use_adaptive_brain_graph','use_disease_aware_readout','use_multirelation_population_graph'),flags):
        net[name]=value
    roi = torch.randn(nodes,prompt_dim); disease = torch.randn(classes,prompt_dim)
    model=BrainPromptProNet(net,roi,disease,prompt_dim,4)
    fc=torch.stack([torch.corrcoef(torch.randn(nodes,40)) for _ in range(batch)])
    inputs=dict(fc=fc,subject_prompt=torch.randn(batch,prompt_dim),age=torch.linspace(-1,1,batch),
                sex=torch.arange(batch)%2+1,site=torch.arange(batch)%4)
    return model,inputs,config
