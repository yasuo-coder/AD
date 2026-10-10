"""Actual original model is loaded read-only, including its original DGL layer."""
import json
import os
import sys
from pathlib import Path
import torch
from BrainPromptPro.runtime import ROOT,output_path,write_json
from BrainPromptPro.nets.brainprompt_pro_net import BrainPromptProNet
from BrainPromptPro.layers.gcn_layer import legacy_brain_adjacency


def test_original_forward_backward_and_exact_baseline_parity():
    import dgl
    config=json.loads((ROOT/'configs/adni_AAL116/BrainPromptPro_adni_AAL116.json').read_text())
    original=Path(config['baseline_root']);sys.path.insert(0,str(original))
    os.environ['MPLCONFIGDIR']=str(output_path('cache','matplotlib'))
    cwd=Path.cwd()
    try:
        os.chdir(original)
        from nets.brainpromptg_net import BrainPromptGNet
        p=json.loads((original/'configs/adni_AAL116/TUs_graph_classification_BrainPromptG_adni_AAL116_100k.json').read_text())['net_params']
        p.update(in_dim=116,edge_dim=1,n_classes=5,node_num=116,lambda1=1.,
                 label_embs=torch.stack(torch.load(config['data']['disease_prompt'],weights_only=True)))
        torch.manual_seed(41);official=BrainPromptGNet(p)
    finally:
        os.chdir(cwd);sys.path.remove(str(original))
    net=config['net_params'];net.update(use_adaptive_brain_graph=False,use_disease_aware_readout=False,
                                      use_multirelation_population_graph=False,dropout=0.)
    pro=BrainPromptProNet(net,official.pos_emb.init_emb,p['label_embs'],2048,3)
    pro.embedding_h.load_state_dict(official.embedding_h.state_dict())
    pro.roi_projection.load_state_dict(official.pos_emb.mlp.state_dict())
    pro.subject_projection.load_state_dict(official.linear_layer1.state_dict())
    pro.label_transform.load_state_dict(official.label_transform.state_dict())
    pro.predictor.load_state_dict(official.MLP_layer.state_dict())
    with torch.no_grad():
        pro.legacy_population.weight.copy_(official.global_gcs.weight)
        pro.legacy_population.bias.copy_(official.global_gcs.bias)
        for dst,src in zip(pro.brain_gnn,official.layers):
            dst.weight.copy_(src.conv.weight);dst.bias.copy_(src.conv.bias)
            dst.norm.load_state_dict(src.batchnorm_h.norm.state_dict())
    fc=torch.stack([torch.corrcoef(torch.randn(116,50)) for _ in range(4)])
    adjacency=legacy_brain_adjacency(fc,edge_ratio=.2)
    graphs=[]
    for x,a in zip(fc,adjacency):
        src,dst=a.nonzero(as_tuple=True);g=dgl.graph((src,dst),num_nodes=116)
        g.ndata['feat']=x;g.edata['feat']=x[src,dst,None];graphs.append(g)
    g=dgl.batch(graphs);llm=torch.randn(4,2048)
    inputs=dict(fc=fc,subject_prompt=llm,age=torch.randn(4),sex=torch.ones(4,dtype=torch.long),site=torch.zeros(4,dtype=torch.long))
    errors={}
    for training in (False,True):
        official.train(training);pro.train(training)
        old=official(g,g.ndata['feat'],g.edata['feat'],llm)
        new=pro(**inputs)
        torch.testing.assert_close(new['logits'],old,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(new['fused_repr'],official.fused_repr,atol=2e-5,rtol=2e-5)
        errors['train' if training else 'eval']=(new['logits']-old).abs().max().item()
        y=torch.arange(4)
        old_loss=official.loss(old,y)
        new_loss=torch.nn.functional.cross_entropy(new['logits'],y)+torch.nn.functional.cross_entropy(new['disease_logits'],y)
        torch.testing.assert_close(old_loss,new_loss,atol=2e-5,rtol=2e-5)
        official.zero_grad();pro.zero_grad();old_loss.backward();new_loss.backward()
        torch.testing.assert_close(pro.embedding_h.weight.grad,official.embedding_h.weight.grad,atol=2e-5,rtol=2e-4)
    assert new['legacy_population_adjacency'].count_nonzero()==0
    official.eval()
    out1=official(g,g.ndata['feat'],g.edata['feat'],llm)
    out2=official(g,g.ndata['feat'],torch.randn_like(g.edata['feat']),llm)
    torch.testing.assert_close(out1,out2)
    write_json(output_path('results','baseline_compatibility.json'),dict(max_logit_error=errors,
               original_initialize_forward_backward='PASS',edge_features_ignored=True,
               dropout_for_training_parity=0,official_commit='c85304a023636a2701d465e4cb0acb13b637127e'))
