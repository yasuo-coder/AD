import copy
import torch
from torch.nn import functional as F
from tests.legacy_pro.common import fixture
from BrainPromptPro.nets.load_net import gnn_model
from BrainPromptPro.optimization_cv import class_weights,optimization_losses


def stable_fixture(flags=(True,True,True)):
    old,inputs,config=fixture(batch=5,nodes=12,flags=flags)
    config['net_params']['stability_profile']='stable_v1'
    model=gnn_model('BrainPromptPro',config['net_params'],old.roi_prompt,old.disease_prompt,32,4)
    return model,inputs,config


def test_all_stable_modules_receive_classification_gradients():
    model,x,c=stable_fixture();o=model(**x);y=torch.arange(5)
    F.cross_entropy(o['logits'],y).backward()
    for path in ['adaptive_brain.edge_mlp.0.weight','disease_readout.q_projection.weight',
                 'population.scorers.age.0.weight','population.scorers.sex.0.weight',
                 'population.scorers.site.0.weight','population.relation_gate.0.weight',
                 'population_gate','metadata_gate']:
        grad=dict(model.named_parameters())[path].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,path
    assert o['logits'].shape==(5,5)
    assert o['roi_attention'].shape==(5,5,12)
    torch.testing.assert_close(o['roi_attention'].sum(-1),torch.ones(5,5))
    for a in [o['A_brain'],*o['population_adjacencies'].values()]:
        torch.testing.assert_close(a,a.transpose(-1,-2))
        assert a.min()>=0 and a.max()<=1
    assert (o['A_brain'][0]-o['A_brain'][1]).abs().mean()>.001


def test_layernorm_has_no_running_statistics_or_batch_coupling_without_population():
    model,x,c=stable_fixture(flags=(True,True,False));model.eval()
    assert not any(isinstance(m,torch.nn.BatchNorm1d) for m in model.modules())
    with torch.no_grad():
        all_logits=model(**x)['logits']
        single=torch.cat([model(**{k:v[i:i+1] for k,v in x.items()})['logits'] for i in range(5)])
    torch.testing.assert_close(all_logits,single,atol=2e-5,rtol=2e-5)


def test_class_weights_depend_only_on_given_training_labels_and_zero_power_is_uniform():
    y=torch.tensor([0]*100+[1]*10+[2]*20+[3]*10+[4]*40)
    w=class_weights(y,5,.5)
    assert w[1]>w[0]
    torch.testing.assert_close((w*torch.bincount(y)/len(y)).sum(),torch.tensor(1.))
    torch.testing.assert_close(class_weights(y,5,0),torch.ones(5))
    model,x,c=stable_fixture();o=model(**x);labels=torch.arange(5)
    loss=optimization_losses(o,labels,c['loss'],model.config,w,.05)
    torch.testing.assert_close(loss['cls'],F.cross_entropy(o['logits'],labels,weight=w,label_smoothing=.05))


def test_stable_model_learns_a_small_training_batch():
    torch.manual_seed(7);model,x,c=stable_fixture();model.train()
    labels=torch.arange(5);optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    with torch.no_grad():before=F.cross_entropy(model(**x)['logits'],labels).item()
    for _ in range(200):
        optimizer.zero_grad();loss=F.cross_entropy(model(**x)['logits'],labels);loss.backward();optimizer.step()
    model.eval()
    with torch.no_grad():
        logits=model(**x)['logits'];after=F.cross_entropy(logits,labels).item()
    assert after<before*.6,(before,after)
    assert (logits.argmax(1)==labels).sum()>=4
