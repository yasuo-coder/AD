import torch
from .common import fixture
from BrainPromptPro.runtime import output_path,write_json


def test_classification_only_gradients():
    report={}
    for gate in ('global','subject_conditioned'):
        m,x,_=fixture(gate=gate)
        out=m(**x)
        torch.nn.functional.cross_entropy(out['logits'],torch.arange(4)).backward()
        modules={'brain_edge_scorer':m.adaptive_brain.edge_mlp,
                 'disease_query':m.disease_readout.q_projection,
                 'disease_key':m.disease_readout.k_projection,
                 'disease_value':m.disease_readout.v_projection,
                 **{r+'_relation_scorer':s for r,s in m.population.scorers.items()}}
        values={}
        for name,module in modules.items():
            grads=[p.grad for p in module.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads),name
            values[name]=sum(g.abs().sum().item() for g in grads)
            assert values[name]>0,name
        params=[m.population.relation_gate] if gate=='global' else list(m.population.relation_gate.parameters())
        values['relation_gate']=sum(p.grad.abs().sum().item() for p in params)
        assert values['relation_gate']>0
        report[gate]=values
    write_json(output_path('results','gradient_verification.json'),report)


def test_graph_weights_change_messages_and_receive_gradients():
    m,x,_=fixture()
    out=m(**x)
    out['A_brain'].retain_grad()
    for a in out['population_adjacencies'].values():a.retain_grad()
    torch.nn.functional.cross_entropy(out['logits'],torch.arange(4)).backward()
    for a in [out['A_brain'],*out['population_adjacencies'].values()]:
        assert a.grad is not None and a.grad.abs().sum()>0
