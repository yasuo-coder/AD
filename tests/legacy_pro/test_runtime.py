from BrainPromptPro.runtime import output_path
from .common import fixture
from BrainPromptPro.losses import compute_losses
import torch


def test_output_cannot_escape_pro():
    try:
        output_path('..','..','out','test.json')
    except ValueError:
        pass
    else:
        raise AssertionError('Output escaped BrainPromptPro')


def test_singleton_regularizers_are_finite_and_graphs_subject_specific():
    m,x,c=fixture(batch=1,nodes=8)
    losses=compute_losses(m(**x),torch.zeros(1,dtype=torch.long),c['loss'],m.config)
    assert all(torch.isfinite(v) for v in losses.values())
    assert losses['pop_sparse']==0 and losses['relation_div']==0
    losses['total'].backward()
    m,x,_=fixture();m.eval();a=m(**x)['A_brain']
    assert not torch.allclose(a[0],a[1],atol=1e-6)
