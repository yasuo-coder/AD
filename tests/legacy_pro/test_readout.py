import inspect
import torch
from BrainPromptPro.layers.disease_aware_readout import DiseaseAwareSemanticReadout


def test_all_disease_queries_without_labels():
    torch.manual_seed(2)
    m = DiseaseAwareSemanticReadout(16, 32, 12)
    h = torch.randn(4, 116, 16)
    prompt = torch.randn(7, 32)
    subject, logits, attention = m(h, prompt)
    assert subject.shape == (4, 16) and logits.shape == (4, 7)
    assert attention.shape == (4, 7, 116)
    torch.testing.assert_close(attention.sum(-1), torch.ones(4, 7))
    assert set(inspect.signature(m.forward).parameters) == {'h', 'disease_prompt'}
    torch.nn.functional.cross_entropy(torch.nn.Linear(16, 7)(subject), torch.arange(4)).backward()
    for projection in (m.q_projection, m.k_projection, m.v_projection):
        assert projection.weight.grad.abs().sum() > 0
