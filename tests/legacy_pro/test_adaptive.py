import torch
from BrainPromptPro.layers.adaptive_brain_graph import SemanticGuidedAdaptiveBrainGraph
from BrainPromptPro.layers.gcn_layer import WeightedGCNLayer


def test_adaptive_classification_gradient_and_isolation():
    torch.manual_seed(41)
    fc = torch.corrcoef(torch.randn(116, 60)).repeat(4, 1, 1)
    model = SemanticGuidedAdaptiveBrainGraph(116, 32, 16, pair_chunk_size=16)
    gcn = WeightedGCNLayer(16, 16, batch_norm=False)
    predictor = torch.nn.Linear(16, 5)
    a, h = model(fc, torch.randn(116, 32))
    assert a.shape == (4, 116, 116)
    out = gcn(h, a)
    loss = torch.nn.functional.cross_entropy(predictor(out.mean(1)), torch.arange(4))
    loss.backward()
    assert sum(p.grad.abs().sum() for p in model.edge_mlp.parameters()) > 0
    a2 = a.detach().clone(); a2[0] = torch.eye(116)
    out2 = gcn(h.detach(), a2)
    torch.testing.assert_close(out[1:].detach(), out2[1:])
    assert not torch.allclose(out[0].detach(), out2[0])
