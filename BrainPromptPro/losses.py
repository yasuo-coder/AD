"""Labels enter here only, after all graphs and queries have been computed."""
import itertools
import torch
from torch.nn import functional as F
from .layers.gcn_layer import off_diagonal


def _safe_mean(value, zero):
    return value.mean() if value.numel() else zero


def compute_losses(output, labels, weights, model_config):
    zero = output['logits'].sum() * 0.
    cls = F.cross_entropy(output['logits'], labels)
    disease = F.cross_entropy(output['disease_logits'], labels)
    brain = _safe_mean(off_diagonal(output['A_brain']), zero) if model_config['use_adaptive_brain_graph'] else zero
    relations = list(output['population_adjacencies'].values())
    pop = torch.stack([_safe_mean(off_diagonal(a), zero) for a in relations]).mean() if relations else zero
    # Penalize cosine similarity of off-diagonal relation weights; diagonal
    # self-loops are excluded. Optional and disabled by default.
    diversity = [F.cosine_similarity(off_diagonal(a).flatten(), off_diagonal(b).flatten(),
                                     dim=0, eps=1e-8).square()
                 for a, b in itertools.combinations(relations, 2) if a.shape[-1] > 1]
    div = torch.stack(diversity).mean() if diversity else zero
    total = (cls + weights['lambda_disease'] * disease + weights['lambda_brain_sparse'] * brain
             + weights['lambda_pop_sparse'] * pop + weights['lambda_relation_div'] * div)
    return {'total': total, 'cls': cls, 'disease': disease, 'brain_sparse': brain,
            'pop_sparse': pop, 'relation_div': div}
