"""Training and evaluation with explicit feature inputs and final-only export."""
import csv
import numpy as np
import torch
from .metrics import classification_metrics
from .losses import compute_losses

INPUTS = ('fc', 'subject_prompt', 'age', 'sex', 'site')


def model_inputs(batch, device):
    return {name: batch[name].to(device) for name in INPUTS}


def train_epoch(model, loader, optimizer, device, loss_weights):
    model.train()
    totals, count = {}, 0
    for batch in loader:
        labels = batch['label'].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(**model_inputs(batch, device))
        losses = compute_losses(output, labels, loss_weights, model.config)
        if not torch.isfinite(losses['total']):
            raise FloatingPointError('Nonfinite training loss')
        losses['total'].backward()
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise FloatingPointError(f'Nonfinite gradient: {name}')
        optimizer.step()
        n = labels.numel(); count += n
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.) + value.detach().item() * n
    if not count:
        raise ValueError('Empty training loader')
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device, loss_weights, capture=False):
    model.eval()
    count, total = 0, 0.
    predictions, labels, ids, batches, sources = [], [], [], [], []
    for batch in loader:
        output = model(**model_inputs(batch, device))
        y = batch['label'].to(device)
        loss = compute_losses(output, y, loss_weights, model.config)['total']
        if not torch.isfinite(loss) or not torch.isfinite(output['logits']).all():
            raise FloatingPointError('Nonfinite evaluation output')
        total += loss.item() * len(y); count += len(y)
        predictions.append(output['logits'].softmax(-1).cpu())
        labels.append(y.cpu()); ids.extend(batch['subject_id'])
        sources.extend(batch.get('source_batch', ['unspecified']*len(y)))
        if capture:
            batches.append(dict(
                subject_ids=list(batch['subject_id']),
                adaptive_brain_edge_weights=output['A_brain'].cpu() if model.use_adaptive else None,
                disease_roi_attention=output['roi_attention'].cpu() if output['roi_attention'] is not None else None,
                relation_weights=output['relation_weights'].cpu() if output['relation_weights'] is not None else None,
                population_edges={r: a.cpu() for r, a in output['population_adjacencies'].items()},
                legacy_population_edges=output['legacy_population_adjacency'].cpu()
                    if output['legacy_population_adjacency'] is not None else None))
    if not count:
        raise ValueError('Empty evaluation loader')
    probability = torch.cat(predictions); truth = torch.cat(labels)
    pred = probability.argmax(-1)
    metrics = classification_metrics(truth.numpy(), probability.numpy(), model.num_classes)
    metrics['loss'] = total/count
    payload = dict(subject_ids=ids, true_label=truth, predicted_label=pred,
                   probability=probability, batches=batches, source_batches=sources)
    return metrics, payload


def save_interpretability(payload, path, class_names, roi_names, relation_names):
    # Population matrices remain batch-local with explicit row/column IDs.
    result = dict(payload, class_names=class_names, roi_names=roi_names, relation_names=relation_names,
                  population_scope='batch', note='Attention and learned weights are model explanations, not causal biomarkers.')
    for key in ('adaptive_brain_edge_weights', 'disease_roi_attention', 'relation_weights'):
        parts = [b[key] for b in payload['batches'] if b[key] is not None]
        result[key] = torch.cat(parts) if parts else None
    torch.save(result, path)
    with path.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject_id', 'true_label', 'predicted_label'] + ['p_' + c for c in class_names]
                        + ['relation_' + r for r in relation_names])
        for i, identity in enumerate(result['subject_ids']):
            gate = result['relation_weights'][i].tolist() if result['relation_weights'] is not None else []
            writer.writerow([identity, int(result['true_label'][i]), int(result['predicted_label'][i])]
                            + result['probability'][i].tolist() + gate)
