"""Official BrainPrompt scan-stratified ten-fold schedule, last-epoch evaluation.

This explicit protocol deliberately permits longitudinal subject overlap.
It does not replace the separate subject-disjoint / best-checkpoint CLI.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from .data.data import LoadData
from .data.brainnet_pro import FoldDataset, sha256
from .data.metadata import MetadataEncoder, record_id
from .losses import compute_losses
from .metrics import classification_metrics
from .runtime import ROOT, output_path
from .train_TUs_graph_classification_llm import model_inputs

UPSTREAM_COMMIT = 'c85304a023636a2701d465e4cb0acb13b637127e'
FLAGS = ('use_adaptive_brain_graph', 'use_disease_aware_readout', 'use_multirelation_population_graph')


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def make_official_splits(labels, seed=41):
    """Official fallback split algorithm, with its previously unset RNG pinned."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    folds = []
    for remain, test in StratifiedKFold(10, shuffle=True, random_state=rng).split(np.zeros(len(labels)), labels):
        train, val = train_test_split(remain, test_size=0.111, stratify=labels[remain], random_state=rng)
        folds.append(dict(train=train.tolist(), val=val.tolist(), test=test.tolist()))
    return folds


def audit_splits(folds, records):
    if len(folds) != 10:
        raise ValueError('Exactly ten folds required')
    universe = set(range(len(records)))
    test_indices, audit = [], []
    for fold, split in enumerate(folds):
        if set(split) != {'train', 'val', 'test'}:
            raise ValueError('Expected train/val/test')
        sets = {k: set(v) for k, v in split.items()}
        if any(len(sets[k]) != len(split[k]) for k in split):
            raise ValueError('Duplicate scan within split')
        if set.union(*sets.values()) != universe or sum(map(len, sets.values())) != len(universe):
            raise ValueError('Scan partitions overlap or do not cover the cohort')
        subjects = {k: {records[i]['subject'] for i in v} for k, v in split.items()}
        classes = {k: np.bincount([int(records[i]['label']) for i in v], minlength=5).tolist() for k, v in split.items()}
        if any(0 in counts for counts in classes.values()):
            raise ValueError('A partition is missing a class')
        overlap = {a + '_' + b: sorted(subjects[a] & subjects[b])
                   for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
        audit.append(dict(fold=fold, scans={k: len(v) for k, v in split.items()},
                          subjects={k: len(v) for k, v in subjects.items()}, class_counts=classes,
                          subject_overlap=overlap,
                          test_scans_with_train_subject=sum(records[i]['subject'] in subjects['train'] for i in split['test'])))
        test_indices.extend(split['test'])
    if sorted(test_indices) != list(range(len(records))):
        raise ValueError('Each scan must be tested exactly once across ten folds')
    return audit


def stop_reason(epoch, lr, elapsed, params):
    # Match the official strict inequality, not <=, and no best-loss early stop.
    if lr < params['min_lr']:
        return 'learning_rate_below_minimum'
    if elapsed > params['max_time'] * 3600 / 10:
        return 'official_per_fold_time_limit'
    if epoch + 1 >= params['epochs']:
        return 'maximum_epochs'
    return None


def source_hashes():
    files = [ROOT / name for name in ['official_cv.py', 'losses.py', 'metrics.py', 'train_TUs_graph_classification_llm.py']]
    for folder in ['data', 'nets', 'layers']:
        files.extend((ROOT / folder).rglob('*.py'))
    baseline = ROOT.parent / 'BrainPrompt'
    for folder in ['nets', 'layers']:
        files.extend((baseline / folder).rglob('*.py'))
    return {str(p): sha256(p) for p in sorted(set(files))}


def prepare(run_id):
    config = json.loads((ROOT / 'configs/adni123/BrainPromptPro_adni123_download_holdout.json').read_text())
    config['data']['split_mode'] = 'official_scan_stratified_10fold'
    config['params'].update(epochs=1000, n_splits=10, early_stopping_patience=None,
                            device='cuda:0', max_time=60, checkpoint_selection='last_epoch',
                            deterministic_algorithms=False)
    torch.set_num_threads(config['params']['threads'])
    data = LoadData(config['data'])
    folds = make_official_splits(data.labels.numpy(), config['params']['seed'])
    audit = audit_splits(folds, data.records)
    root = output_path('comparisons', run_id)
    root.mkdir(exist_ok=False)
    dump(root / 'config.json', config)
    dump(root / 'splits.json', folds)
    dump(root / 'split_audit.json', audit)
    dump(root / 'provenance.json', data.provenance)
    dump(root / 'source_sha256.json', source_hashes())
    dump(root / 'protocol.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(), upstream_commit=UPSTREAM_COMMIT,
        upstream_url='https://github.com/AngusMonroe/BrainPrompt',
        official_sources=['baseline.sh', 'llm-main.py', 'data/BrainNet.py', 'train_TUs_graph_classification_llm.py'],
        selection='Actual last epoch, no restoration of a best checkpoint',
        split='Scan-level StratifiedKFold(10, shuffle=True) and stratified validation test_size=0.111',
        loss_aggregation='Unweighted mean of batch losses, as in the official trainer',
        scheduler='Adam; ReduceLROnPlateau(val_loss, factor=0.5, patience=25); stop lr < 1e-6',
        test_monitoring='Twice per epoch as in upstream; never used to stop, select, or tune',
        differences_from_unmodified_upstream=[
            'ADNI_1+ADNI_2+ADNI_3 audited subset (598 scans), not the full paper cohort; MCI merged into LMCI.',
            'Fallback split RNG explicitly seeded at 41; upstream fallback leaves this RNG unset before fold creation.',
            'Uses existing audited input adapter and unchanged original BrainPromptG model; bypasses broken upstream spatial keyword.',
            'Honors baseline.sh edge_ratio=0.2. Upstream second LoadData_llm call drops this argument and would use |FC|>=0.3.',
            'Correct macro Precision/Recall/F1 and softmax-probability macro OvR ROC-AUC replace upstream multiclass accuracy placeholders.',
            'BrainPromptPro retains its architecture and configured regularization under the same outer training protocol.',
            'Folds run as independent GPU processes; state_dict last.pt replaces upstream pickled-model serialization.'
        ],
        caution='Same-subject follow-ups may cross train/validation/test. This protocol does not estimate unseen-subject generalization.',
        batch_size=20, seed=41, epochs_cap=1000, per_fold_time_cap_hours=6, ddof=0))
    with (root / 'cohort.csv').open('w', newline='') as stream:
        keys = ['scan_index', 'subject_id', 'source_batch', 'true_label', 'diagnosis', 'fmri_image_id']
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for i, row in enumerate(data.records):
            writer.writerow(dict(scan_index=i, subject_id=record_id(row), source_batch=row['source_batch'],
                                 true_label=row['label'], diagnosis=row['diagnosis'], fmri_image_id=row['fmri_image_id']))
    print(json.dumps(dict(run_root=str(root), samples=len(data), subjects=data.provenance['subjects'],
                         test_sizes=[len(s['test']) for s in folds],
                         overlapping_test_scans=sum(a['test_scans_with_train_subject'] for a in audit))), flush=True)


def epoch_pass(model, loader, device, weights, optimizer=None, capture=False):
    model.train(optimizer is not None)
    totals, batches, count, correct = {}, 0, 0, 0
    probabilities, labels, ids = [], [], []
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            y = batch['label'].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(**model_inputs(batch, device))
            losses = compute_losses(output, y, weights, model.config)
            if not torch.isfinite(losses['total']) or not torch.isfinite(output['logits']).all():
                raise FloatingPointError('Nonfinite loss/logits')
            if optimizer is not None:
                losses['total'].backward()
                optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + value.detach().item()
            correct += int((output['logits'].argmax(-1) == y).sum())
            count += len(y); batches += 1
            if capture:
                probabilities.append(output['logits'].detach().softmax(-1).cpu())
                labels.append(y.cpu()); ids.extend(batch['subject_id'])
    if not batches:
        raise ValueError('Empty loader')
    result = {k: v / batches for k, v in totals.items()}
    result.update(loss=result['total'], accuracy=correct / count, samples=count)
    payload = (dict(probability=torch.cat(probabilities), truth=torch.cat(labels), ids=ids) if capture else None)
    return result, payload


def worker(run_id, name, fold):
    root = output_path('comparisons', run_id)
    if source_hashes() != json.loads((root / 'source_sha256.json').read_text()):
        raise ValueError('Source changed after experiment preparation')
    config = json.loads((root / 'config.json').read_text())
    if name == 'BrainPrompt':
        config['model'] = 'OriginalBrainPromptG'
        config['net_params'].update({f: False for f in FLAGS})
    elif name != 'BrainPromptPro':
        raise ValueError(name)
    params = config['params']
    torch.set_num_threads(params['threads'])
    data = LoadData(config['data'])
    folds = json.loads((root / 'splits.json').read_text())
    audit_splits(folds, data.records)
    split = folds[fold]
    out = root / name / ('fold_' + str(fold))
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'config.json', config)
    start = time.monotonic()
    import dgl
    seed = params['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dgl.seed(seed); dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    device = torch.device(params['device'])
    if not torch.cuda.is_available():
        raise RuntimeError('This full experiment requires CUDA')
    encoder = MetadataEncoder().fit([data.records[i] for i in split['train']], config['data']['site_column'], config['data']['missing_site'])
    if name == 'BrainPrompt':
        from .nets.original_adapter import OriginalBrainPromptAdapter
        model = OriginalBrainPromptAdapter(config, data.roi_prompt, data.disease_prompt).to(device)
    else:
        from .nets.load_net import gnn_model
        model = gnn_model(config['model'], config['net_params'], data.roi_prompt, data.disease_prompt,
                          data.subject_prompt.shape[-1], len(encoder.site_mapping) + 1).to(device)
    loaders = {k: DataLoader(FoldDataset(data, v, encoder), batch_size=params['batch_size'],
                            shuffle=k == 'train', num_workers=0, drop_last=False) for k, v in split.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                    factor=params['lr_reduce_factor'], patience=params['lr_schedule_patience'])
    history = []
    for epoch in range(params['epochs']):
        epoch_start = time.monotonic()
        lr = optimizer.param_groups[0]['lr']
        train, _ = epoch_pass(model, loaders['train'], device, config['loss'], optimizer)
        val, _ = epoch_pass(model, loaders['val'], device, config['loss'])
        test, _ = epoch_pass(model, loaders['test'], device, config['loss'])
        test_again, _ = epoch_pass(model, loaders['test'], device, config['loss'])
        # CUDA graph reductions can vary at floating-point precision. Upstream
        # does not require bitwise equality and this must not stop training.
        repeated_test_delta = abs(test['loss'] - test_again['loss'])
        scheduler.step(val['loss'])
        elapsed = time.monotonic() - start
        reason = stop_reason(epoch, optimizer.param_groups[0]['lr'], elapsed, params)
        entry = dict(epoch=epoch, epoch_number=epoch+1, train=train, validation=val, test=test,
                     lr=lr, next_lr=optimizer.param_groups[0]['lr'], elapsed_seconds=elapsed,
                     epoch_seconds=time.monotonic()-epoch_start, stop_reason=reason,
                     repeated_test_loss_abs_delta=repeated_test_delta)
        history.append(entry)
        with (out / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(entry, allow_nan=False) + '\n')
        checkpoint = dict(model=model.state_dict(), config=config, metadata_encoder=encoder.state_dict(),
                          split=split, training_indices_used=split['train'], fold=fold, last_epoch=epoch,
                          epochs_completed=epoch+1, checkpoint_selection='last_epoch', provenance=data.provenance,
                          optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), stop_reason=reason,
                          split_sha256=sha256(root / 'splits.json'), upstream_commit=UPSTREAM_COMMIT)
        torch.save(checkpoint, out / 'last.tmp.pt')
        (out / 'last.tmp.pt').replace(out / 'last.pt')
        dump(out / 'status.json', dict(model=name, fold=fold, **entry))
        if epoch == 0 or (epoch+1) % 20 == 0 or reason:
            print(json.dumps(dict(model=name, fold=fold, epoch=epoch+1, lr=lr,
                                  train_loss=train['loss'], val_loss=val['loss'], test_accuracy=test['accuracy'],
                                  elapsed_seconds=elapsed, stop_reason=reason)), flush=True)
        if reason:
            break
    final_train, _ = epoch_pass(model, loaders['train'], device, config['loss'])
    # Load the actual on-disk last checkpoint, never a validation-best model.
    last = torch.load(out / 'last.pt', map_location=device, weights_only=True)
    if last['last_epoch'] != history[-1]['epoch']:
        raise RuntimeError('Final checkpoint epoch mismatch')
    model.load_state_dict(last['model'])
    final_test, payload = epoch_pass(model, loaders['test'], device, config['loss'], capture=True)
    expected = [record_id(data.records[i]) for i in split['test']]
    if payload['ids'] != expected:
        raise ValueError('Test IDs/order mismatch')
    y, prob = payload['truth'].numpy(), payload['probability'].numpy()
    metrics = classification_metrics(y, prob, data.num_classes)
    metrics['loss'] = final_test['loss']
    with (out / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scan_index','subject_id','source_batch','true_label','predicted_label'] + ['p_' + c for c in data.class_names])
        for j, i in enumerate(split['test']):
            writer.writerow([i, expected[j], data.records[i]['source_batch'], int(y[j]), int(prob[j].argmax())] + prob[j].tolist())
    result = dict(model=name, fold=fold, class_names=data.class_names, test=metrics, final_train=final_train,
                  epochs_completed=len(history), last_epoch=history[-1]['epoch'], checkpoint_selection='last_epoch',
                  checkpoint=str(out/'last.pt'), checkpoint_sha256=sha256(out/'last.pt'),
                  stop_reason=reason, final_lr=optimizer.param_groups[0]['lr'], elapsed_seconds=time.monotonic()-start,
                  physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu_name=torch.cuda.get_device_name(0),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  metadata_encoder=encoder.state_dict(), split_sha256=sha256(root/'splits.json'),
                  bundle_sha256=config['data']['bundle_sha256'])
    dump(out / 'result.json', result)
    print(json.dumps(dict(event='complete', model=name, fold=fold, test=metrics, epochs=len(history))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','worker'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--model', choices=['BrainPrompt','BrainPromptPro'])
    parser.add_argument('--fold', type=int, choices=range(10))
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ('.','..'):
        parser.error('Invalid run ID')
    if args.command == 'prepare':
        prepare(args.run_id)
    else:
        if args.model is None or args.fold is None:
            parser.error('worker requires --model and --fold')
        worker(args.run_id, args.model, args.fold)


if __name__ == '__main__':
    main()
