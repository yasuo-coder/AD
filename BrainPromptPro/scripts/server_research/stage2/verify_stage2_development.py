#!/usr/bin/env python3
"""Independent saved-artifact audit. No project imports, loader, model, or inference.

Only complete result.json directories are audited. Missing results remain pending.
A complete JSON marker is written last by candidate_runner.py, so mutable partial
histories/checkpoints are intentionally excluded. Split metadata/hash inspection
is allowed; no held-out test probabilities, checkpoints, or loaders are evaluated.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path('/home/panyao/AD/BrainPromptPro')
ROOTRUN = ROOT / 'outputs/research/final_optimization_20261007'
STAGE = ROOTRUN / 'stage2_20261008'
CLASSES = ['CN', 'AD', 'SMC', 'EMCI', 'LMCI']
METRICS = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'roc_auc_macro']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def close(actual, expected, message, atol=1e-10, rtol=0):
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f'{message}: actual={actual!r}, expected={expected!r}')


def identity(row):
    return row['subject'].strip() + '/' + row['session'].strip()


def selection_key(m):
    composite = .2*m['accuracy'] + .25*m['f1_macro'] + .2*m['precision_macro'] + .2*m['recall_macro'] + .15*m['roc_auc_macro']
    return m['f1_macro'], composite, -m['loss']


def auc_binary(y, score):
    # Mann-Whitney rank AUC with average ranks for all exact score ties.
    order = np.argsort(score, kind='stable')
    ordered = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positive = int(y.sum()); negative = len(y) - positive
    require(positive > 0 and negative > 0, 'AUC requires both binary classes')
    return float((ranks[y].sum() - positive*(positive + 1)/2) / (positive*negative))


def metrics_from_probabilities(y, p):
    pred = p.argmax(1)
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    true_n = cm.sum(1); pred_n = cm.sum(0); tp = cm.diagonal()
    precision = np.divide(tp, pred_n, out=np.zeros(5), where=pred_n != 0)
    recall = np.divide(tp, true_n, out=np.zeros(5), where=true_n != 0)
    f1 = np.divide(2*tp, true_n+pred_n, out=np.zeros(5), where=(true_n+pred_n) != 0)
    aucs = [auc_binary(y == k, p[:, k]) for k in range(5)]
    m = dict(accuracy=float(tp.sum()/len(y)), precision_macro=float(precision.mean()),
             recall_macro=float(recall.mean()), f1_macro=float(f1.mean()),
             roc_auc_macro=float(np.mean(aucs)), confusion_matrix=cm.tolist(),
             true_distribution=true_n.tolist(), prediction_distribution=pred_n.tolist(),
             per_class_auc=aucs)
    return m, precision, recall, f1


def context():
    cfg = read(ROOTRUN/'config.json'); protocol = read(ROOTRUN/'protocol.json')
    declaration = read(STAGE/'protocol.json'); registry = read(STAGE/'registry.json')
    require(sha(ROOTRUN/'config.json') == declaration['source_configuration_sha256'], 'Source config hash')
    require(sha(ROOTRUN/'protocol.json') == declaration['source_protocol_sha256'], 'Source protocol hash')
    require(declaration['seeds'] == [42, 3407, 2026], 'Declared seeds')
    splitdir = ROOTRUN/'splits/ADNI_Combined'; split = read(splitdir/'indices.json')
    require(sha(splitdir/'indices.json') == protocol['split_sha256'], 'Split SHA256')
    require(set(split) == {'train', 'val', 'test'}, 'Exact split roles')
    for role, expected in protocol['split_csv_sha256'].items():
        require(sha(splitdir/(role+'_samples.csv')) == expected, f'{role} manifest hash')
    bundlepath = cfg['data']['bundle']
    bundlepath = ROOT/bundlepath[5:] if bundlepath.startswith('@pro/') else Path(bundlepath)
    require(sha(bundlepath) == cfg['data']['bundle_sha256'], 'Bundle SHA256')
    bundle = torch.load(bundlepath, map_location='cpu', weights_only=True)
    require(cfg['data']['class_names'] == bundle['class_names'] == protocol['class_names'] == CLASSES, 'Five-class order')
    require(cfg['data']['label_policy'] == bundle['label_policy'] == bundle['provenance']['label_policy'] == 'mci_to_lmci_v1', 'Label policy')
    rec = bundle['records']; ids = [identity(r) for r in rec]
    require(len(ids) == len(set(ids)) and ids == bundle['subject_ids'], 'Bundle IDs')
    flat = sum([split[r] for r in ('train', 'val', 'test')], [])
    require(len(flat) == len(set(flat)) == len(rec) and set(flat) == set(range(len(rec))), 'Split coverage and disjointness')
    labels = np.array([int(r['label']) for r in rec])
    for r in rec:
        require(r['diagnosis'] == CLASSES[int(r['label'])], 'Record diagnosis/class alignment')
        require(r['label_policy'] == 'mci_to_lmci_v1', 'Record label policy')
        if r.get('original_diagnosis') == 'MCI':
            require(r['diagnosis'] == 'LMCI', 'MCI to LMCI mapping')
    require({r['source_batch'] for r in rec} == set(cfg['data']['required_batches']), 'Requested source batches')
    # No test sample manifest is parsed: its frozen hash and index disjointness suffice.
    for role in ('train', 'val'):
        manifest = rows(splitdir/(role+'_samples.csv'))
        require([int(r['scan_index']) for r in manifest] == split[role], f'{role} manifest index order')
        require([r['Sample_ID'] for r in manifest] == [ids[i] for i in split[role]], f'{role} manifest ID order')
        require([int(r['class_id']) for r in manifest] == labels[split[role]].tolist(), f'{role} manifest labels')
        require(set(labels[split[role]]) == set(range(5)), f'{role} all five classes')
    tr = [rec[i] for i in split['train']]
    ages = np.array([float(r['age']) for r in tr], dtype=np.float64)
    require(np.isfinite(ages).all(), 'Finite training ages')
    column = cfg['data']['site_column']; missing = cfg['data']['missing_site']
    sites = [str(r.get(column, '')).strip() for r in tr]
    unknown = {'', 'nan', 'none', 'unknown'}
    require(missing != 'error' or all(s.lower() not in unknown for s in sites), 'Training sites present')
    mapping = {s: i+1 for i, s in enumerate(sorted(set(s for s in sites if s.lower() not in unknown)))}
    encoder = dict(site_column=column, missing_site=missing, age_mean=float(ages.mean()),
                   age_std=max(float(ages.std()), 1e-8), site_mapping=mapping, fit_ids=[ids[i] for i in split['train']])
    counts = np.bincount(labels[split['train']], minlength=5)
    weights = counts.astype(np.float64)**-.5; weights /= weights.mean()
    for key in ('roi_prompt', 'roi_text', 'disease_prompt', 'disease_text'):
        require(sha(cfg['data'][key]) == bundle['provenance']['sha256'][key], f'{key} provenance')
    return dict(cfg=cfg, protocol=protocol, declaration=declaration, registry=registry, split=split,
                bundle=bundle, records=rec, ids=ids, labels=labels, encoder=encoder, weights=weights,
                train_counts=counts.tolist())


def replay(history, cfg, result, ck, last, smoke=False):
    params = cfg['params']; lr = float(params['init_lr']); loss_best = math.inf; bad = 0
    best = None; best_epoch = -1; checkpoint_scheduler = None
    for i, entry in enumerate(history):
        require(entry['epoch'] == i+1, 'History consecutive epochs')
        require(not any('test' in k.lower() for k in entry), 'Per-epoch test key forbidden')
        val = entry['validation']
        require(all(np.isfinite(val[k]) for k in METRICS+['loss']), 'Finite history selection metrics')
        key = selection_key(val)
        if best is None or key > best:
            best, best_epoch = key, i
        # Independent ReduceLROnPlateau min/relative-threshold/cooldown=0 replay.
        if val['loss'] < loss_best*(1-1e-4):
            loss_best, bad = val['loss'], 0
        else:
            bad += 1
        if bad > params['lr_schedule_patience']:
            candidate = max(lr*params['lr_reduce_factor'], 0.0)
            if lr-candidate > 1e-8:
                lr = candidate
            bad = 0
        close(entry['lr'], lr, 'History learning rate', atol=0)
        require(entry['best_epoch'] == best_epoch+1, 'History best epoch tie-break replay')
        stop = None
        if lr < params['min_lr']:
            stop = 'lr_floor'
        elif i+1 >= params['min_epochs'] and i-best_epoch >= params['early_stopping_patience']:
            stop = 'validation_patience'
        elif i+1 == (result['epochs_completed'] if smoke else params['epochs']):
            stop = 'smoke_cap' if smoke else 'epoch_cap'
        require(entry['stop_reason'] == stop, f'Epoch {i+1} prescribed stop')
        require((stop is not None) == (i == len(history)-1), 'First prescribed stop must be final epoch')
        state = dict(best=loss_best, num_bad_epochs=bad, last_epoch=i+1, cooldown_counter=0, _last_lr=[lr])
        if i == ck['epoch']:
            checkpoint_scheduler = state
        if i == len(history)-1:
            last_scheduler = state
    require(result['best_epoch'] == ck['epoch']+1 == best_epoch+1, 'Result/checkpoint best epoch')
    require(len(history) == result['epochs_completed'] == last['epoch']+1, 'Completed epoch consistency')
    require(history[-1]['stop_reason'] == result['stop_reason'], 'Result stop reason')
    require(history[best_epoch]['validation'] == ck['validation'], 'Best history validation checkpoint')
    require(history[-1]['validation'] == last['validation'], 'Last history validation checkpoint')
    for checkpoint, expected in ((ck, checkpoint_scheduler), (last, last_scheduler)):
        s = checkpoint['scheduler']
        for k, value in expected.items():
            require(s[k] == value, f'Scheduler {k} replay')
        require(s['mode'] == 'min' and s['threshold_mode'] == 'rel' and s['threshold'] == 1e-4, 'Scheduler policy')
        require(s['factor'] == params['lr_reduce_factor'] and s['patience'] == params['lr_schedule_patience'], 'Scheduler configuration')
        require(s['cooldown'] == 0 and s['eps'] == 1e-8 and all(x == 0 for x in s['min_lrs']), 'Scheduler defaults')
        require(all(g['lr'] == expected['_last_lr'][0] for g in checkpoint['optimizer']['param_groups']), 'Checkpoint optimizer LR')
    return dict(best_epoch=best_epoch+1, epochs=len(history), stop_reason=result['stop_reason'],
                best_selection_key=list(best), first_prescribed_stop_replayed=True,
                independent_scheduler_replayed=True)


def audit_experiment(path, c, current_hashes, smoke=False):
    out = path.parent; result = read(path); cfg = read(out/'config.json'); spec = read(out/'spec.json')
    model = result['model']; seed = result['seed']; profile = result['supporting_profile']
    require((out.parent.parent.name, out.parent.name, out.name) == (model, profile, f'seed_{seed}'), 'Experiment path identity')
    require(profile == c['declaration']['supporting_profile'] and seed in c['declaration']['seeds'], 'Registered profile/seed')
    require(spec == c['registry'][model], 'Registry specification')
    if model != 'BP_control':
        overlap = spec['overlap_check']
        require(overlap['Decision'] == 'ALLOW' and overlap['Similarity'] in ('Low', 'Medium') and overlap['Does it reproduce a BrainPromptPlus novelty?'] == 'NO', 'Novelty decision')
        require(sha(overlap['evidence_path']) == overlap['evidence_sha256'], 'Novelty evidence SHA256')
    expected_cfg = copy.deepcopy(c['cfg']); expected_cfg['params']['seed'] = seed
    for module, flag in [('M1', 'use_adaptive_brain_graph'), ('M2', 'use_disease_aware_readout'), ('M3', 'use_multirelation_population_graph')]:
        expected_cfg['net_params'][flag] = module in spec['modules']
    require(cfg == expected_cfg, 'Frozen experiment configuration')
    require(result['test'] == 'NOT_EVALUATED', 'Test must remain not evaluated')
    require(result['selection'] == 'validation_macro_f1', 'Validation selection marker')
    require(sha(out/'best_validation.pt') == result['checkpoint_sha256'], 'Checkpoint SHA256')
    ck = torch.load(out/'best_validation.pt', map_location='cpu', weights_only=True)
    last = torch.load(out/'last.pt', map_location='cpu', weights_only=True)
    source = read(out/'source_sha256.json')
    require(source == result['source_sha256'], 'Result source manifest')
    for checkpoint in (ck, last):
        require(checkpoint['model_name'] == model and checkpoint['seed'] == seed and checkpoint['supporting_profile'] == profile, 'Checkpoint identity')
        require(checkpoint['config'] == cfg and checkpoint['spec'] == spec, 'Checkpoint config/spec')
        require(checkpoint['split_sha256'] == result['split_sha256'] == c['protocol']['split_sha256'], 'Checkpoint split')
        require(checkpoint['bundle_sha256'] == c['cfg']['data']['bundle_sha256'], 'Checkpoint bundle')
        require(checkpoint['training_indices_used'] == c['split']['train'], 'Train-only fitting indices')
        require(checkpoint['metadata_encoder'] == c['encoder'], 'Independent training metadata fit')
        require(checkpoint['source_sha256'] == source, 'Checkpoint source manifest')
        close(checkpoint['class_weights'].numpy(), c['weights'], 'Independent train-only class weights', atol=3e-7, rtol=3e-7)
    for original, digest in source.items():
        original_path = Path(original)
        if original not in current_hashes:
            current_hashes[original] = sha(original_path)
        require(current_hashes[original] == digest, f'Current source SHA256 {original}')
        archived = out/'source'/original_path.relative_to(ROOT.parent)
        require(sha(archived) == digest, f'Archived source SHA256 {archived}')
    require(str(ROOT/'candidate_runner.py') in source and str(ROOT/'nets/research_candidates_v1.py') in source, 'Runner/model included in snapshot')
    history = [json.loads(s) for s in (out/'history.jsonl').read_text().splitlines() if s.strip()]
    require(history, 'Nonempty completed history')
    history_audit = replay(history, cfg, result, ck, last, smoke)
    require(read(out/'status.json') == history[-1], 'Final status equals final history')
    metric_audit = {}
    for role in ('train', 'val'):
        table = rows(out/(role+'_predictions.csv')); indexes = c['split'][role]
        require([r['Sample_ID'] for r in table] == [c['ids'][i] for i in indexes], f'{role} CSV sample IDs/order')
        require([r['Source_Dataset'] for r in table] == [c['records'][i]['source_batch'] for i in indexes], f'{role} CSV provenance')
        y = np.array([int(r['true_label']) for r in table]); p = np.array([[float(r['p_'+k]) for k in CLASSES] for r in table])
        require(np.array_equal(y, c['labels'][indexes]), f'{role} CSV truth alignment')
        require(p.shape == (len(indexes), 5) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), f'{role} valid probabilities')
        close(p.sum(1), 1., f'{role} probability normalization', atol=2e-7)
        require([int(r['predicted_label']) for r in table] == p.argmax(1).tolist(), f'{role} argmax predictions')
        close([float(r['confidence']) for r in table], p.max(1), f'{role} confidence')
        payload = torch.load(out/(role+'_payload.pt'), map_location='cpu', weights_only=True)
        require(payload['sample_ids'] == [r['Sample_ID'] for r in table], f'{role} payload IDs')
        require(payload['source_datasets'] == [r['Source_Dataset'] for r in table], f'{role} payload sources')
        require(np.array_equal(payload['truth'].numpy(), y) and np.array_equal(payload['probability'].numpy(), p), f'{role} payload/CSV exact agreement')
        computed, precision, recall, f1 = metrics_from_probabilities(y, p)
        saved = result[role]
        for k in METRICS:
            close(saved[k], computed[k], f'{role} independent {k}')
        for k in ('confusion_matrix', 'true_distribution', 'prediction_distribution'):
            require(saved[k] == computed[k], f'{role} {k}')
        require(saved['class_order'] == list(range(5)) and saved['samples'] == len(y) and saved['zero_division'] == 0, f'{role} metric policy')
        for k, perclass in enumerate(saved['per_class']):
            close([perclass['precision'], perclass['recall'], perclass['f1']], [precision[k], recall[k], f1[k]], f'{role} per-class metrics')
        entropy = -(p*np.log(np.clip(p, 1e-12, None))).sum(1)
        close([float(r['entropy']) for r in table], entropy, f'{role} entropy', atol=5e-7)
        close(saved['entropy_mean'], entropy.mean(), f'{role} entropy mean', atol=5e-7)
        close(saved['confidence_mean'], p.max(1).mean(), f'{role} confidence mean', atol=5e-7)
        top = np.sort(p, axis=1)[:, -2:]
        close(saved['margin_mean'], (top[:, 1]-top[:, 0]).mean(), f'{role} margin mean', atol=5e-7)
        require(saved['classification_collapse'] == (max(computed['prediction_distribution']) == len(y)), f'{role} collapse flag')
        if role == 'val':
            for k in METRICS+['loss']:
                close(saved[k], ck['validation'][k], f'Exported validation agrees with selected checkpoint: {k}', atol=1e-6 if k == 'loss' else 1e-10)
        metric_audit[role] = {k: computed[k] for k in METRICS+['per_class_auc']}
    reference_audit = 'not_applicable'
    if 'M3' in spec['modules']:
        indexes = c['split']['train']; tr = [c['records'][i] for i in indexes]
        upper = torch.triu_indices(c['cfg']['data']['node_num'], c['cfg']['data']['node_num'], offset=1)
        expected_fc = c['bundle']['fc'][indexes][:, upper[0], upper[1]]
        enc = c['encoder']
        expected_meta = dict(age=torch.tensor([(float(r['age'])-enc['age_mean'])/enc['age_std'] for r in tr], dtype=torch.float32),
                             sex=torch.tensor([{'F': 1, 'M': 2}[r['sex'].strip().upper()] for r in tr]),
                             site=torch.tensor([enc['site_mapping'].get(str(r.get(enc['site_column'], '')).strip(), 0) for r in tr]))
        for checkpoint in (ck, last):
            state = checkpoint['model']
            require(state['population._extra_state']['reference_ids'] == enc['fit_ids'], 'M3 exact train-only reference membership')
            require(torch.equal(state['population.upper'], upper) and torch.equal(state['population.reference_fc'], expected_fc), 'M3 exact train-only FC buffer')
            for name, value in expected_meta.items():
                require(torch.equal(state['population.reference_'+name], value), f'M3 train-only {name} reference buffer')
        reference_audit = dict(passed=True, reference_count=len(indexes), both_checkpoints=True)
    return dict(experiment=str(out), model=model, seed=seed, profile=profile, passed=True,
                checkpoint_sha256=result['checkpoint_sha256'], metrics=metric_audit,
                selection_and_stop=history_audit, train_class_counts=c['train_counts'],
                train_metadata_fit_verified=True, class_weights_verified=True,
                class_weight_tolerance=dict(atol=3e-7, rtol=3e-7, reason='Independent float64 formula versus stored float32 arithmetic'),
                source_files_verified=len(source), m3_reference_bank=reference_audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['experiments', 'smoke', 'all'], default='experiments')
    parser.add_argument('--output', type=Path, default=STAGE/'verification/development_verification.json')
    args = parser.parse_args(); torch.set_num_threads(2)
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), scope=args.scope,
                   script_sha256=sha(__file__), test_loader_instantiated=False, test_inference_performed=False,
                   model_forward_performed=False, experiments=[], failures=[], pending=[], seed_completeness={})
    try:
        forbidden = sorted(str(p) for name in ['selection_lock.json', 'test_result.json', 'test_payload.pt', 'test_predictions.csv'] for p in ROOTRUN.rglob(name))
        require(not forbidden, f'Test locks/outputs exist: {forbidden}')
        summary['test_outputs_absent'] = True
        c = context(); summary['frozen_bundle_split_and_manifests_verified'] = True
        scopes = ['experiments', 'smoke'] if args.scope == 'all' else [args.scope]
        current_hashes = {}
        for scope in scopes:
            completeness = {}; summary['seed_completeness'][scope] = completeness
            captured = {p.parent: p for p in (STAGE/scope).glob('*/*/*/result.json')}
            expected_seeds = [42] if scope == 'smoke' else c['declaration']['seeds']
            for model in sorted(c['registry']):
                complete, pending, failed = [], [], []
                for seed in expected_seeds:
                    out = STAGE/scope/model/c['declaration']['supporting_profile']/f'seed_{seed}'
                    if out not in captured:
                        pending.append(seed)
                        summary['pending'].append(dict(experiment=str(out), model=model, seed=seed, status='PENDING', directory_exists=out.exists()))
                        continue
                    try:
                        verified = audit_experiment(captured.pop(out), c, current_hashes, scope == 'smoke')
                        verified['scope'] = scope; summary['experiments'].append(verified); complete.append(seed)
                    except Exception as exc:
                        failed.append(seed)
                        summary['failures'].append(dict(experiment=str(out), error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
                completeness[model] = dict(verified_seeds=complete, pending_or_not_run_seeds=pending, failed_seeds=failed,
                                           complete=not pending and not failed, status='FAILED' if failed else 'PENDING' if pending else 'PASSED')
            for out in captured:
                summary['failures'].append(dict(experiment=str(out), error='Completed result outside declared registry/profile/seeds'))
        # Rehash current files after inspection to detect concurrent source edits.
        for path, digest in current_hashes.items():
            require(sha(path) == digest, f'Source changed during audit: {path}')
    except Exception as exc:
        summary['failures'].append(dict(experiment='GLOBAL', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
    summary['experiments_verified'] = len(summary['experiments'])
    summary['completed_artifacts_passed'] = not summary['failures']
    summary['passed'] = not summary['failures']
    summary['all_requested_seeds_complete'] = not summary['pending'] and not summary['failures']
    summary['status'] = 'FAILED' if summary['failures'] else 'PENDING' if summary['pending'] else 'PASSED'
    summary['limitations'] = ['Saved probability verification does not rerun checkpoint forward inference.',
                             'Probability exports cannot reconstruct disease-head/regularizer loss; selected checkpoint/history/export loss consistency is checked.',
                             'Pending results are excluded, not counted as failures. Completeness is separate from completed-artifact correctness.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix+'.tmp')
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n'); temp.replace(args.output)
    print(json.dumps({k: summary[k] for k in ['status', 'experiments_verified', 'completed_artifacts_passed', 'all_requested_seeds_complete']}))
    print('Report:', args.output)
    for f in summary['failures']:
        print(f['experiment'], f['error'])
    return 1 if summary['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
