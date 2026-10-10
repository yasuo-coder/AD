#!/usr/bin/env python3
"""Audit ONLY explicitly locked final test artifacts, without models or inference.

Default execution is intended AFTER final selection has been frozen by the root
workflow. Missing locks or test exports remain PENDING. This script never creates
or modifies a selection lock. --self-test uses synthetic arrays only and never
reads experiment, cohort, lock, or real test probability files.

The legacy research_runner and candidate_runner (including the V3 factory) share
the same export schema: test_result contains model, seed, test, by_source,
checkpoint_sha256 and selection_lock_sha256. The supporting profile and source/
split provenance are bound through lock membership and its hashed checkpoint;
they are not falsely required as additional top-level test_result fields.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path('/home/panyao/AD/BrainPromptPro')
ROOTRUN = ROOT/'outputs/research/final_optimization_20261007'
STAGE = ROOTRUN/'stage2_20261008'
CLASSES = ['CN', 'AD', 'SMC', 'EMCI', 'LMCI']
SEEDS = [42, 3407, 2026]
EXPECTED_TEST_N = 100


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def identity(row):
    return row['subject'].strip()+'/'+row['session'].strip()


def close(actual, expected, label, atol=1e-10, rtol=0):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol), f'{label}: numerical disagreement')


def binary_auc(truth, scores):
    positive = scores[truth]; negative = scores[~truth]
    require(len(positive) and len(negative), 'Binary AUC requires positive and negative examples')
    # Exact probability-score Mann-Whitney statistic, half credit to ties.
    differences = positive[:, None]-negative[None, :]
    return float(((differences > 0).sum()+.5*(differences == 0).sum())/differences.size)


def manual_metrics(y, p):
    """Fixed-five-class macro metrics; undefined full-class OvR AUC stays None."""
    y = np.asarray(y); p = np.asarray(p)
    require(len(y) > 0 and p.shape == (len(y), 5), 'Metric array shapes')
    require(np.isin(y, np.arange(5)).all() and np.isfinite(p).all(), 'Metric label/probability validity')
    pred = p.argmax(1); matrix = np.zeros((5, 5), dtype=np.int64)
    np.add.at(matrix, (y, pred), 1)
    support = matrix.sum(1); predicted = matrix.sum(0); tp = matrix.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(5), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(5), where=support != 0)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros(5), where=(support+predicted) != 0)
    accuracy = float(tp.sum()/len(y)); present = np.flatnonzero(support).tolist()
    result = dict(accuracy=accuracy, samples=len(y), present_classes=present,
                  confusion_matrix=matrix.tolist(), confusion_matrix_convention='rows=true, columns=predicted',
                  balanced_accuracy=float(recall[support > 0].mean()), zero_division=0, class_order=list(range(5)),
                  roc_auc_macro=float(np.mean([binary_auc(y == k, p[:, k]) for k in range(5)])) if len(present) == 5 else None)
    for name, values in [('precision', precision), ('recall', recall), ('f1', f1)]:
        result[name+'_macro'] = float(values.mean())
        result[name+'_weighted'] = float((values*support).sum()/len(y))
        result[name+'_micro'] = accuracy
    result['per_class'] = []
    for k in range(5):
        fp = int(predicted[k]-tp[k]); tn = int(len(y)-support[k]-fp)
        result['per_class'].append(dict(class_index=k, support=int(support[k]), predicted_count=int(predicted[k]),
                                        precision=float(precision[k]), recall=float(recall[k]), f1=float(f1[k]),
                                        specificity=float(tn/(tn+fp)) if tn+fp else None))
    return result


def compare_metrics(saved, computed, label):
    for key, value in computed.items():
        require(key in saved, f'{label}: missing metric {key}')
        if key == 'per_class':
            require(len(saved[key]) == 5, f'{label}: exactly five per-class rows')
            for k, row in enumerate(value):
                compare_metrics(saved[key][k], row, f'{label}/class_{k}')
        elif value is None or isinstance(value, (str, list, int)):
            require(saved[key] == value, f'{label}/{key}: exact value mismatch')
        else:
            close(saved[key], value, f'{label}/{key}')


def grouped_seeds(entries):
    groups = defaultdict(list); seen = set()
    require(entries, 'Selection lock has no experiments')
    for entry in entries:
        path = entry['experiment']
        require(path not in seen, 'Duplicate locked experiment')
        seen.add(path)
        require(type(entry['seed']) is int and entry['seed'] in SEEDS, 'Locked seed must be 42/3407/2026')
        groups[(entry['model'], entry['supporting_profile'])].append(entry['seed'])
    for group, seeds in groups.items():
        require(sorted(seeds) == sorted(SEEDS), f'Locked model/profile {group} must include exactly all three seeds')
    return [{'model': k[0], 'profile': k[1], 'seeds': sorted(v)} for k, v in sorted(groups.items())]


def validate_lock(owner, label):
    path = owner/'selection_lock.json'; lock = read(path); digest = sha(path)
    evidence = Path(lock['evidence']).resolve()
    require(evidence.is_file(), f'{label} selection evidence missing')
    require(sha(evidence) == lock['evidence_sha256'], f'{label} selection evidence hash mismatch')
    require(lock.get('created_utc'), f'{label} lock timestamp missing')
    groups = grouped_seeds(lock['experiments'])
    for entry in lock['experiments']:
        out = Path(entry['experiment']).resolve()
        require(str(out) == entry['experiment'], 'Locked experiment path must be canonical')
        require(out.is_relative_to(owner/'experiments'), f'{label} experiment outside declared owner')
        require(out.relative_to(owner/'experiments').parts == (entry['model'], entry['supporting_profile'], f"seed_{entry['seed']}"), 'Locked directory identity')
        require(len(entry['checkpoint_sha256']) == 64, 'Checkpoint digest malformed')
    return dict(owner=owner, label=label, lock=lock, path=path, sha256=digest, groups=groups,
                evidence=str(evidence), evidence_sha256=lock['evidence_sha256'])


def load_context():
    cfg = read(ROOTRUN/'config.json'); protocol = read(ROOTRUN/'protocol.json')
    declaration = read(STAGE/'protocol.json')
    require(sha(ROOTRUN/'config.json') == declaration['source_configuration_sha256'], 'Frozen source configuration')
    require(sha(ROOTRUN/'protocol.json') == declaration['source_protocol_sha256'], 'Frozen source protocol')
    splitdir = ROOTRUN/'splits/ADNI_Combined'; split = read(splitdir/'indices.json')
    require(sha(splitdir/'indices.json') == protocol['split_sha256'], 'Frozen split hash')
    require(set(split) == {'train', 'val', 'test'}, 'Split roles')
    require(len(split['test']) == protocol['sizes']['test'] == EXPECTED_TEST_N, 'Final TEST must contain exactly N=100')
    for role, expected in protocol['split_csv_sha256'].items():
        require(sha(splitdir/(role+'_samples.csv')) == expected, f'Frozen {role} manifest hash')
    name = cfg['data']['bundle']; bundlepath = ROOT/name[5:] if name.startswith('@pro/') else Path(name)
    require(sha(bundlepath) == cfg['data']['bundle_sha256'], 'Frozen bundle hash')
    bundle = torch.load(bundlepath, map_location='cpu', weights_only=True)
    require(bundle['class_names'] == cfg['data']['class_names'] == protocol['class_names'] == CLASSES, 'Fixed five-class order')
    require(bundle['label_policy'] == cfg['data']['label_policy'] == 'mci_to_lmci_v1', 'Fixed label policy')
    records = bundle['records']; ids = [identity(r) for r in records]
    require(ids == bundle['subject_ids'] and len(set(ids)) == len(ids), 'Bundle identity/order')
    flat = sum([split[r] for r in ('train', 'val', 'test')], [])
    require(len(flat) == len(set(flat)) == len(records) and set(flat) == set(range(len(records))), 'Complete disjoint split')
    manifest = read_csv(splitdir/'test_samples.csv'); expected = [records[i] for i in split['test']]
    require(len(manifest) == EXPECTED_TEST_N, 'Test manifest N=100')
    for row, index, rec in zip(manifest, split['test'], expected):
        require(int(row['scan_index']) == index and row['Sample_ID'] == identity(rec), 'Test manifest index/identity')
        require(row['Subject_ID'] == rec['subject'] and row['acquisition_id'] == rec['fmri_image_id'], 'Test acquisition/subject')
        require(int(row['class_id']) == int(rec['label']) and row['diagnosis'] == rec['diagnosis'] == CLASSES[int(rec['label'])], 'Test truth diagnosis alignment')
        require(row['Source_Dataset'] == rec['source_batch'], 'Test source alignment')
        require(float(row['age']) == float(rec['age']) and row['sex'] == rec['sex'] and row['site'] == str(rec['site_id']), 'Test metadata manifest alignment')
    tr = [records[i] for i in split['train']]; ages = np.array([float(r['age']) for r in tr], dtype=np.float64)
    site_column = cfg['data']['site_column']; sites = [str(r.get(site_column, '')).strip() for r in tr]
    mapping = {s: i+1 for i, s in enumerate(sorted(set(s for s in sites if s.lower() not in ('', 'nan', 'none', 'unknown'))))}
    encoder = dict(site_column=site_column, missing_site=cfg['data']['missing_site'], age_mean=float(ages.mean()),
                   age_std=max(float(ages.std()), 1e-8), site_mapping=mapping, fit_ids=[identity(r) for r in tr])
    counts = np.bincount([int(r['label']) for r in tr], minlength=5)
    weights = counts.astype(np.float64)**-.5; weights /= weights.mean()
    return dict(cfg=cfg, protocol=protocol, split=split, expected=expected, encoder=encoder, weights=weights,
                sample_ids=[identity(r) for r in expected], truth=np.array([int(r['label']) for r in expected]),
                sources=[r['source_batch'] for r in expected])


def validate_checkpoint(entry, lock, c, current_hashes):
    out = Path(entry['experiment']); ckpath = out/'best_validation.pt'
    require(sha(ckpath) == entry['checkpoint_sha256'], 'Locked checkpoint hash')
    ck = torch.load(ckpath, map_location='cpu', weights_only=True)
    dev = read(out/'result.json'); cfg = read(out/'config.json'); spec = read(out/'spec.json')
    registry = read(lock['owner']/'registry.json')
    require(dev['test'] == 'NOT_EVALUATED', 'Development result must preserve no-test marker')
    require(ck['model_name'] == dev['model'] == entry['model'] and ck['seed'] == dev['seed'] == entry['seed'], 'Checkpoint/development/lock identity')
    require(ck['supporting_profile'] == dev['supporting_profile'] == entry['supporting_profile'], 'Locked profile')
    require(dev['checkpoint_sha256'] == entry['checkpoint_sha256'], 'Development checkpoint hash')
    require(ck['split_sha256'] == dev['split_sha256'] == c['protocol']['split_sha256'], 'Checkpoint split binding')
    require(ck['bundle_sha256'] == c['cfg']['data']['bundle_sha256'], 'Checkpoint bundle binding')
    require(ck['config'] == cfg and ck['spec'] == spec == registry[entry['model']], 'Checkpoint config/spec/registry')
    require(cfg['data'] == c['cfg']['data'] and cfg['params']['seed'] == entry['seed'], 'Checkpoint frozen data/seed')
    require(ck['training_indices_used'] == c['split']['train'] and ck['metadata_encoder'] == c['encoder'], 'Train-only fitting provenance')
    require(dev['best_epoch'] == ck['epoch']+1, 'Development selected epoch')
    if entry['supporting_profile'] == 'unweighted':
        require(ck['class_weights'] is None, 'Unweighted profile class weights')
    elif entry['supporting_profile'] == 'sqrt_inverse_frequency':
        close(ck['class_weights'].numpy(), c['weights'], 'Training-only class weights', atol=3e-7, rtol=3e-7)
    else:
        raise AssertionError('Unknown supporting profile')
    source = read(out/'source_sha256.json')
    require(source == ck['source_sha256'] == dev['source_sha256'], 'Source manifest binding')
    for path, digest in source.items():
        if path not in current_hashes:
            current_hashes[path] = sha(path)
        require(current_hashes[path] == digest, f'Current trained source {path}')
        archived = out/'source'/Path(path).relative_to(ROOT.parent)
        require(sha(archived) == digest, f'Archived trained source {path}')
    if 'overlap_check' in spec:
        novelty = spec['overlap_check']
        require(novelty['Decision'] == 'ALLOW' and novelty['Does it reproduce a BrainPromptPlus novelty?'] == 'NO', 'Reviewed novelty decision')
        require(sha(novelty['evidence_path']) == novelty['evidence_sha256'], 'Novelty evidence hash')
    return ck, len(source)


def audit_test(entry, lock, c, source_count):
    out = Path(entry['experiment'])
    paths = {name: out/name for name in ('test_result.json', 'test_payload.pt', 'test_predictions.csv')}
    # Caller checks availability of all three before this function reads any export.
    original_hashes = {name: sha(path) for name, path in paths.items()}
    result = read(paths['test_result.json']); table = read_csv(paths['test_predictions.csv'])
    payload = torch.load(paths['test_payload.pt'], map_location='cpu', weights_only=True)
    require(result['model'] == entry['model'] and result['seed'] == entry['seed'], 'Test result identity')
    require(result['checkpoint_sha256'] == entry['checkpoint_sha256'], 'Test result checkpoint binding')
    require(result['selection_lock_sha256'] == lock['sha256'], 'Test result selection-lock binding')
    for name, expected in [('split_sha256', c['protocol']['split_sha256']), ('bundle_sha256', c['cfg']['data']['bundle_sha256']),
                           ('supporting_profile', entry['supporting_profile'])]:
        if name in result:
            require(result[name] == expected, f'Test result {name}')
    require(len(table) == EXPECTED_TEST_N, 'Exported test CSV must contain N=100')
    require([r['Sample_ID'] for r in table] == payload['sample_ids'] == c['sample_ids'], 'Test sample IDs/order')
    require([r['Source_Dataset'] for r in table] == payload['source_datasets'] == c['sources'], 'Test provenance/order')
    y = np.array([int(r['true_label']) for r in table]); p = np.array([[float(r['p_'+name]) for name in CLASSES] for r in table])
    require(np.array_equal(y, c['truth']) and np.array_equal(y, payload['truth'].numpy()), 'Test truth/manifest/payload agreement')
    require(p.shape == (100, 5) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), 'Test finite five-class probabilities')
    close(p.sum(1), 1., 'Probability row sums', atol=2e-7)
    require(np.array_equal(p, payload['probability'].numpy()), 'Test CSV/payload probabilities exact')
    require([int(r['predicted_label']) for r in table] == p.argmax(1).tolist(), 'Test argmax predictions')
    close([float(r['confidence']) for r in table], p.max(1), 'CSV confidence')
    entropy = -(p*np.log(np.clip(p, 1e-12, None))).sum(1)
    close([float(r['entropy']) for r in table], entropy, 'CSV entropy', atol=5e-7)
    metrics = manual_metrics(y, p); compare_metrics(result['test'], metrics, 'overall test')
    require(metrics['present_classes'] == list(range(5)) and metrics['roc_auc_macro'] is not None, 'Full test has all five classes and defined AUC')
    saved = result['test']; prediction_counts = np.bincount(p.argmax(1), minlength=5).tolist()
    require(saved['true_distribution'] == np.bincount(y, minlength=5).tolist() and saved['prediction_distribution'] == prediction_counts, 'Test class distributions')
    require(saved['classification_collapse'] == (max(prediction_counts) == EXPECTED_TEST_N), 'Test collapse marker')
    close(saved['entropy_mean'], entropy.mean(), 'Mean test entropy', atol=5e-7)
    close(saved['confidence_mean'], p.max(1).mean(), 'Mean test confidence', atol=5e-7)
    top = np.sort(p, axis=1)[:, -2:]
    close(saved['margin_mean'], (top[:, 1]-top[:, 0]).mean(), 'Mean test margin', atol=5e-7)
    require(np.isfinite(saved['loss']) and saved['loss'] >= 0, 'Reported test loss finite/nonnegative')
    sources = sorted(set(c['sources'])); require(set(result['by_source']) == set(sources), 'Exact per-source diagnostic keys')
    by_source = {}
    for source in sources:
        mask = np.array([x == source for x in c['sources']])
        computed = manual_metrics(y[mask], p[mask]); compare_metrics(result['by_source'][source], computed, f'source {source}')
        if len(computed['present_classes']) < 5:
            require(result['by_source'][source]['roc_auc_macro'] is None, f'{source}: missing-class AUC must be None')
        by_source[source] = computed
    if 'embeddings' in payload:
        require(payload['embeddings'].ndim == 2 and len(payload['embeddings']) == EXPECTED_TEST_N and torch.isfinite(payload['embeddings']).all(), 'Test embedding export validity')
    for name, path in paths.items():
        require(sha(path) == original_hashes[name], f'{name} unchanged during audit')
    require(sha(out/'best_validation.pt') == entry['checkpoint_sha256'], 'Checkpoint unchanged during test audit')
    return dict(experiment=str(out), owner=lock['label'], runner_format='research_runner' if lock['owner'] == ROOTRUN else 'candidate_runner',
                model=entry['model'], seed=entry['seed'], profile=entry['supporting_profile'], passed=True,
                test_samples=100, checkpoint_sha256=entry['checkpoint_sha256'], selection_lock_sha256=lock['sha256'],
                selection_evidence_sha256=lock['evidence_sha256'], source_files_verified=source_count,
                split_sha256=c['protocol']['split_sha256'], artifact_sha256=original_hashes,
                metrics=metrics, by_source=by_source, loss_recomputed=False)


def self_test():
    """Pure synthetic checks: never calls context, lock or experiment routines."""
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    rng = np.random.default_rng(84); tests = []
    for label, y in [('all_five', np.tile(np.arange(5), 20)), ('missing_classes', np.tile(np.arange(3), 11)), ('one_class', np.zeros(13, dtype=int))]:
        p = rng.integers(1, 5, size=(len(y), 5)).astype(float); p /= p.sum(1, keepdims=True)
        actual = manual_metrics(y, p)
        for avg in ('macro', 'weighted', 'micro'):
            reference = precision_recall_fscore_support(y, p.argmax(1), labels=range(5), average=avg, zero_division=0)[:3]
            close([actual[f'{metric}_{avg}'] for metric in ('precision', 'recall', 'f1')], reference, f'Synthetic {label} {avg}')
        if label == 'all_five':
            close(actual['roc_auc_macro'], roc_auc_score(y, p, average='macro', multi_class='ovr', labels=range(5)), 'Synthetic tied-score AUC')
        else:
            require(actual['roc_auc_macro'] is None, 'Synthetic missing-class AUC None')
        compare_metrics(actual, actual, 'Synthetic valid result')
        corrupted = dict(actual, f1_macro=actual['f1_macro']+.01)
        try:
            compare_metrics(corrupted, actual, 'Synthetic corrupted result')
        except AssertionError:
            pass
        else:
            raise AssertionError('Corrupted macro-F1 accepted')
        tests.append(label)
    entries = [dict(experiment=f'/synthetic/seed_{s}', model='example', supporting_profile='unweighted', seed=s) for s in SEEDS]
    grouped_seeds(entries)
    for invalid in (entries[:-1], entries+[entries[0]]):
        try:
            grouped_seeds(invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError('Incomplete or duplicate selected seeds accepted')
    print(json.dumps(dict(passed=True, synthetic_metric_cases=tests, incomplete_and_duplicate_seeds_rejected=True,
                          real_test_probability_files_read=0, locks_read=0, model_forward_performed=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='Synthetic-only unit checks; reads no experiment/cohort/lock files')
    parser.add_argument('--output', type=Path, default=STAGE/'verification/final_test_verification.json')
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.self_test:
        self_test(); return 0
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
                  scope='Only experiments explicitly named by root and stage selection locks', expected_test_samples=100,
                  expected_seeds=SEEDS, model_forward_performed=False, test_loader_instantiated=False,
                  locks_created=False, locks=[], experiments=[], pending=[], failures=[])
    valid_locks = []
    for label, owner in [('root', ROOTRUN), ('stage2', STAGE)]:
        if not (owner/'selection_lock.json').is_file():
            report['pending'].append(dict(owner=label, status='PENDING', reason='Selection lock absent; no test artifacts read for this owner'))
            continue
        try:
            lock = validate_lock(owner, label); valid_locks.append(lock)
            report['locks'].append({k: str(lock[k]) if isinstance(lock[k], Path) else lock[k] for k in ('label', 'path', 'sha256', 'evidence', 'evidence_sha256', 'groups')})
        except Exception as exc:
            report['failures'].append(dict(owner=label, error=f'{type(exc).__name__}: {exc}'))
    if valid_locks and not report['failures']:
        try:
            c = load_context(); current_hashes = {}
            for lock in valid_locks:
                for entry in lock['lock']['experiments']:
                    try:
                        _, source_count = validate_checkpoint(entry, lock, c, current_hashes)
                        missing = [name for name in ('test_result.json', 'test_payload.pt', 'test_predictions.csv') if not (Path(entry['experiment'])/name).is_file()]
                        if missing:
                            report['pending'].append(dict(experiment=entry['experiment'], owner=lock['label'], status='PENDING', missing_artifacts=missing,
                                                          reason='Await complete final export; no partial test probabilities read', checkpoint_provenance_verified=True))
                            continue
                        report['experiments'].append(audit_test(entry, lock, c, source_count))
                    except Exception as exc:
                        report['failures'].append(dict(experiment=entry['experiment'], error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
            for path, digest in current_hashes.items():
                require(sha(path) == digest, f'Source changed during audit: {path}')
            for lock in valid_locks:
                require(sha(lock['path']) == lock['sha256'] and sha(lock['evidence']) == lock['evidence_sha256'], 'Selection lock/evidence changed during audit')
        except Exception as exc:
            report['failures'].append(dict(experiment='GLOBAL', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
    report['experiments_verified'] = len(report['experiments'])
    report['completed_artifacts_passed'] = not report['failures']
    report['all_locked_test_artifacts_complete'] = bool(valid_locks) and not report['failures'] and not report['pending']
    report['status'] = 'FAILED' if report['failures'] else 'PENDING' if report['pending'] else 'PASSED'
    report['passed'] = report['status'] == 'PASSED'
    report['limitations'] = ['Verifies saved exports; does not rerun checkpoint inference.',
                             'Probability-only exports do not reconstruct disease-head/regularizer loss; loss is checked for finite nonnegative value.',
                             'Per-source macro metrics always average all five fixed classes; OvR macro AUC is None if any class is absent.',
                             'Only explicit lock membership is audited; unselected experiments are never scanned for test probabilities.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix+'.tmp')
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'); tmp.replace(args.output)
    print(json.dumps({k: report[k] for k in ('status', 'experiments_verified', 'completed_artifacts_passed', 'all_locked_test_artifacts_complete')}))
    print('Report:', args.output)
    for failure in report['failures']:
        print(failure)
    return 1 if report['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
