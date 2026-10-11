"""Verify completed validation results and export only a scalar whitelist."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score

from BrainPromptPro.scripts.adni_1to7.audit_dataset import sha, write_json

METRICS = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'balanced_accuracy', 'roc_auc_macro', 'f1_weighted']
PLANNED = ['A0', 'LegacyPro', 'HistoricalM2', 'HistoricalM3v3', 'HistoricalM3v3_no_metadata',
           'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
           'P2_finetune_no_dropout_pair_best', 'P2_finetune_no_dropout_pair_last40',
           'P2_joint_rec_only', 'P2_joint_cons_only', 'P2_finetune_rec10', 'P2_finetune_cons10',
           'P3_random', 'P3_learned_query', 'P3_parameter_matched', 'P5_fixed', 'P5_no_reliability']


def verify(run, protocol="sample"):
    result = json.loads((run / 'result.json').read_text())
    if result['smoke_epochs'] or result['stage'] != 'supervised' or result['test'] != 'NOT_EVALUATED':
        raise ValueError('Only completed formal validation runs are allowed')
    if result['seed'] not in [42, 3407, 2026]:
        raise ValueError('Unregistered seed')
    cp = run / 'best_validation.pt'
    if sha(cp) != result['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    cfg = json.loads((run / 'config.json').read_text())
    if sha(run/'config.json') != result['config_sha256']:
        raise ValueError('Archived configuration changed')
    if sha(cfg['study'][protocol+'_split']) != result['split_hash'] or cfg['data']['bundle_sha256'] != result['dataset_hash']:
        raise ValueError('This summary requires its explicitly selected frozen protocol')
    common = {k: cfg[k] for k in ['data', 'net_params', 'loss', 'params', 'baseline_root']}
    common['split_hash'] = result['split_hash']
    common_hash = hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()
    for path, expected in result['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Immutable training source changed')
    payload = torch.load(run / 'val_payload.private.pt', map_location='cpu', weights_only=True)
    truth, prob = payload['truth'].numpy(), payload['probability'].numpy()
    if not np.isfinite(prob).all() or not np.allclose(prob.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid probability payload')
    pred = prob.argmax(-1)
    p, r, f, _ = precision_recall_fscore_support(truth, pred, labels=np.arange(5), average='macro', zero_division=0)
    computed = dict(accuracy=accuracy_score(truth, pred), f1_macro=f, precision_macro=p, recall_macro=r,
                    balanced_accuracy=balanced_accuracy_score(truth, pred),
                    roc_auc_macro=roc_auc_score(truth, prob, multi_class='ovr', average='macro', labels=np.arange(5)))
    class_p, class_r, class_f, support = precision_recall_fscore_support(truth, pred, labels=np.arange(5), zero_division=0)
    computed['f1_weighted'] = float(np.average(class_f, weights=support))
    for index, values in enumerate(zip(class_p, class_r, class_f)):
        for name, value in zip(['precision','recall','f1'], values):
            if not np.isclose(value, result['val']['per_class'][index][name], rtol=0, atol=1e-10):
                raise ValueError('Per-class verification failed')
    if bool(result['val']['classification_collapse']) != (len(np.unique(pred)) == 1):
        raise ValueError('Collapse flag mismatch')
    for metric in METRICS:
        if not np.isclose(computed[metric], result['val'][metric], rtol=0, atol=1e-10):
            raise ValueError('Independent metric verification failed: ' + metric)
    # Strict whitelist: no payload IDs, rows, membership, support or confusion.
    return dict(experiment=result['experiment'], seed=result['seed'], metrics=computed,
                best_epoch=result['best_epoch'], stopped_epoch=result['epochs_completed'], stop_reason=result['stop_reason'],
                training_seconds=result['elapsed_seconds'], parameters=result['parameter_count'],
                trainable_parameters=result['trainable_parameters'], checkpoint_sha256=result['checkpoint_sha256'],
                common_training_hash=common_hash, dataset_hash=result['dataset_hash'], split_hash=result['split_hash'],
                per_class_recall=dict(zip(cfg['data']['class_names'], class_r.tolist())),
                per_class_precision=dict(zip(cfg['data']['class_names'], class_p.tolist())),
                per_class_f1=dict(zip(cfg['data']['class_names'], class_f.tolist())),
                prediction_fraction={name: float(np.mean(pred==i)) for i,name in enumerate(cfg['data']['class_names'])},
                mean_class_probability=dict(zip(cfg['data']['class_names'], prob.mean(0).tolist())),
                inference_seconds=result['val']['inference_seconds'],
                inference_scope='validation pass; shared device; data iteration included',
                collapsed=bool(result['val']['classification_collapse']),
                code=json.loads((run / 'git_version.json').read_text()))



def queue_activity(queue_root, protocol):
    """Read training queues only; evaluation controllers have a different schema."""
    activity, waiting = {}, {}
    for queue in sorted(queue_root.glob('*/state.private.json')):
        state = json.loads(queue.read_text())
        if 'queue_pid' not in state:
            if 'pid' in state and 'phase' in state:
                continue  # Observer/finalization state, never a training queue.
            raise ValueError(f'Unrecognized queue state: {queue}')
        for field, target in [('running', activity), ('pending', waiting)]:
            jobs = state.get(field)
            if not isinstance(jobs, list):
                raise ValueError(f'Malformed training queue {field}: {queue}')
            for job in jobs:
                if str(job.get('split', '0')) != ('0' if protocol == 'sample' else '1'):
                    continue
                target.setdefault(job['experiment'], set()).add(job['seed'])
    return activity, waiting


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol', choices=['sample','subject'], default='sample')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    experiment_root = a.root / ('experiments' if a.protocol=='sample' else 'subject_experiments')
    rows = [verify(path.parent,a.protocol) for path in sorted(experiment_root.glob('*/seed*/result.json'))]
    if len({row['common_training_hash'] for row in rows}) > 1:
        raise ValueError('Different data/split/common training configuration; do not mix summaries')
    activity, waiting = queue_activity(a.root / 'queues', a.protocol)
    grouped = []
    planned = PLANNED if a.protocol=='sample' else sorted(set(row['experiment'] for row in rows)|set(activity)|set(waiting))
    for name in planned:
        group = [row for row in rows if row['experiment'] == name]
        seeds = sorted(row['seed'] for row in group)
        if len(set(seeds)) != len(seeds):
            raise ValueError('Duplicate seed')
        running=sorted(activity.get(name,set()));queued=sorted(waiting.get(name,set()))
        incomplete=any(not (p/'result.json').exists() for p in (experiment_root/name).glob('seed*'))
        status = 'COMPLETE' if len(group)==3 else 'PARTIAL_RUNNING' if group and running else 'PARTIAL' if group else 'RUNNING' if running else 'QUEUED' if queued else 'INCOMPLETE_NEEDS_AUDIT' if incomplete else 'NOT_RUN'
        values = {metric: dict(mean=float(np.mean([row['metrics'][metric] for row in group])),
                               sample_sd=float(np.std([row['metrics'][metric] for row in group], ddof=1)) if len(group) > 1 else None)
                  for metric in METRICS} if group else None
        grouped.append(dict(experiment=name, status=status, seeds=seeds, running_seeds=running, queued_seeds=queued, metrics=values))
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=a.protocol,snapshot_utc=datetime.now(timezone.utc).isoformat(), evaluation='validation only; test not evaluated', verification='PASSED',
                  aggregation='mean and sample SD over completed training seeds; partial groups are not final evidence',
                  runs=rows, groups=grouped)
    write_json(a.output / 'validation_summary.public.json', report)
    lines = ['# 七批研究验证结果', '', f'独立{a.protocol}协议；验证集选 checkpoint。以下不是最终测试结果，禁止混合两个协议。', '',
             '| 方法 | 状态/完成seed数 | Accuracy | Macro-F1 | Macro-P | Macro-R / Balanced Acc | ROC-AUC |',
             '|---|---|---|---|---|---|---|']
    for group in grouped:
        running=group['running_seeds'];queued=group['queued_seeds'];incomplete=group['status']=='INCOMPLETE_NEEDS_AUDIT'
        cells = []
        for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'roc_auc_macro']:
            value = group['metrics'][metric] if group['metrics'] else None
            cells.append(('运行中' if running else '待启动' if queued else '待核验' if incomplete else '未运行') if value is None else f"{100*value['mean']:.2f}" +
                         (f" ± {100*value['sample_sd']:.2f}" if value['sample_sd'] is not None else '（单seed）'))
        lines.append('| ' + ' | '.join([group['experiment'], f"{group['status']}/{len(group['seeds'])}"] + cells) + ' |')
    lines += ['', '单位为%；Balanced Accuracy与固定五类Macro-Recall在本验证集上相同。SD为训练种子间样本标准差，不是患者层面的置信区间。',
              '未完成方法不参与收益结论；部分种子结果不代表完整三seed结果。受控逐样本预测与混淆矩阵保留在服务器。']
    (a.output / 'Validation_Results.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(verified_runs=len(rows), output=str(a.output))))


if __name__ == '__main__':
    main()
