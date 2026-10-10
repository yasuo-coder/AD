"""Verify completed validation results and export only a scalar whitelist."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score

from BrainPromptPro.scripts.adni_1to7.audit_dataset import sha, write_json

METRICS = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'balanced_accuracy', 'roc_auc_macro']
PLANNED = ['A0', 'LegacyPro', 'HistoricalM2', 'HistoricalM3v3', 'HistoricalM3v3_no_metadata',
           'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
           'P3_random', 'P3_learned_query', 'P3_parameter_matched', 'P5_fixed', 'P5_no_reliability']


def verify(run):
    result = json.loads((run / 'result.json').read_text())
    if result['smoke_epochs'] or result['stage'] != 'supervised' or result['test'] != 'NOT_EVALUATED':
        raise ValueError('Only completed formal validation runs are allowed')
    if result['seed'] not in [42, 3407, 2026]:
        raise ValueError('Unregistered seed')
    cp = run / 'best_validation.pt'
    if sha(cp) != result['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    cfg = json.loads((run / 'config.json').read_text())
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
    for metric in METRICS:
        if not np.isclose(computed[metric], result['val'][metric], rtol=0, atol=1e-10):
            raise ValueError('Independent metric verification failed: ' + metric)
    # Strict whitelist: no payload IDs, rows, membership, support or confusion.
    return dict(experiment=result['experiment'], seed=result['seed'], metrics=computed,
                best_epoch=result['best_epoch'], stopped_epoch=result['epochs_completed'], stop_reason=result['stop_reason'],
                training_seconds=result['elapsed_seconds'], parameters=result['parameter_count'],
                trainable_parameters=result['trainable_parameters'], checkpoint_sha256=result['checkpoint_sha256'],
                common_training_hash=common_hash, dataset_hash=result['dataset_hash'], split_hash=result['split_hash'],
                per_class_recall={name: row['recall'] for name, row in zip(cfg['data']['class_names'], result['val']['per_class'])},
                collapsed=bool(result['val']['classification_collapse']),
                code=json.loads((run / 'git_version.json').read_text()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    rows = [verify(path.parent) for path in sorted((a.root / 'experiments').glob('*/seed*/result.json'))]
    if len({row['common_training_hash'] for row in rows}) > 1:
        raise ValueError('Different data/split/common training configuration; do not mix summaries')
    grouped = []
    for name in PLANNED:
        group = [row for row in rows if row['experiment'] == name]
        seeds = sorted(row['seed'] for row in group)
        if len(set(seeds)) != len(seeds):
            raise ValueError('Duplicate seed')
        status = 'COMPLETE' if len(group) == 3 else 'PARTIAL' if group else 'NOT_RUN'
        values = {metric: dict(mean=float(np.mean([row['metrics'][metric] for row in group])),
                               sample_sd=float(np.std([row['metrics'][metric] for row in group], ddof=1)) if len(group) > 1 else None)
                  for metric in METRICS} if group else None
        grouped.append(dict(experiment=name, status=status, seeds=seeds, metrics=values))
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(evaluation='validation only; test not evaluated', verification='PASSED',
                  aggregation='mean and sample SD over completed training seeds; partial groups are not final evidence',
                  runs=rows, groups=grouped)
    write_json(a.output / 'validation_summary.public.json', report)
    lines = ['# 七批研究验证结果', '', '固定新协议；验证集选 checkpoint。以下不是最终测试结果。', '',
             '| 方法 | 状态/完成seed数 | Accuracy | Macro-F1 | Macro-P | Macro-R / Balanced Acc | ROC-AUC |',
             '|---|---|---|---|---|---|---|']
    for group in grouped:
        cells = []
        for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'roc_auc_macro']:
            value = group['metrics'][metric] if group['metrics'] else None
            cells.append('未运行' if value is None else f"{100*value['mean']:.2f}" +
                         (f" ± {100*value['sample_sd']:.2f}" if value['sample_sd'] is not None else '（单seed）'))
        lines.append('| ' + ' | '.join([group['experiment'], f"{group['status']}/{len(group['seeds'])}"] + cells) + ' |')
    lines += ['', '单位为%；Balanced Accuracy与固定五类Macro-Recall在本验证集上相同。SD为训练种子间样本标准差，不是患者层面的置信区间。',
              '未完成方法不参与收益结论；部分种子结果不代表完整三seed结果。受控逐样本预测与混淆矩阵保留在服务器。']
    (a.output / 'Validation_Results.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(verified_runs=len(rows), output=str(a.output))))


if __name__ == '__main__':
    main()
