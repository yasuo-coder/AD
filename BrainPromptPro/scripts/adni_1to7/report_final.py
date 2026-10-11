"""Audit locked predictions without model execution; export aggregate report/figures.

Raw labels, identities, split membership, predictions and counts remain private.
This tool never selects a model, creates a TEST loader or changes a checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

CLASSES = ['CN', 'AD', 'SMC', 'EMCI', 'LMCI']
METRICS = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro',
           'balanced_accuracy', 'roc_auc_macro', 'f1_weighted']
ALIASES = {'A0': 'BrainPrompt', 'LegacyPro': '旧版 BrainPromptPro',
           'A1': 'P2（验证选定候选）', 'A2': 'P3', 'A3': 'P5',
           'A4': 'P2+P3', 'A5': 'P2+P5', 'A6': 'P3+P5',
           'A7': '新版完整 P2+P3+P5'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def independent_metrics(truth, probability):
    """Calculate counts/P/R/F1 directly, independently of training metrics code."""
    y, prob = np.asarray(truth), np.asarray(probability)
    if y.ndim != 1 or not len(y) or prob.shape != (len(y), 5):
        raise ValueError('Invalid shape')
    if not np.isin(y, np.arange(5)).all() or not np.isfinite(prob).all():
        raise ValueError('Invalid class/probability')
    if (prob < 0).any() or (prob > 1).any() or not np.allclose(prob.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid probability simplex')
    pred = prob.argmax(1)
    matrix = np.zeros((5, 5), dtype=np.int64)
    np.add.at(matrix, (y, pred), 1)
    support, predicted, tp = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    p = np.divide(tp, predicted, out=np.zeros(5), where=predicted > 0)
    r = np.divide(tp, support, out=np.zeros(5), where=support > 0)
    f = np.divide(2*p*r, p+r, out=np.zeros(5), where=p+r > 0)
    # Fixed study has all classes; refuse an unreported protocol change.
    if (support == 0).any():
        raise ValueError('Missing TEST class: report unavailable AUC separately')
    tn = matrix.sum()-support-predicted+tp
    specificity = tn/(matrix.sum()-support)
    metrics = dict(accuracy=float(tp.sum()/len(y)), f1_macro=float(f.mean()),
                   precision_macro=float(p.mean()), recall_macro=float(r.mean()),
                   balanced_accuracy=float(r.mean()), f1_weighted=float(f @ support/len(y)),
                   roc_auc_macro=float(roc_auc_score(y, prob, labels=np.arange(5), multi_class='ovr', average='macro')))
    return metrics, matrix, dict(precision=p, recall=r, f1=f, specificity=specificity)


def audit(root, summary):
    lock_path = root/'selection_lock.json'
    lock = json.loads(lock_path.read_text())
    if summary['verification'] != 'PASSED' or summary['selection_lock_sha256'] != sha(lock_path):
        raise ValueError('Unverified or changed selection lock')
    expected = {(x['protocol'], x['experiment'], x['seed']): x for x in lock['evaluations']}
    actual = {(x['protocol'], x['experiment'], x['seed']): x for x in summary['runs']}
    if len(expected) != 63 or len(actual) != len(summary['runs']) or set(actual) != set(expected):
        raise ValueError('Incomplete fixed 63-evaluation scope')
    datasets, protocol_truth, matrices, audit_rows = {}, {}, {}, []
    for identity, entry in expected.items():
        protocol, exp, seed = identity
        training = Path(entry['run'])
        if sha(training/'result.json') != entry['training_result_sha256']:
            raise ValueError('Changed training record')
        cfg = json.loads((training/'config.json').read_text())
        bundle_path = cfg['data']['bundle']
        if bundle_path not in datasets:
            if sha(bundle_path) != entry['dataset_hash']:
                raise ValueError('Changed dataset')
            datasets[bundle_path] = torch.load(bundle_path, map_location='cpu', weights_only=True)
        bundle = datasets[bundle_path]
        split_path = Path(cfg['study'][protocol+'_split'])
        if sha(split_path) != entry['split_hash']:
            raise ValueError('Changed split')
        indices = json.loads(split_path.read_text())['test']
        ids = [bundle['subject_ids'][i] for i in indices]
        labels = np.array([bundle['records'][i]['label'] for i in indices])
        out = root/'final_test'/protocol/exp/f'seed{seed}'
        record = json.loads((out/'result.private.json').read_text())
        payload_path = out/'predictions.private.pt'
        payload = torch.load(payload_path, map_location='cpu', weights_only=True)
        y, prob = payload['truth'].numpy(), payload['probability'].numpy()
        if payload['sample_ids'] != ids or not np.array_equal(y, labels):
            raise ValueError('Prediction identity/labels do not match frozen TEST')
        if protocol in protocol_truth and protocol_truth[protocol] != (ids, labels.tolist()):
            raise ValueError('Methods evaluated different TEST samples')
        protocol_truth[protocol] = (ids, labels.tolist())
        metrics, matrix, per_class = independent_metrics(y, prob)
        for key in METRICS:
            if not all(np.isclose(metrics[key], x[key], atol=1e-10, rtol=0)
                       for x in (record['metrics'], actual[identity]['metrics'])):
                raise ValueError('Metric mismatch: '+key)
        if matrix.tolist() != record['metrics']['confusion_matrix']:
            raise ValueError('Confusion mismatch')
        for i, name in enumerate(CLASSES):
            for metric, values in per_class.items():
                if not np.isclose(values[i], actual[identity]['per_class'][name][metric], atol=1e-10, rtol=0):
                    raise ValueError('Class metric mismatch')
        collapsed = np.count_nonzero(matrix.sum(0)) == 1
        if collapsed != actual[identity]['classification_collapse']:
            raise ValueError('Collapse mismatch')
        matrices[identity] = matrix / matrix.sum(1, keepdims=True)
        audit_rows.append(dict(protocol=protocol, experiment=exp, seed=seed,
                               prediction_sha256=sha(payload_path), status='PASSED'))
    for g in summary['groups']:
        rows = [x for x in summary['runs'] if (x['protocol'], x['experiment']) == (g['protocol'], g['experiment'])]
        if sorted(x['seed'] for x in rows) != [42, 2026, 3407]:
            raise ValueError('Seed group changed')
        for metric in METRICS:
            values = [x['metrics'][metric] for x in rows]
            for k, v in [('mean', np.mean(values)), ('sample_sd', np.std(values, ddof=1))]:
                if not np.isclose(v, g['metrics'][metric][k], atol=1e-10, rtol=0):
                    raise ValueError('Aggregation mismatch')
    return matrices, dict(status='PASSED', verified_runs=len(audit_rows),
                         selection_lock_sha256=sha(lock_path),
                         checks=['locked identities/labels', 'same protocol TEST across methods',
                                 'direct count-based P/R/F1/accuracy', 'probability macro OvR AUC',
                                 'per-class metrics/confusion/collapse', 'three-seed mean/sample SD'],
                         runs=audit_rows)


def paired(summary, protocol, newer, older):
    rows = {(x['experiment'], x['seed']): x for x in summary['runs'] if x['protocol'] == protocol}
    return {metric: dict(mean=float(np.mean(v := [rows[newer, s]['metrics'][metric]-rows[older, s]['metrics'][metric]
                                                   for s in [42, 3407, 2026]])),
                         sample_sd=float(np.std(v, ddof=1)), values=v) for metric in METRICS}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    summary = json.loads(a.summary.read_text())
    matrices, audit_record = audit(a.root, summary)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'audit.private.json').write_text(json.dumps(audit_record, indent=2)+'\n')
    pairs = {protocol: {f'{newer}-{older}': paired(summary, protocol, newer, older)
                        for newer, older in ([('A1', 'A0'), ('A7', 'A0'), ('A7', 'LegacyPro'), ('A1', 'LegacyPro')]
                                             if protocol == 'sample' else [('A1', 'A0')])}
             for protocol in ['sample', 'subject']}
    normalized = []
    for group in summary['groups']:
        protocol, exp = group['protocol'], group['experiment']
        values = [matrices[protocol, exp, s] for s in [42, 3407, 2026]]
        normalized.append(dict(protocol=protocol, experiment=exp,
                               row_normalized_mean=np.mean(values, axis=0).tolist(),
                               row_normalized_seed_sd=np.std(values, axis=0, ddof=1).tolist()))
    public = dict(verification='PASSED', verified_test_runs=63,
                  summary_sha256=sha(a.summary), selection_lock_sha256=audit_record['selection_lock_sha256'],
                  primary_candidate=summary['primary_candidate_from_validation'], paired_differences=pairs,
                  normalized_confusion=normalized,
                  uncertainty='Paired training-seed differences; not subject confidence intervals or significance tests')
    (a.output/'final_analysis.public.json').write_text(json.dumps(public, indent=2, allow_nan=False)+'\n')
    lines = ['# 冻结测试补充分析', '', '五类顺序 CN/AD/SMC/EMCI/LMCI。以下为固定比较对象的一次测试，不用于重新选型。', '',
             '全部63份预测逐一对齐冻结TEST成员和标签；用计数公式重新计算P/R/F1/Accuracy，用概率计算Macro OvR AUC；核验混淆矩阵、种子均值与样本SD。未重新执行任何模型。', '',
             '## 同seed指标差值', '', '单位为百分点，均值±3个配对训练seed的样本SD；不代表显著性或患者置信区间。', '',
             '| 协议 | 比较 | Accuracy | Macro-F1 | Macro-P | Macro-R / BalAcc | AUC |',
             '|---|---|---:|---:|---:|---:|---:|']
    for protocol, contrasts in pairs.items():
        for name, values in contrasts.items():
            cells = [f"{values[m]['mean']*100:+.2f} ± {values[m]['sample_sd']*100:.2f}"
                     for m in METRICS[:4]+['roc_auc_macro']]
            lines.append('| '+' | '.join([protocol, name]+cells)+' |')
    lines += ['', '## 类别Recall、预测分布与成本', '', 'Recall/预测比例为各seed平均百分数。时间为原训练循环分钟，共享GPU且不含初始P5 bank构建，不作为严格速度benchmark。', '',
              '| 协议 | 方法 | CN Recall | AD Recall | SMC Recall | EMCI Recall | LMCI Recall | CN预测比例 | Weighted-F1 | 单类塌缩seed数 | 总参数 | 训练分钟 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in summary['groups']:
        protocol, exp = group['protocol'], group['experiment']
        rows = [r for r in summary['runs'] if (r['protocol'], r['experiment']) == (protocol, exp)]
        recall = [f"{100*np.mean([r['per_class'][c]['recall'] for r in rows]):.2f}" for c in CLASSES]
        cells = [protocol, exp]+recall+[f"{100*np.mean([r['prediction_fraction']['CN'] for r in rows]):.2f}",
                   f"{100*group['metrics']['f1_weighted']['mean']:.2f}",
                   str(sum(r['classification_collapse'] for r in rows)), str(rows[0]['training_parameters']),
                   f"{np.mean([r['training_seconds'] for r in rows])/60:.2f}"]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', 'P5总参数含冻结参考编码器。每类Precision/F1/Specificity与全部种子指标见test_summary.public.json；行归一化混淆矩阵见final_analysis.public.json。绝对计数及逐例数据仅在服务器。',
              '', '跨协议存在样本角色交叉，历史开发也曾使用部分样本；此处只能解释为各自协议内的holdout。不同协议不可合并。']
    (a.output/'Final_Analysis.md').write_text('\n'.join(lines)+'\n')
    # Standalone, deterministic vector figures from aggregate results only.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    def save(fig, name):
        fig.savefig(a.output/(name+'.svg'), bbox_inches='tight', metadata={'Date': None})
        fig.savefig(a.output/(name+'.png'), bbox_inches='tight', dpi=180)
        svg = a.output/(name+'.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig)
    groups = {(x['protocol'], x['experiment']): x for x in summary['groups']}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, protocol, methods in zip(axes, ['sample', 'subject'], [['A0', 'LegacyPro', 'A1', 'A7'], ['A0', 'A1']]):
        for j, metric in enumerate(['accuracy', 'f1_macro', 'balanced_accuracy', 'roc_auc_macro']):
            xs = np.arange(len(methods))+(j-1.5)*.19
            vals = [groups[protocol, name]['metrics'][metric] for name in methods]
            ax.bar(xs, [100*x['mean'] for x in vals], .18, yerr=[100*x['sample_sd'] for x in vals], capsize=2, label=metric)
        ax.set(xticks=np.arange(len(methods)), xticklabels=methods, ylim=(0, 100), ylabel='%', title=f'{protocol}: locked TEST')
        ax.grid(axis='y', alpha=.2)
    axes[1].legend(fontsize=8)
    fig.text(.5, .01, 'A1 was selected on validation before TEST; A7 is the full P2+P3+P5 model. Error bars: training-seed SD.\nProtocols are reported separately; cross-protocol and historical exposure limit generalization.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, 1)); save(fig, 'test_comparison')
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    selections = [('sample', x) for x in ['A0', 'LegacyPro', 'A1', 'A7']]+[('subject', x) for x in ['A0', 'A1']]
    for ax, (protocol, exp) in zip(axes.flat, selections):
        matrix = np.mean([matrices[protocol, exp, s] for s in [42, 3407, 2026]], axis=0)
        ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
        ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
               title=f'{protocol} / {exp}', xlabel='Predicted', ylabel='True')
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f'{matrix[i,j]*100:.0f}', ha='center', va='center', color='white' if matrix[i,j]>.5 else 'black', fontsize=8)
    fig.suptitle('Row-normalized TEST confusion (%) averaged across three training seeds')
    fig.tight_layout(rect=(0, 0, 1, .95)); save(fig, 'test_confusion')
    print(json.dumps(dict(status='PASSED', independently_verified=63, output=str(a.output))))


if __name__ == '__main__':
    main()
