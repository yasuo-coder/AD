"""Verify BrainPrompt's ten last-epoch folds and export anonymous metrics.

Run this file directly; the audited training snapshot supplies model/data imports.
No checkpoint, split, stopping rule or model is selected by this report.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

CLASSES = ['CN', 'AD', 'SMC', 'EMCI', 'LMCI']
METRICS = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'balanced_accuracy', 'roc_auc_macro', 'f1_weighted']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def calculate(y, prob):
    if prob.shape != (len(y), 5) or not np.isfinite(prob).all() or (prob < 0).any() or not np.allclose(prob.sum(1), 1, atol=1e-6):
        raise ValueError('Invalid probability payload')
    matrix = confusion_matrix(y, prob.argmax(1), labels=range(5))
    precision, recall, f1, support = precision_recall_fscore_support(y, prob.argmax(1), labels=range(5), zero_division=0)
    if (support == 0).any():
        raise ValueError('Frozen fold unexpectedly lacks a class')
    metrics = dict(accuracy=float(matrix.trace()/matrix.sum()), f1_macro=float(f1.mean()),
                   precision_macro=float(precision.mean()), recall_macro=float(recall.mean()),
                   balanced_accuracy=float(recall.mean()), f1_weighted=float(f1 @ support/support.sum()),
                   roc_auc_macro=float(roc_auc_score(y, prob, multi_class='ovr', average='macro', labels=range(5))))
    return metrics, matrix, {c: dict(precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i])) for i, c in enumerate(CLASSES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', help='Optionally re-forward each saved last checkpoint, e.g. cuda:0')
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    study = a.study_root.resolve()
    version = read(study/'source_version.json')
    code = Path(version['code_root'])
    sys.path.insert(0, str(code))
    from BrainPromptPro.scripts.adni_1to7.cv10_baseline import validate_sources
    from BrainPromptPro.data.metadata import MetadataEncoder, record_id
    from BrainPromptPro.official_cv import audit_splits, stop_reason
    from BrainPromptPro.merged_cv import method_config
    root = (study/'results').resolve()
    expected = code/'BrainPromptPro/outputs/comparisons'/version['run_id']
    if root != expected:
        raise ValueError('Unexpected result root')
    cfg = validate_sources(root)
    state = read(root/'queue_status.json')
    if state['phase'] != 'COMPLETE' or state['failed'] or state['active'] or state['pending'] or sorted(x['fold'] for x in state['completed']) != list(range(10)):
        raise ValueError('Exactly ten successful folds required; no partial final report')
    torch.set_num_threads(2)
    bundle = torch.load(cfg['data']['bundle'], map_location='cpu', weights_only=True)
    splits = read(root/'splits.json')
    if audit_splits(splits, bundle['records']) != read(root/'split_audit.json'):
        raise ValueError('Split audit changed')
    rows, oof, private_checks = [], {}, []
    dataset = None
    if a.device:
        from torch.utils.data import DataLoader
        from BrainPromptPro.data.data import LoadData
        from BrainPromptPro.data.brainnet_pro import FoldDataset
        from BrainPromptPro.nets.original_adapter import OriginalBrainPromptAdapter
        from BrainPromptPro.official_cv import epoch_pass
        dataset = LoadData(cfg['data'])
    expected_config = method_config(cfg, 'BrainPrompt')
    for fold, split in enumerate(splits):
        directory = root/'BrainPrompt'/f'fold_{fold}'
        result = read(directory/'result.json')
        history = [json.loads(line) for line in (directory/'history.jsonl').read_text().splitlines()]
        if result['smoke'] or result['model'] != 'BrainPrompt' or result['fold'] != fold:
            raise ValueError('Unexpected run identity')
        if result['checkpoint_selection'] != 'last_epoch' or result['epochs_completed'] != len(history) or result['last_epoch'] != len(history)-1:
            raise ValueError('Not the actual stopping last epoch')
        ckpath = directory/'last.pt'
        if sha(ckpath) != result['checkpoint_sha256']:
            raise ValueError('Changed last checkpoint')
        ck = torch.load(ckpath, map_location='cpu', weights_only=True)
        if ck['config'] != expected_config or read(directory/'config.json') != expected_config or ck['split'] != split or ck['training_indices_used'] != split['train']:
            raise ValueError('Changed fold configuration/membership')
        if ck['last_epoch'] != result['last_epoch'] or ck['epochs_completed'] != len(history) or ck['stop_reason'] != result['stop_reason']:
            raise ValueError('Checkpoint/result epoch mismatch')
        if ck['split_sha256'] != sha(root/'splits.json') or result['split_sha256'] != ck['split_sha256'] or result['bundle_sha256'] != cfg['data']['bundle_sha256']:
            raise ValueError('Wrong dataset/split provenance')
        enc = MetadataEncoder().fit([bundle['records'][i] for i in split['train']], cfg['data']['site_column'], cfg['data']['missing_site'])
        if enc.state_dict() != ck['metadata_encoder'] or enc.state_dict() != result['metadata_encoder']:
            raise ValueError('Metadata encoding was not TRAIN-only')
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))], lr=cfg['params']['init_lr'], weight_decay=cfg['params']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=cfg['params']['lr_reduce_factor'], patience=cfg['params']['lr_schedule_patience'])
        for epoch, h in enumerate(history):
            if h['epoch'] != epoch or h['epoch_number'] != epoch+1 or h['lr'] != opt.param_groups[0]['lr']:
                raise ValueError('History epoch/LR mismatch')
            scheduler.step(h['validation']['loss'])
            reason = stop_reason(epoch, opt.param_groups[0]['lr'], h['elapsed_seconds'], cfg['params'])
            if h['next_lr'] != opt.param_groups[0]['lr'] or h['stop_reason'] != reason or (reason is not None) != (epoch == len(history)-1):
                raise ValueError('Not the first official stopping epoch')
        if scheduler.state_dict() != ck['scheduler'] or result['final_lr'] != opt.param_groups[0]['lr']:
            raise ValueError('Scheduler/final LR mismatch')
        predictions = list(csv.DictReader((directory/'predictions.csv').open()))
        indices = [int(x['scan_index']) for x in predictions]
        if indices != split['test']:
            raise ValueError('Predictions not from this frozen TEST fold')
        for p, i in zip(predictions, indices):
            record = bundle['records'][i]
            if p['subject_id'] != record_id(record) or p['source_batch'] != record['source_batch'] or int(p['true_label']) != record['label']:
                raise ValueError('Wrong input identity/label')
        y = np.array([int(x['true_label']) for x in predictions])
        prob = np.array([[float(x['p_'+c]) for c in CLASSES] for x in predictions])
        if not np.array_equal(prob.argmax(1), [int(x['predicted_label']) for x in predictions]):
            raise ValueError('Argmax label mismatch')
        metrics, matrix, per_class = calculate(y, prob)
        for key in METRICS:
            if not np.isclose(metrics[key], result['test'][key], rtol=0, atol=1e-10):
                raise ValueError('Independent metric mismatch: '+key)
        if matrix.tolist() != result['test']['confusion_matrix']:
            raise ValueError('Confusion matrix mismatch')
        max_error = None
        if a.device:
            device = torch.device(a.device)
            model = OriginalBrainPromptAdapter(expected_config, dataset.roi_prompt, dataset.disease_prompt).to(device)
            model.load_state_dict(ck['model'])
            loader = DataLoader(FoldDataset(dataset, split['test'], enc), batch_size=cfg['params']['batch_size'], shuffle=False)
            _, payload = epoch_pass(model, loader, device, cfg['loss'], capture=True)
            newprob = payload['probability'].numpy()
            if payload['ids'] != [p['subject_id'] for p in predictions] or not np.array_equal(payload['truth'].numpy(), y) or not np.array_equal(newprob.argmax(1), prob.argmax(1)) or not np.allclose(newprob, prob, rtol=2e-5, atol=2e-5):
                raise ValueError('Last-checkpoint re-forward failed')
            max_error = float(np.max(np.abs(newprob-prob)))
            del model, loader
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        for i, label, p in zip(indices, y, prob):
            if i in oof:
                raise ValueError('Repeated OOF scan')
            oof[i] = (int(label), p)
        rows.append(dict(fold=fold, epochs=result['epochs_completed'], stop_reason=result['stop_reason'], final_lr=result['final_lr'],
                         checkpoint_sha256=result['checkpoint_sha256'], elapsed_seconds=result['elapsed_seconds'], metrics=metrics, per_class=per_class,
                         prediction_fraction={c: float(matrix[:,i].sum()/matrix.sum()) for i, c in enumerate(CLASSES)},
                         row_normalized_confusion=(matrix/matrix.sum(1, keepdims=True)).tolist(),
                         all_cn=bool(np.all(prob.argmax(1) == 0)), collapsed=bool(len(np.unique(prob.argmax(1))) == 1),
                         last_checkpoint_reforward_max_error=max_error))
        private_checks.append(dict(fold=fold, predictions_sha256=sha(directory/'predictions.csv'), last_checkpoint_sha256=sha(ckpath)))
        print(json.dumps(dict(verified_fold=fold, epochs=len(history), reforward_max_error=max_error)), flush=True)
    if sorted(oof) != list(range(1323)):
        raise ValueError('OOF must cover each scan exactly once')
    ys = np.array([oof[i][0] for i in range(1323)])
    probabilities = np.stack([oof[i][1] for i in range(1323)])
    pooled, cm, pc = calculate(ys, probabilities)
    aggregate = {m: dict(mean=float(np.mean([r['metrics'][m] for r in rows])), population_sd=float(np.std([r['metrics'][m] for r in rows], ddof=0))) for m in METRICS}
    validate_sources(root)
    public = dict(status='PASSED', model='BrainPrompt', protocol='ADNI1to7_scan_stratified_cv10_last_epoch', samples=1323, biological_subjects=895,
                  seed=41, folds_completed=10, training_commit=version['local_commit'], data_sha256=cfg['data']['bundle_sha256'],
                  split_sha256=sha(root/'splits.json'), reporter_sha256=sha(__file__), aggregation='Ten fold equal-weight mean and population SD (ddof=0); OOF separate',
                  checkpoint_selection='actual stopping last.pt', all_cn_folds=sum(r['all_cn'] for r in rows),
                  collapse_folds=sum(r['collapsed'] for r in rows), folds=rows, fold_mean=aggregate,
                  oof=dict(metrics=pooled, per_class=pc, row_normalized_confusion=(cm/cm.sum(1, keepdims=True)).tolist(),
                           prediction_fraction={c: float(cm[:,i].sum()/cm.sum()) for i,c in enumerate(CLASSES)}),
                  verification=['source/input/split/previous holdout lock hashes unchanged', 'frozen TEST identity and labels', 'TRAIN-only metadata',
                                'scheduler and first stopping epoch replay', 'actual on-disk last checkpoint', 'independent probability metrics and confusion',
                                'each scan exactly once in OOF']+(['CUDA last-checkpoint re-forward'] if a.device else []),
                  limitations='Scan-level longitudinal overlap and prior development exposure; descriptive baseline replication, not unseen-subject/external validation')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'cv10_summary.public.json').write_text(json.dumps(public, indent=2, allow_nan=False)+'\n')
    (a.output/'verification.private.json').write_text(json.dumps(dict(status='PASSED', checks=private_checks), indent=2)+'\n')
    torch.save(dict(truth=torch.from_numpy(ys), probability=torch.from_numpy(probabilities), confusion_matrix=torch.from_numpy(cm)), a.output/'oof.private.pt')
    lines = ['# BrainPrompt：ADNI_1–ADNI_7 合并数据十折测试结果', '',
             '1323 FC / 895位受试者；CN/AD/SMC/EMCI/LMCI五分类，MCI→LMCI。固定seed41扫描分层十折，每扫描恰好测试一次；停止后重新加载磁盘last.pt评估。', '',
             'Accuracy、F1、Precision、Recall和AUC均为百分数；P/R/F1固定五类macro，AUC为概率macro OvR。', '',
             '| 测试折 | 停止轮数 | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall / BalAcc | ROC-AUC |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+' | '.join([str(r['fold']+1), str(r['epochs'])]+[f"{100*r['metrics'][m]:.2f}" for m in METRICS[:4]+['roc_auc_macro']])+' |')
    cells = [f"{100*aggregate[m]['mean']:.2f} ± {100*aggregate[m]['population_sd']:.2f}" for m in METRICS[:4]+['roc_auc_macro']]
    lines += ['| 十折均值±SD | — | '+' | '.join(cells)+' |',
              '| 合并OOF | — | '+' | '.join(f'{100*pooled[m]:.2f}' for m in METRICS[:4]+['roc_auc_macro'])+' |', '',
              'SD采用历史官方报告口径ddof=0；十折训练集重叠且含纵向关联，不能将其当独立患者置信区间。OOF是合并全部折外预测后重新计算，不能与十折均值混为一个指标。', '',
              f"全预测CN折数：{public['all_cn_folds']}/10；全单类折数：{public['collapse_folds']}/10。", '',
              '| 类别 | OOF Precision | OOF Recall | OOF F1 |', '|---|---:|---:|---:|']
    for c in CLASSES:
        lines.append('| '+c+' | '+' | '.join(f'{100*pc[c][m]:.2f}' for m in ['precision','recall','f1'])+' |')
    lines += ['', '共同配置：原BrainPrompt hidden116/L4/dropout.5，batch20，Adam5e-5/wd.005，两个无权重CE；无梯度裁剪/平滑。最多1000轮或每折6小时，val-loss Plateau(.5,25)，lr严格<1e-6停止；评估实际末轮，不恢复验证最佳轮。',
              '沿用官方每轮两次TEST监测，但测试不参与调度、停止、checkpoint选择或调参。固定原先未设定的fallback split RNG，遵循baseline.sh edge_ratio=.2，修正多分类macro指标及概率AUC；这是本地七批队列复现，不能声称论文完整队列数值复现。',
              '输入/源码/划分/旧holdout锁、每折身份标签、TRAIN编码器、scheduler首次停止、last.pt、概率指标/混淆与OOF覆盖已核验。'+('十折磁盘last.pt的CUDA重载预测复核通过。' if a.device else ''),
              '允许扫描级不同访视纳入，同人可跨划分；队列存在历史开发暴露，不能用于证明独立未见受试者泛化。旧holdout结果与本表分开。', '',
              '[匿名完整数值](cv10_summary.public.json)。原始预测/混淆计数/划分和权重仅留受控目录；不能用测试结果重新选择方案。']
    (a.output/'CV10_Test_Results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASSED', folds=10, mean=aggregate, oof=pooled)))


if __name__ == '__main__':
    main()
