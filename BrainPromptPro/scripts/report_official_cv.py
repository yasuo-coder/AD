"""Validate all twenty last checkpoints and independently recompute CV metrics."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BrainPromptPro.data.brainnet_pro import sha256
from BrainPromptPro.official_cv import source_hashes

CLASSES = ['CN','AD','SMC','EMCI','LMCI']
MODELS = ['BrainPrompt','BrainPromptPro']
METRICS = ['accuracy','f1_macro','precision_macro','recall_macro','roc_auc_macro']
TITLES = ['Accuracy','F1-Score (macro)','Precision (macro)','Recall (macro)','ROC-AUC (macro OvR)']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    with Path(path).open('w',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def recompute(y, p):
    pred = p.argmax(1)
    result = dict(accuracy=float(accuracy_score(y,pred)),
                  roc_auc_macro=float(roc_auc_score(y,p,multi_class='ovr',average='macro',labels=range(5))),
                  confusion_matrix=confusion_matrix(y,pred,labels=range(5)).tolist(),
                  predicted_counts=np.bincount(pred,minlength=5).tolist(),
                  true_counts=np.bincount(y,minlength=5).tolist())
    for average in ['macro','micro','weighted']:
        precision, recall, f1, _ = precision_recall_fscore_support(y,pred,labels=range(5),average=average,zero_division=0)
        result.update({f'precision_{average}':float(precision),f'recall_{average}':float(recall),f'f1_{average}':float(f1)})
    precision,recall,f1,support = precision_recall_fscore_support(y,pred,labels=range(5),zero_division=0)
    result['per_class'] = [dict(class_name=name, support=int(support[i]), predicted=int((pred==i).sum()),
                                precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]),
                                roc_auc=float(roc_auc_score(y==i,p[:,i]))) for i,name in enumerate(CLASSES)]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args = parser.parse_args()
    root = Path('/home/panyao/AD/BrainPromptPro/outputs/comparisons')/args.run_id
    config, splits, audit = read(root/'config.json'), read(root/'splits.json'), read(root/'split_audit.json')
    if source_hashes() != read(root/'source_sha256.json'):
        raise ValueError('Training source changed during the experiment')
    queue = read(root/'queue_status.json')
    if queue['failed'] or queue['pending'] or queue['active'] or len(queue['completed']) != 20:
        raise ValueError('All twenty folds must complete successfully before final reporting')
    cohort = list(csv.DictReader((root/'cohort.csv').open()))
    by_index = {int(row['scan_index']):row for row in cohort}
    if len(by_index) != 598:
        raise ValueError('Unexpected cohort')
    table, summary, pooled, histories, epoch_rows, per_class_rows = [], {}, {}, {}, [], []
    pairs = {}
    for model in MODELS:
        rows, fold_metrics, histories[model] = [], [], []
        for fold in range(10):
            directory = root/model/f'fold_{fold}'
            result = read(directory/'result.json')
            history = [json.loads(line) for line in (directory/'history.jsonl').read_text().splitlines()]
            if result['checkpoint_selection'] != 'last_epoch' or result['last_epoch'] != history[-1]['epoch'] or result['epochs_completed'] != len(history):
                raise ValueError('Result is not the last epoch')
            if result['checkpoint_sha256'] != sha256(directory/'last.pt'):
                raise ValueError('Checkpoint hash mismatch')
            checkpoint = torch.load(directory/'last.pt',map_location='cpu',weights_only=True)
            if checkpoint['last_epoch'] != history[-1]['epoch'] or checkpoint['split'] != splits[fold]:
                raise ValueError('Checkpoint epoch/split mismatch')
            if checkpoint['split_sha256'] != sha256(root/'splits.json') or result['bundle_sha256'] != config['data']['bundle_sha256']:
                raise ValueError('Checkpoint provenance mismatch')
            if checkpoint['stop_reason'] != result['stop_reason']:
                raise ValueError('Checkpoint stop reason mismatch')
            if result['stop_reason'] == 'learning_rate_below_minimum' and not result['final_lr'] < 1e-6:
                raise ValueError('Incorrect LR stop')
            key = (fold,'metadata')
            if model == MODELS[0]:
                pairs[key] = checkpoint['metadata_encoder']
            elif checkpoint['metadata_encoder'] != pairs[key]:
                raise ValueError('Preprocessing differs between paired folds')
            del checkpoint
            predictions = list(csv.DictReader((directory/'predictions.csv').open()))
            indices = [int(row['scan_index']) for row in predictions]
            if indices != splits[fold]['test']:
                raise ValueError('Prediction order differs from frozen test split')
            for row in predictions:
                source = by_index[int(row['scan_index'])]
                for key in ['subject_id','source_batch','true_label']:
                    if row[key] != source[key]:
                        raise ValueError(f'Wrong input identity/label: {key}')
                row.update(model=model,fold=fold)
            y = np.array([int(row['true_label']) for row in predictions])
            p = np.array([[float(row['p_'+c]) for c in CLASSES] for row in predictions])
            if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-6):
                raise ValueError('Invalid probabilities')
            if not np.array_equal(p.argmax(1),[int(row['predicted_label']) for row in predictions]):
                raise ValueError('Predicted label does not match probabilities')
            metrics = recompute(y,p)
            for key in METRICS:
                if not np.isclose(metrics[key], result['test'][key],rtol=0,atol=1e-10):
                    raise ValueError(f'Independent metric mismatch {model}/{fold}/{key}')
            if metrics['confusion_matrix'] != result['test']['confusion_matrix']:
                raise ValueError('Confusion matrix mismatch')
            fold_metrics.append(metrics)
            table.append(dict(model=model,fold=fold,epochs=result['epochs_completed'],
                              test_scans=len(y),**{key:metrics[key] for key in METRICS},
                              **{'predicted_'+c:metrics['predicted_counts'][i] for i,c in enumerate(CLASSES)},
                              final_lr=result['final_lr'],stop_reason=result['stop_reason'],gpu=result['physical_gpu']))
            rows.extend(predictions); histories[model].append(history)
            epoch_rows.append(dict(model=model,fold=fold,epochs=result['epochs_completed'],
                                   elapsed_seconds=result['elapsed_seconds'],stop_reason=result['stop_reason'],
                                   max_repeated_test_loss_delta=max(h['repeated_test_loss_abs_delta'] for h in history)))
        rows.sort(key=lambda r:int(r['scan_index']))
        if [int(row['scan_index']) for row in rows] != list(range(598)):
            raise ValueError('OOF coverage must be exactly once per scan')
        write_csv(root/f'{model}_oof_predictions.csv',rows)
        y = np.array([int(row['true_label']) for row in rows])
        p = np.array([[float(row['p_'+c]) for c in CLASSES] for row in rows])
        pooled[model] = dict(y=y,p=p,metrics=recompute(y,p))
        summary[model] = {key:dict(mean=float(np.mean([m[key] for m in fold_metrics])),
                                  std=float(np.std([m[key] for m in fold_metrics],ddof=0))) for key in METRICS}
        summary[model]['oof'] = pooled[model]['metrics']
        summary[model]['all_cn_folds'] = sum(m['predicted_counts'][0] == sum(m['true_counts']) for m in fold_metrics)
        for row in pooled[model]['metrics']['per_class']:
            per_class_rows.append(dict(model=model,**row))
    if not np.array_equal(pooled[MODELS[0]]['y'],pooled[MODELS[1]]['y']):
        raise ValueError('Model OOF labels differ')
    write_csv(root/'fold_metrics.csv',table)
    write_csv(root/'training_summary.csv',epoch_rows)
    write_csv(root/'per_class_oof_metrics.csv',per_class_rows)
    write_csv(root/'comparison_mean_std.csv',[dict(model=model,metric=key,**summary[model][key]) for model in MODELS for key in METRICS])
    write(root/'comparison_summary.json',dict(models=summary,aggregation='Unweighted fold mean and population SD (ddof=0)',
          samples=598,subjects=398,protocol='official_scan_stratified_10fold_last_epoch',
          class_order=CLASSES,subject_overlap_test_scans=sum(a['test_scans_with_train_subject'] for a in audit)))
    write(root/'verification.json',dict(passed=True,folds_verified=20,oof_scans_per_model=598,
          independent_metrics_recomputed=True,last_checkpoint_epochs_verified=True,
          paired_metadata_encoders_equal=True,source_hashes_unchanged=True,
          report_script_sha256=sha256(Path(__file__)),split_sha256=sha256(root/'splits.json')))

    os.environ['MPLCONFIGDIR'] = str(root/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax, model in zip(axes,MODELS):
        matrix = np.array(pooled[model]['metrics']['confusion_matrix'])
        ax.imshow(matrix,cmap='Blues')
        ax.set(xticks=range(5),yticks=range(5),xticklabels=CLASSES,yticklabels=CLASSES,
               xlabel='Predicted',ylabel='True',title=f'{model}: pooled OOF (598 scans)')
        for (i,j),value in np.ndenumerate(matrix):
            ax.text(j,i,str(value),ha='center',va='center',color='white' if value>matrix.max()/2 else 'black')
    fig.savefig(root/'oof_confusion_matrices.png',dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax, model in zip(axes,MODELS):
        y,p = pooled[model]['y'],pooled[model]['p']
        for i,name in enumerate(CLASSES):
            fpr,tpr,_ = roc_curve(y==i,p[:,i]); auc=roc_auc_score(y==i,p[:,i])
            ax.plot(fpr,tpr,label=f'{name}: {auc:.3f}')
        ax.plot([0,1],[0,1],'k--',lw=1); ax.legend(fontsize=9)
        ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'{model}: OOF OvR ROC',xlim=(0,1),ylim=(0,1))
    fig.savefig(root/'oof_roc_curves.png',dpi=180); plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,model in zip(axes,MODELS):
        for fold,history in enumerate(histories[model]):
            ax.plot([h['epoch_number'] for h in history],[h['validation']['loss'] for h in history],label=f'Fold {fold}',lw=0.8)
        ax.set(title=model,xlabel='Epoch (1-based)',ylabel='Validation loss (mean of batches)')
        ax.legend(ncol=2,fontsize=7)
    fig.savefig(root/'validation_loss_curves.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    x=np.arange(len(METRICS))
    for i,model in enumerate(MODELS):
        ax.bar(x+(i-.5)*.36,[summary[model][key]['mean']*100 for key in METRICS],.36,
               yerr=[summary[model][key]['std']*100 for key in METRICS],label=model,capsize=4)
    ax.set(xticks=x,xticklabels=['Accuracy','Macro F1','Macro precision','Macro recall','Macro OvR AUC'],
           ylabel='Fold mean ± population SD (%)',ylim=(0,100),title='Official schedule: ten folds, last epoch')
    ax.legend();fig.savefig(root/'cv_comparison.png',dpi=180);plt.close(fig)

    text = ['# BrainPrompt 与 BrainPromptPro：官方训练规则十折对比','',
        '已完成两模型全部 20 折。主表为十折测试指标的均值 ± 标准差（ddof=0），每折使用停止时的最后一轮模型。', '',
        '| 模型 | Accuracy | F1-Score（macro） | Precision（macro） | Recall（macro） | ROC-AUC（macro OvR） |',
        '|---|---:|---:|---:|---:|---:|']
    for model in MODELS:
        values=[]
        for key in METRICS:
            m=summary[model][key]
            values.append(f"{m['mean']:.4f} ± {m['std']:.4f}" if key=='roc_auc_macro' else f"{100*m['mean']:.2f}% ± {100*m['std']:.2f}%")
        text.append('| '+model+' | '+' | '.join(values)+' |')
    text += ['', '## 输入与协议', '',
        '- 输入仍为 ADNI_1、ADNI_2、ADNI_3：598 张扫描、398 位受试者；CN 367、AD 19、SMC 38、EMCI 26、LMCI 148。MCI 统一并入 LMCI。',
        '- 标签沿用已核验的各批下载清单 Group 字段；缺少可核验预处理映射的两张原始扫描未纳入。不是论文完整 ADNI 数据集的数值复现。',
        '- 每折 478/479 张训练、60 张验证、60/59 张测试；按扫描分层，十折共 598 张扫描各测试一次。两模型使用相同索引与训练集拟合的 metadata encoder。',
        '- seed=41；Adam；batch=20；lr=5e-5；weight_decay=5e-3；L=4；dropout=0.5；最多 1000 轮。',
        '- 验证损失为各 batch 损失的等权平均；ReduceLROnPlateau，factor=0.5、patience=25；lr < 1e-6 或每折超过 6 小时停止。没有额外的 30 轮早停。',
        '- 每轮按官方入口执行两次测试监测，测试指标不参与模型选择、调度或停止；保存并重新载入 last.pt 完成最终评估。',
        '- 在 GPU 5、6、7 上并行运行独立训练折，每张 GPU 最多两个进程。', '',
        '## 最后一轮模型的实际行为','']
    for model in MODELS:
        oof=summary[model]['oof']; epochs=[r['epochs'] for r in epoch_rows if r['model']==model]
        counts=', '.join(f'{name}={n}' for name,n in zip(CLASSES,oof['predicted_counts']))
        text.append(f"- {model}：各折训练 {min(epochs)}–{max(epochs)} 轮；{summary[model]['all_cn_folds']}/10 折全部预测 CN。合并折外预测：{counts}。")
    text += ['', '各类别指标见 [per_class_oof_metrics.csv](per_class_oof_metrics.csv)。OOF 指标是合并 598 个折外预测重新计算，和主表的各折均值属于不同汇总方式。', '',
        '| 模型 | OOF Accuracy | OOF macro F1 | OOF macro Precision | OOF macro Recall | OOF macro OvR AUC |',
        '|---|---:|---:|---:|---:|---:|']
    for model in MODELS:
        m=summary[model]['oof']
        values=[f'{m[key]:.4f}' if key=='roc_auc_macro' else f'{100*m[key]:.2f}%' for key in METRICS]
        text.append('| '+model+' | '+' | '.join(values)+' |')
    text += ['', '## 与原始发布入口的必要差异及解释范围','',
        '参考官方 commit `'+read(root/'protocol.json')['upstream_commit']+'` 的 `baseline.sh`、`llm-main.py` 和数据划分代码。使用统一可运行入口调用本地原始 BrainPromptG 模型，核心模型未修改；Pro 保留自身结构及正则项。', '',
        '- 官方扫描划分的未固定随机源设为 41，原论文的完整数据索引不适用于本次子集。',
        '- 官方入口的 spatial 参数不兼容由输入适配层绕过；edge_ratio 使用 baseline.sh 声明的 0.2。原发布代码第二次读取数据时会丢失该参数并退回绝对 FC 阈值 0.3，本次未复刻这一参数传递缺陷。',
        '- 官方多分类 Precision/Recall/F1/AUC 使用 Accuracy 占位。本报告改为 sklearn 标准计算：P/R/F1 使用 macro 平均，ROC-AUC 使用 softmax 概率的 macro one-vs-rest；没有预测的类别 precision/F1 设为 0。',
        '- 两次 CUDA 推理允许浮点归约的微小差异，并保存差值；不额外引入停止条件。',
        f"- **纵向受试者重叠：598 张折外测试扫描中，{sum(a['test_scans_with_train_subject'] for a in audit)} 张的受试者在对应训练集中出现。此结果不能解释为对未见受试者的泛化性能。** 逐折重叠 ID 见 split_audit.json。", '',
        '## 核验与产物','',
        '20 个结果均已核对 last.pt 的实际轮数、划分及校验值；CSV 中的概率已独立复算全部指标；598 个测试 ID 每模型恰好覆盖一次；训练前后源码校验值一致。', '',
        '- [逐折指标](fold_metrics.csv) · [均值与标准差](comparison_mean_std.csv) · [训练轮数和耗时](training_summary.csv)',
        '- [BrainPrompt 逐扫描 OOF 概率](BrainPrompt_oof_predictions.csv) · [BrainPromptPro 逐扫描 OOF 概率](BrainPromptPro_oof_predictions.csv)',
        '- [指标图](cv_comparison.png) · [混淆矩阵](oof_confusion_matrices.png) · [ROC 曲线](oof_roc_curves.png) · [验证损失曲线](validation_loss_curves.png)',
        '- 每折完整配置、history.jsonl、result.json、predictions.csv、last.pt：`BrainPrompt/fold_0` 至 `fold_9`，以及 `BrainPromptPro/fold_0` 至 `fold_9`。',
        '- [协议及差异清单](protocol.json) · [输入记录](cohort.csv) · [独立验证](verification.json) · [冻结划分](splits.json)', '',
        '## 复运行','',
        '在 `/home/panyao/AD` 使用 brainprompt 环境，选择新的 RUN_ID（已有目录禁止覆盖）：','',
        '```bash',
        'python -m BrainPromptPro.official_cv prepare --run-id NEW_RUN_ID',
        'python BrainPromptPro/scripts/run_official_cv_queue.py --run-id NEW_RUN_ID --gpus 5 6 7 --jobs-per-gpu 2',
        'python BrainPromptPro/scripts/report_official_cv.py --run-id NEW_RUN_ID',
        '```','']
    (root/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(dict(run_root=str(root), summary=summary, verification='passed'),indent=2),flush=True)


if __name__ == '__main__':
    main()
