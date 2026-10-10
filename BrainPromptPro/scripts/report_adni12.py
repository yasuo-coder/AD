#!/usr/bin/env python3
"""Recompute held-out metrics from saved predictions and write a shareable report."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from BrainPromptPro.data.brainnet_pro import sha256
from BrainPromptPro.data.prepared_cohort import pro_path
from BrainPromptPro.data.metadata import record_id
from BrainPromptPro.metrics import classification_metrics, subject_metrics
from BrainPromptPro.runtime import ROOT, output_path, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_id',required=True)
    args=parser.parse_args()
    directory=output_path('results',args.run_id,'fold_0')
    result=json.loads((directory/'result.json').read_text())
    config=json.loads((directory.parent/'config.json').read_text())
    split=json.loads((directory.parent/'splits.json').read_text())[0]
    rows=list(csv.DictReader((directory/'predictions.csv').open()))
    names=result['class_names']
    truth=np.array([int(r['true_label']) for r in rows])
    probabilities=np.array([[float(r['p_'+c]) for c in names] for r in rows],dtype=np.float32)
    metrics=classification_metrics(truth,probabilities,len(names))
    subjects=subject_metrics([r['subject_id'] for r in rows],truth,probabilities,len(names))
    for scope,current,saved in [('scan',metrics,result['test']),('subject',subjects,result['test_subjects'])]:
        for key in ('accuracy','precision_macro','recall_macro','f1_macro','f1_weighted','roc_auc_macro'):
            assert np.isclose(current[key],saved[key],atol=1e-7), (scope,key,current[key],saved[key])
        assert current['confusion_matrix']==saved['confusion_matrix']
    # The saved run owns its cohort. Never read a mutable current manifest.
    bundle_path = pro_path(config['data']['bundle'])
    if sha256(bundle_path) != config['data']['bundle_sha256']:
        raise ValueError('Saved run cohort hash mismatch')
    bundle = torch.load(bundle_path,map_location='cpu',weights_only=True)
    manifest = bundle['records']
    exclusions = bundle['provenance']['excluded_scans']
    policy = bundle.get('label_policy','historical_unversioned')
    if policy != config['data'].get('label_policy','historical_unversioned'):
        raise ValueError('Saved run label policy mismatch')
    if [r['subject_id'] for r in rows] != [record_id(manifest[i]) for i in split['test']]:
        raise ValueError('Saved predictions do not match the saved test split')
    if truth.tolist() != [int(manifest[i]['label']) for i in split['test']]:
        raise ValueError('Saved predictions use different labels from the cohort')
    split_summary={role:dict(samples=len(indices),subjects=len({manifest[i]['subject'] for i in indices}),
        classes={name:sum(manifest[i]['diagnosis']==name for i in indices) for name in names})
        for role,indices in split.items()}
    majority=Counter(int(manifest[i]['label']) for i in split['train']).most_common(1)[0][0]
    baseline_probs=np.zeros_like(probabilities);baseline_probs[:,majority]=1
    majority_metrics=classification_metrics(truth,baseline_probs,len(names))
    known_sites=set(result['metadata_encoder']['site_mapping'])
    unknown_sites={role:sum(manifest[i]['site_id'] not in known_sites for i in indices) for role,indices in split.items()}
    report=dict(label_policy=policy,metrics_recomputed_from_predictions=True,class_names=names,split=split_summary,
        scan_metrics=metrics,subject_metrics=subjects,by_source=result['test_by_source'],
        training_majority_class=names[majority],majority_predictor=majority_metrics,
        unknown_site_scan_counts=unknown_sites,exclusions=dict(Counter(r['diagnosis'] for r in exclusions)),
        best_epoch_zero_based=result['best_epoch'],epochs_completed=result['epochs_completed'])
    write_json(directory/'metrics_report.json',report)
    with (directory/'per_class_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['class_name']+list(metrics['per_class'][0]))
        writer.writeheader()
        writer.writerows(dict(class_name=names[r['class_index']],**r) for r in metrics['per_class'])
    with (directory/'confusion_matrix.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['true/predicted']+names)
        writer.writerows([name]+row for name,row in zip(names,metrics['confusion_matrix']))
    os.environ['MPLCONFIGDIR']=str(output_path('cache','matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matrix=np.asarray(metrics['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(6,5))
    picture=ax.imshow(matrix,cmap='Blues');fig.colorbar(picture,ax=ax)
    ax.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=names,yticklabels=names,
           xlabel='Predicted class',ylabel='True class',title='ADNI_1 + ADNI_2: held-out scans')
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    fig.tight_layout();fig.savefig(directory/'confusion_matrix.png',dpi=180);plt.close(fig)
    history=result['history'];epochs=[r['epoch']+1 for r in history]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(epochs,[r['train']['total'] for r in history],label='Train')
    axes[0].plot(epochs,[r['validation']['loss'] for r in history],label='Validation')
    axes[0].set(xlabel='Epoch',ylabel='Total loss')
    for metric,label in [('accuracy','Accuracy'),('f1_macro','Macro F1')]:
        axes[1].plot(epochs,[r['validation'][metric] for r in history],label=label)
    axes[1].set(xlabel='Epoch',ylabel='Validation score',ylim=(0,1))
    for ax in axes:
        ax.axvline(result['best_epoch']+1,color='gray',linestyle='--',label='Selected checkpoint');ax.legend()
    fig.tight_layout();fig.savefig(directory/'training_curves.png',dpi=180);plt.close(fig)
    pct=lambda x:f'{100*x:.2f}%'
    predicted=Counter(int(r['predicted_label']) for r in rows)
    conclusion=(f'本次所有测试扫描均被判为 {names[next(iter(predicted))]}；其余 {len(names)-1} 类未被识别。'
                if len(predicted)==1 else '请同时检查 macro 指标和逐类结果，不能仅根据总体 Accuracy 判断模型表现。')
    lines=['# ADNI_1 + ADNI_2 单次固定划分评估','',conclusion,'',
        '本报告只对应已预处理的两个批次，不代表完整 ADNI 论文数据或十折交叉验证。','',
        f'{len(manifest)} 次扫描，{len({r["subject"] for r in manifest})} 位受试者；标签策略 `{policy}`。',
        f'排除记录：{dict(Counter((r["diagnosis"],r["reason"]) for r in exclusions))}。',
        ('MCI 已统一并入 LMCI，EMCI 独立保留。' if policy=='mci_to_lmci_v1' else
         '历史数据策略：本报告不能作为当前 MCI→LMCI 队列的评估结果。'),
        f'两批按受试者联合划分，seed={config["params"]["seed"]}；各集合受试者完全不交叉。','',
        '| 集合 | 扫描数 | 受试者数 | '+' | '.join(names)+' |',
        '|---|---:|---:|'+'---:|'*len(names)]
    for role,v in split_summary.items():
        lines.append(f'| {role} | {v["samples"]} | {v["subjects"]} | '+' | '.join(str(v['classes'][c]) for c in names)+' |')
    lines+=['',f'模型 `{config["model"]}`；冻结的本地 Llama encoder 使用统一、无标签的 age/sex/site 模板生成 subject embeddings。',
        'site 通过完整 subject ID 从现有 ADNI_metadata.csv 匹配，各 visit 必须一致；原始 age/sex/diagnosis 保留各批已整理 manifest 的值。',
        '所有图与原始时序逐元素核对，完整 FC 使用 Pearson；逐扫描核对 AAL116 ROI 标签与 ROI prompt 顺序。',
        'age 统计和 site 编码仅在训练集合拟合。测试集预测在训练和模型选择完成后计算一次。','',
        f'训练设备 `{config["params"]["device"]}`；batch={config["params"]["batch_size"]}，Adam lr={config["params"]["init_lr"]}，weight_decay={config["params"]["weight_decay"]}，dropout={config["net_params"]["dropout"]}，',
        f'最多 {config["params"]["epochs"]} epoch，early_stopping_patience={config["params"].get("early_stopping_patience")}；实际完成 {result["epochs_completed"]} epoch，选择第 {result["best_epoch"]+1} epoch。',
        '分类损失未按类别加权；只按验证损失选择 checkpoint，没有根据测试结果选模型或调参。','',
        '| 指标 | 扫描级 | 受试者级 |','|---|---:|---:|']
    for key,label in [('accuracy','Accuracy'),('precision_macro','Precision (macro)'),('recall_macro','Recall (macro)'),
                      ('f1_macro','F1 (macro)'),('f1_weighted','F1 (weighted)'),('roc_auc_macro','ROC-AUC (macro, OvR)')]:
        lines.append(f'| {label} | {pct(metrics[key])} | {pct(subjects[key])} |')
    lines+=['','受试者级指标先平均同一测试受试者的全部扫描预测概率，再取 argmax；不是多数投票。',
        'Macro 对五个类别等权，weighted 按测试支持数加权；无正类预测时 Precision/F1 按 zero_division=0 记录。','',
        '| 类别 | 测试扫描数 | Precision | Recall | F1 |','|---|---:|---:|---:|---:|']
    for r in metrics['per_class']:
        lines.append(f'| {names[r["class_index"]]} | {r["support"]} | {pct(r["precision"])} | {pct(r["recall"])} | {pct(r["f1"])} |')
    lines+=['','| 测试扫描来源 | 扫描数 | Accuracy | Macro-F1 |','|---|---:|---:|---:|']
    for source,m in result['test_by_source'].items():
        lines.append(f'| {source} | {m["samples"]} | {pct(m["accuracy"])} | {pct(m["f1_macro"])} |')
    lines+=['','上述来源分组是同一次联合模型测试预测的分组统计，不是分别训练两个模型。',
        f'仅预测训练集多数类 {names[majority]} 的参照 Accuracy={pct(majority_metrics["accuracy"])}、Macro-F1={pct(majority_metrics["f1_macro"])}。',
        '该参照是多数类分类器。类别分布不平衡，少数类指标需结合支持数解释。',
        f'验证／测试未知 site 的扫描数分别为 {unknown_sites["val"]}/{unknown_sites["test"]}（映射到显式 unknown=0）。',
        '本次为单个固定划分，不能由此推断稳定的泛化性能或相对 baseline 的改进。','',
        '![Confusion matrix](confusion_matrix.png)','![Training curves](training_curves.png)','',
        '指标已从 `predictions.csv` 独立重算并与训练器输出比对一致。完整数字见 `metrics_report.json`；',
        '`confusion_matrix.csv` 行为真实类别、列为预测类别；模型位于对应 run 的 `outputs/checkpoints/.../fold_0/best.pt`。']
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(directory/'REPORT.md'),scan_metrics=metrics,subject_metrics=subjects)))


if __name__=='__main__': main()
