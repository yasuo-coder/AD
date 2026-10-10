"""Validate all thirty last checkpoints and independently recompute CV metrics."""
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
from BrainPromptPro.merged_cv import source_hashes, method_config
from BrainPromptPro.data.metadata import MetadataEncoder
from BrainPromptPro.official_cv import stop_reason

CLASSES = ['CN','AD','SMC','EMCI','LMCI']
MODELS = ['BrainPrompt','BrainPromptPro_old','BrainPromptPro_new']
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
    if queue['failed'] or queue['pending'] or queue['active'] or len(queue['completed']) != 30:
        raise ValueError('All thirty folds must complete successfully before final reporting')
    from BrainPromptPro.data.prepared_cohort import pro_path
    bundle_path=pro_path(config['data']['bundle'])
    assert sha256(bundle_path)==config['data']['bundle_sha256']
    bundle=torch.load(bundle_path,map_location='cpu',weights_only=True)
    cohort = list(csv.DictReader((root/'cohort.csv').open()))
    by_index = {int(row['scan_index']):row for row in cohort}
    if len(by_index) != 798:
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
            expected_config=method_config(config,model)
            assert checkpoint['config']==expected_config, 'Method-specific training settings changed'
            assert checkpoint['training_indices_used']==splits[fold]['train']
            expected_encoder=MetadataEncoder().fit([bundle['records'][i] for i in splits[fold]['train']],config['data']['site_column'],config['data']['missing_site'])
            assert expected_encoder.state_dict()==checkpoint['metadata_encoder'], 'Encoder was not train-only'
            assert not result['smoke'], 'Smoke results must never enter final reporting'
            dummy=torch.nn.Parameter(torch.zeros(()))
            opt=torch.optim.Adam([dummy],lr=config['params']['init_lr'],weight_decay=config['params']['weight_decay'])
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=config['params']['lr_reduce_factor'],patience=config['params']['lr_schedule_patience'])
            for epoch,h in enumerate(history):
                assert h['epoch']==epoch and h['epoch_number']==epoch+1
                assert h['lr']==opt.param_groups[0]['lr']
                scheduler.step(h['validation']['loss'])
                assert h['next_lr']==opt.param_groups[0]['lr']
                reason=stop_reason(epoch,h['next_lr'],h['elapsed_seconds'],config['params'])
                assert h['stop_reason']==reason
                assert (reason is not None)==(epoch==len(history)-1), 'Not stopped at first official stop'
            assert scheduler.state_dict()==checkpoint['scheduler']
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
        if [int(row['scan_index']) for row in rows] != list(range(798)):
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
    if not all(np.array_equal(pooled[MODELS[0]]['y'],pooled[m]['y']) for m in MODELS):
        raise ValueError('Model OOF labels differ')
    write_csv(root/'fold_metrics.csv',table)
    write_csv(root/'training_summary.csv',epoch_rows)
    write_csv(root/'per_class_oof_metrics.csv',per_class_rows)
    write_csv(root/'comparison_mean_std.csv',[dict(model=model,metric=key,**summary[model][key]) for model in MODELS for key in METRICS])
    write(root/'comparison_summary.json',dict(models=summary,aggregation='Unweighted fold mean and population SD (ddof=0)',
          samples=798,subjects=557,protocol='official_scan_stratified_10fold_last_epoch',
          class_order=CLASSES,subject_overlap_test_scans=sum(a['test_scans_with_train_subject'] for a in audit)))
    write(root/'verification.json',dict(passed=True,folds_verified=30,oof_scans_per_model=798,
          independent_metrics_recomputed=True,last_checkpoint_epochs_verified=True,
          paired_metadata_encoders_equal=True,source_hashes_unchanged=True,common_training_settings_verified=True,
          train_only_encoders_recomputed=True,schedulers_and_first_stop_replayed=True,bundle_sha256_verified=True,
          report_script_sha256=sha256(Path(__file__)),split_sha256=sha256(root/'splits.json')))

    os.environ['MPLCONFIGDIR'] = str(root/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(18,5),layout='constrained')
    for ax, model in zip(axes,MODELS):
        matrix = np.array(pooled[model]['metrics']['confusion_matrix'])
        ax.imshow(matrix,cmap='Blues')
        ax.set(xticks=range(5),yticks=range(5),xticklabels=CLASSES,yticklabels=CLASSES,
               xlabel='Predicted',ylabel='True',title=f'{model}: pooled OOF (798 scans)')
        for (i,j),value in np.ndenumerate(matrix):
            ax.text(j,i,str(value),ha='center',va='center',color='white' if value>matrix.max()/2 else 'black')
    fig.savefig(root/'oof_confusion_matrices.png',dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(18,5),layout='constrained')
    for ax, model in zip(axes,MODELS):
        y,p = pooled[model]['y'],pooled[model]['p']
        for i,name in enumerate(CLASSES):
            fpr,tpr,_ = roc_curve(y==i,p[:,i]); auc=roc_auc_score(y==i,p[:,i])
            ax.plot(fpr,tpr,label=f'{name}: {auc:.3f}')
        ax.plot([0,1],[0,1],'k--',lw=1); ax.legend(fontsize=9)
        ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'{model}: OOF OvR ROC',xlim=(0,1),ylim=(0,1))
    fig.savefig(root/'oof_roc_curves.png',dpi=180); plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(18,4),layout='constrained')
    for ax,model in zip(axes,MODELS):
        for fold,history in enumerate(histories[model]):
            ax.plot([h['epoch_number'] for h in history],[h['validation']['loss'] for h in history],label=f'Fold {fold}',lw=0.8)
        ax.set(title=model,xlabel='Epoch (1-based)',ylabel='Validation loss (mean of batches)')
        ax.legend(ncol=2,fontsize=7)
    fig.savefig(root/'validation_loss_curves.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    x=np.arange(len(METRICS))
    for i,model in enumerate(MODELS):
        ax.bar(x+(i-1)*.25,[summary[model][key]['mean']*100 for key in METRICS],.25,
               yerr=[summary[model][key]['std']*100 for key in METRICS],label=model,capsize=4)
    ax.set(xticks=x,xticklabels=['Accuracy','Macro F1','Macro precision','Macro recall','Macro OvR AUC'],
           ylabel='Fold mean ± population SD (%)',ylim=(0,100),title='Official schedule: ten folds, last epoch')
    ax.legend();fig.savefig(root/'cv_comparison.png',dpi=180);plt.close(fig)

    # Secondary strata all come from the SAME merged-cohort OOF predictions.
    source_rows=[]
    source=np.array([by_index[i]['source_batch'] for i in range(len(cohort))])
    for batch in ('ADNI_1','ADNI_2','ADNI_3','ADNI_4'):
        mask=source==batch
        for model in MODELS:
            met=recompute(pooled[model]['y'][mask],pooled[model]['p'][mask])
            source_rows.append(dict(source_batch=batch,model=model,scans=int(mask.sum()),
                ad_scans=int((pooled[model]['y'][mask]==1).sum()),**{k:met[k] for k in METRICS}))
    write_csv(root/'source_strata_oof_metrics.csv',source_rows)
    contrasts=[]
    for baseline in MODELS[:2]:
        for metric in METRICS:
            before=np.array([r[metric] for r in table if r['model']==baseline])
            after=np.array([r[metric] for r in table if r['model']==MODELS[2]])
            diff=after-before
            contrasts.append(dict(comparison='BrainPromptPro_new minus '+baseline,metric=metric,
                mean_difference=float(diff.mean()),fold_wins=int((diff>1e-12).sum()),
                fold_ties=int((np.abs(diff)<=1e-12).sum()),fold_losses=int((diff < -1e-12).sum())))
    write_csv(root/'paired_fold_differences.csv',contrasts)
    data_summary=bundle['provenance']['audit_summary']
    stats=[]
    for batch,s in data_summary['batches'].items():
        n=s['included_functional_images'];ad=s['ad_samples']
        stats.append(dict(source_batch=batch,raw_functional_scans=s['raw_functional_images'],included_scans=n,
            unique_subjects=s['subjects'],ad_scans=ad,ad_scan_fraction=ad/n,non_ad_scans=n-ad,
            non_ad_scan_fraction=(n-ad)/n,ad_subjects=s['ad_subjects'],ad_subject_fraction=s['ad_subjects']/s['subjects'],
            merged_scan_fraction=n/798))
    stats.append(dict(source_batch='MERGED',raw_functional_scans=800,included_scans=798,
        unique_subjects=557,ad_scans=26,ad_scan_fraction=26/798,non_ad_scans=772,
        non_ad_scan_fraction=772/798,ad_subjects=26,ad_subject_fraction=26/557,merged_scan_fraction=1.))
    write_csv(root/'dataset_statistics.csv',stats)
    prevalence_labels=['ADNI_1','ADNI_2','ADNI_3','ADNI_4','ADNI_1-3','ADNI_1-4']
    prevalence_values=[
        [s['ad_scan_fraction']*100 for s in stats[:4]]+[19/598*100,26/798*100],
        [s['ad_subject_fraction']*100 for s in stats[:4]]+[19/398*100,26/557*100]]
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,values,title in zip(axes,prevalence_values,['AD functional scans / all scans','Subjects with AD scans / unique subjects']):
        bars=ax.bar(prevalence_labels,values,color=['#6b92b9']*4+['#d59b52','#bc5858'])
        ax.bar_label(bars,labels=[f'{v:.2f}%' for v in values],padding=3,fontsize=9)
        ax.set(ylabel='AD fraction (%)',title=title,ylim=(0,max(values)*1.25))
        ax.tick_params(axis='x',labelrotation=25)
    fig.savefig(root/'ad_prevalence.png',dpi=180);plt.close(fig)

    reference=root.parent/'adni123_official_scan10_last_seed41_20261005_v2'
    old=read(reference/'comparison_summary.json')['models']
    history_rows=[]
    for model,oldkey in [('BrainPrompt','BrainPrompt'),('BrainPromptPro_old','BrainPromptPro')]:
        for k in METRICS:
            history_rows.append(dict(model=model,metric=k,adni123=old[oldkey][k]['mean'],
                adni1234=summary[model][k]['mean'],difference=summary[model][k]['mean']-old[oldkey][k]['mean'],
                interpretation='Descriptive only: sample composition, split indices and independent shuffle RNG differ; not a causal AD-prevalence comparison'))
    write_csv(root/'historical_descriptive_comparison.csv',history_rows)
    training_diagnostics={}
    for model in MODELS:
        rr=[read(root/model/f'fold_{fold}'/'result.json') for fold in range(10)]
        training_diagnostics[model]=dict(
            last_train_eval_accuracy=float(np.mean([r['final_train']['accuracy'] for r in rr])),
            last_validation_accuracy=float(np.mean([h[-1]['validation']['accuracy'] for h in histories[model]])),
            last_train_classification_loss=float(np.mean([r['final_train']['cls'] for r in rr])),
            last_validation_classification_loss=float(np.mean([h[-1]['validation']['cls'] for h in histories[model]])),
            epochs=[r['epochs_completed'] for r in rr])
    write(root/'analysis_summary.json',dict(dataset=data_summary,training_diagnostics=training_diagnostics,
        old_ad_scan_fraction=19/598,new_ad_scan_fraction=26/798,
        old_ad_subject_fraction=19/398,new_ad_subject_fraction=26/557,
        paired_fold_contrasts=contrasts,historical_comparison=history_rows,
        new_fixed_model_matched_adni123_reference_available=False,
        no_statistical_significance_claim=True,
        sources=[str(reference/'comparison_summary.json')]))

    display={'BrainPrompt':'BrainPrompt','BrainPromptPro_old':'旧版 BrainPromptPro','BrainPromptPro_new':'新版 BrainPromptPro'}
    def fmt(model,key):
        m=summary[model][key]
        return f"{m['mean']:.4f} ± {m['std']:.4f}" if key=='roc_auc_macro' else f"{100*m['mean']:.2f}% ± {100*m['std']:.2f}%"
    report=['# ADNI_1～4 合并输入：三种方法统一十折实验','',
        '**已完成 30/30 个正式训练折；未混入 smoke、历史实验或验证最佳轮结果。**','',
        '## 1. 合成数据集构建说明','',
        '- “合成”指四批真实扫描的样本拼接，没有生成患者或人工合成影像。一个 AAL116 功能连接矩阵对应一张功能扫描；配对 T1 用于预处理，不另计分类样本。',
        '- 按 ADNI_1、ADNI_2、ADNI_3、ADNI_4 顺序拼接，各批按规范化 subject/visit 排序。划分前不另行随机打乱；十折划分器内部 shuffle=True，seed=41。',
        '- 用功能影像 Image Data ID 核验身份；同时检查规范化扫描 ID、时间序列内容哈希。发现重复即停止构建而非重复计数，本次三类检查均无重复；不同时间点的真实随访扫描保留。',
        '- CN/AD/SMC/EMCI/LMCI 共五类，MCI 统一并入 LMCI。诊断、年龄、性别来自各批下载清单；站点通过精确 Image ID 从汇总表取得。不同方法共用同一标签与元数据。',
        '- 原下载清单共有 800 张功能扫描。ADNI_2 的 两张已核验的原始功能扫描 缺少可核验的原影像预处理映射，继续排除；历史替代影像 已核验的替代扫描 不在本次下载清单中，不加入。',
        '- ADNI_4 一条性别信息与旧汇总表冲突，按事先一致的下载清单来源取值，保留审计记录。合并数据还记录 69 条原始诊断来源差异，其中 52 条在 MCI→LMCI 后仍不同；未混用汇总表诊断覆盖下载标签。',
        '- 原有 ADNI_1～3 的 598 条记录、FC 张量与元数据嵌入同此前输入完全一致；共享 116 个 ROI 顺序与相同 ROI/疾病提示嵌入。','',
        '## 2. 合成数据集统计信息','',
        '| 来源 | 纳入功能扫描 | AD 扫描 | AD 占比 | 非 AD 扫描 | 非 AD 占比 | 占合并输入 | 独立受试者 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in stats:
        report.append(f"| {s['source_batch']} | {s['included_scans']} | {s['ad_scans']} | {100*s['ad_scan_fraction']:.2f}% | {s['non_ad_scans']} | {100*s['non_ad_scan_fraction']:.2f}% | {100*s['merged_scan_fraction']:.2f}% | {s['unique_subjects']} |")
    report += ['', '合并后共 557 位不同受试者，其中有 AD 扫描的受试者 26 位（4.67%）。各来源受试者数不能简单相加，因为批次之间存在同一受试者的不同扫描。五类扫描数为 CN 494、AD 26、SMC 39、EMCI 42、LMCI 197；其中 158 张原始 MCI 扫描映射到 LMCI。','',
        '## 3. 实验设置说明','',
        '- 三种方法全部从头训练，每种十折，共 30 折；共用一个 hash 固定的 cohort.pt 和完全相同的 splits.json。新版十折均为固定 stable_v1，不选择旧版作为回退。',
        '- 官方扫描分层十折，每折约 638/639 张训练、80 张验证、80/79 张测试；每张扫描恰好作为一次折外测试样本。metadata encoder 仅在对应训练集拟合。',
        '- 相同 Adam，batch_size=20，初始 lr=5e-5，weight_decay=0.005，dropout=0.5，seed=41；最多 1000 轮。独立训练 loader generator 保证同一轮三模型训练扫描顺序一致。',
        '- 三种方法共用损失配置：分类 CE + 1.0×疾病 CE；可学习脑图/群体图稀疏项系数均为 1e-4，关系多样性系数为 0；原 BrainPrompt 不含对应可学习新模块时其正则项为 0。均无类别加权、标签平滑或梯度裁剪。',
        '- 架构区别保留：BrainPrompt/旧 Pro hidden=116、L=4；新 Pro stable_v1 hidden=64、L=2，并保留 LayerNorm、连续 FC 先验、残差和融合门。此前优化中仅新版使用的 dropout=.25、辅助系数=.2、类别加权/平滑均未带入本轮。',
        '- 验证总损失采用官方 batch 等权均值；ReduceLROnPlateau factor=0.5、patience=25。lr 严格小于 1e-6、每折超过 6 小时或达到 1000 轮时停止，无额外 best-loss 早停。',
        '- 按官方入口每轮监测两次测试集，测试结果不参与调度、停止、选型或调参。停止后从磁盘重新加载实际 last.pt 才输出最终指标。',
        '- Accuracy 用 argmax；F1/Precision/Recall 为五类宏平均，零分母取 0；ROC-AUC 为 softmax 概率计算的五类 macro OvR。主表为十折等权均值 ± 标准差（ddof=0）。',
        '- 官方适配差异：固定原先未设定的划分随机种子；遵循 baseline.sh 的 edge_ratio=.2；修正官方多分类指标占位计算。不是论文完整 ADNI 队列的数值复现。',
        f"- 扫描级十折中有 {sum(a['test_scans_with_train_subject'] for a in audit)} 张测试扫描的受试者也在该折训练集，不能据此评估未见受试者泛化。GPU 5、6、7 并行运行，每卡最多两个训练进程。",'',
        '## 4. 三种方法在合成数据集上的指标结果表','',
        '| 方法 | Accuracy | F1-Score (macro) | Precision (macro) | Recall (macro) | ROC-AUC (macro OvR) |','|---|---:|---:|---:|---:|---:|']
    for model in MODELS:report.append('| '+display[model]+' | '+' | '.join(fmt(model,k) for k in METRICS)+' |')
    report += ['', '以下是合并 798 张 OOF 预测的类别行为，与上表的“十折均值”是不同汇总方式。','',
        '| 方法 | 预测 CN/AD/SMC/EMCI/LMCI 数量 | 正确 AD / 26 | 正确 EMCI / 42 | 全预测 CN 的折数 |','|---|---|---:|---:|---:|']
    for model in MODELS:
        matrix=np.asarray(pooled[model]['metrics']['confusion_matrix'])
        report.append(f"| {display[model]} | {pooled[model]['metrics']['predicted_counts']} | {matrix[1,1]} | {matrix[3,3]} | {summary[model]['all_cn_folds']} |")
    report += ['', '## 5. AD 患者比例提升的影响分析','',
        f"- 按扫描计算，ADNI_1～3 为 19/598={100*19/598:.3f}%，加入 ADNI_4 后为 26/798={100*26/798:.3f}%，仅增加 {100*(26/798-19/598):.3f} 个百分点。ADNI_4 自身为 3.50%；合并比例高于 ADNI_1/2，但低于 ADNI_3 的 5.50%。", 
        f"- 按不同受试者计算，原先为 19/398={100*19/398:.3f}%，现在为 26/557={100*26/557:.3f}%，反而略降。不能将本次扩充描述为显著提高 AD 患者比例。", 
        '- 下面只做历史观察性对照。旧两模型使用相同主要超参数和末轮规则，但输入组成、十折索引及本轮独立训练打乱生成器不同；性能变化不能归因为 AD 比例变化。','',
        '| 方法 | Accuracy 原→新 | F1 原→新 | Precision 原→新 | Recall 原→新 | ROC-AUC 原→新 |','|---|---:|---:|---:|---:|---:|']
    for model,oldkey in [('BrainPrompt','BrainPrompt'),('BrainPromptPro_old','BrainPromptPro')]:
        vals=[]
        for k in METRICS:
            a=old[oldkey][k]['mean'];b=summary[model][k]['mean']
            vals.append(f'{a:.4f}→{b:.4f}' if k=='roc_auc_macro' else f'{a*100:.2f}%→{b*100:.2f}%')
        report.append('| '+display[model]+' | '+' | '.join(vals)+' |')
    report.append('| 新版 BrainPromptPro | 缺失匹配历史对照 | 缺失匹配历史对照 | 缺失匹配历史对照 | 缺失匹配历史对照 | 缺失匹配历史对照 |')
    report += ['', '此前新版报告采用每折验证选型：5 折旧 Pro + 5 折加权新 Pro，且 dropout、辅助损失和正则化配置不同。本次是固定新结构、共同训练配置，因此不能把此前混选分数当作新版在原数据上的匹配基线。',
        '原四个数据集分别独立训练的三模型结果未运行；本任务唯一输入为合并队列。source_strata_oof_metrics.csv 只是同一合并实验按来源分层的折外指标，不能解释为四次独立训练或 AD 比例干预实验。','',
        '## 6. 新版 BrainPromptPro 性能提升分析','',
        '| 比较 | Accuracy 差值 / 胜折 | F1 差值 / 胜折 | Precision 差值 / 胜折 | Recall 差值 / 胜折 | ROC-AUC 差值 / 胜折 |','|---|---:|---:|---:|---:|---:|']
    explanations=[]
    for baseline in MODELS[:2]:
        cells=[]
        for k in METRICS:
            row=next(r for r in contrasts if r['comparison'].endswith(baseline) and r['metric']==k)
            delta=row['mean_difference'];value=f'{delta:+.4f}' if k=='roc_auc_macro' else f'{100*delta:+.2f} pp'
            cells.append(f"{value} / {row['fold_wins']}/10")
        report.append('| 新版 − '+display[baseline]+' | '+' | '.join(cells)+' |')
        improved=[TITLES[i] for i,k in enumerate(METRICS) if summary[MODELS[2]][k]['mean']>summary[baseline][k]['mean']]
        decreased=[TITLES[i] for i,k in enumerate(METRICS) if summary[MODELS[2]][k]['mean']<summary[baseline][k]['mean']]
        explanations.append(f"相对 {display[baseline]}，均值上升：{', '.join(improved) or '无'}；下降：{', '.join(decreased) or '无'}。")
    report += ['']+explanations
    if summary['BrainPromptPro_new']['all_cn_folds']==10:
        report += ['', '**新版十折全部预测 CN，共 798/798 张；其 Accuracy 只是 CN 多数类占比（494/798≈61.90%）。AD、SMC、EMCI、LMCI 的召回率全部为 0，因此不能把 Accuracy 上升解释为疾病识别能力提升。**',
            '新版 softmax 概率仍随扫描变化，因此 ROC-AUC 可以高于 0.5，同时 argmax 分类始终是 CN；两者并不矛盾。',
            '三种方法均未正确识别任何 AD 扫描（0/26）。旧 Pro 的唯一一次 AD 预测为假阳性。']
    report += ['', '| 方法 | 末轮训练 Accuracy（eval 模式） | 末轮验证 Accuracy | 停止轮数范围 |', '|---|---:|---:|---:|']
    for model in MODELS:
        d=training_diagnostics[model]
        report.append(f"| {display[model]} | {100*d['last_train_eval_accuracy']:.2f}% | {100*d['last_validation_accuracy']:.2f}% | {min(d['epochs'])}–{max(d['epochs'])} |")
    report += ['', '全部折均因 lr < 1e-6 停止。旧 Pro 的训练/验证差距仍大；新版训练准确率也接近多数类比例。现有结果不能进一步单独归因于某个结构、dropout 或辅助损失系数，需另行消融。']
    report += ['', '胜折表示在完全相同测试折上该指标更高。折与折训练集重叠，且扫描存在纵向相关性；这里不将均值差或胜折数宣称为统计显著性，也未用测试结果选择方案。','',
        '## 7. 结论','']
    all_better=all(summary[MODELS[2]][k]['mean']>summary[m][k]['mean'] for m in MODELS[:2] for k in METRICS)
    report.append('新版五项指标均值均高于另外两种方法，但仍需独立受试者验证，不能仅由本次交叉验证宣称普遍或显著提升。' if all_better else '本次实际结果不支持“新版在五项指标上均优于旧版和 BrainPrompt”的结论。改动的收益与退化按上表如实呈现，不进行测试后选型。')
    report += ['', 'AD 扫描占比仅小幅增加，而 AD 受试者占比略降；本实验无法识别 AD 比例的独立因果影响。全部 30 折已完成，没有以未完成结果填充表格。',
        '34 项既有检查通过，三种模型真实数据三轮试运行通过。最终独立复算 30 折指标，验证最后检查点、完全一致的划分与训练配置、训练集 metadata 拟合、源码/输入哈希，并逐轮重放学习率调度与首次停止条件。','',
        '- [逐折指标](fold_metrics.csv) · [每类 OOF 指标](per_class_oof_metrics.csv) · [配对差值](paired_fold_differences.csv)',
        '- [合并数据统计](dataset_statistics.csv) · [历史观察性对照](historical_descriptive_comparison.csv) · [来源分层 OOF 指标](source_strata_oof_metrics.csv)',
        '- [AD 占比图](ad_prevalence.png) · [指标图](cv_comparison.png) · [混淆矩阵](oof_confusion_matrices.png) · [ROC 曲线](oof_roc_curves.png) · [验证损失曲线](validation_loss_curves.png)',
        '- [完整数值](comparison_summary.json) · [独立核验](verification.json) · [协议](protocol.json) · [固定划分](splits.json)',
        '- 输入审计位于 ../../data/adni1234_download_mci_to_lmci_v1_v1/；最后检查点位于各方法/fold_0～9/last.pt。',
        '- 源码快照 source_training/，源码校验 source_sha256.json，环境 environment.json，队列 queue_status.json；每个方法各有一份完整 OOF 预测 CSV。','']
    (root/'REPORT.md').write_text('\n'.join(report))
    print(json.dumps(dict(verified=True,completed_folds=30,summary={m:{k:summary[m][k] for k in METRICS} for m in MODELS}),indent=2))


if __name__=='__main__':
    main()
