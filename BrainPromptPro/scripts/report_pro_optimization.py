"""Independent probability/selection/checkpoint audit and optimization report."""
import argparse,csv,json,os,shutil,sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support,roc_auc_score,roc_curve
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from BrainPromptPro.data.brainnet_pro import sha256
from BrainPromptPro.optimization_cv import experiment,hashes,REFERENCE

KEYS=['accuracy','f1_macro','precision_macro','recall_macro','roc_auc_macro']
CLASSES=['CN','AD','SMC','EMCI','LMCI']
AUDIT=Path('/data2/panyao_data/ADNI/model_audits/pro_optimization_20261006')

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csv_write(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def metrics(y,p):
    pred=p.argmax(1);precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=range(5),zero_division=0)
    out=dict(accuracy=float(accuracy_score(y,pred)),precision_macro=float(precision.mean()),recall_macro=float(recall.mean()),f1_macro=float(f1.mean()),
             roc_auc_macro=float(roc_auc_score(y,p,multi_class='ovr',average='macro',labels=range(5))),
             confusion_matrix=confusion_matrix(y,pred,labels=range(5)).tolist(),predicted_counts=np.bincount(pred,minlength=5).tolist())
    out['per_class']=[dict(class_name=c,support=int(support[i]),predicted=int((pred==i).sum()),precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),roc_auc=float(roc_auc_score(y==i,p[:,i]))) for i,c in enumerate(CLASSES)]
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args();root=experiment(args.run_id)
    queue=read(root/'queue_status.json')
    if queue['failed'] or queue['active'] or queue['pending'] or len(queue['completed'])!=20:raise ValueError('Incomplete candidate training')
    if hashes()!=read(root/'source_sha256.json'):raise ValueError('Source changed')
    selection=read(root/'selection.json');splits=read(root/'splits.json');selection_hash=sha256(root/'selection.json')
    if sha256(root/'splits.json')!=sha256(REFERENCE/'splits.json'):raise ValueError('Splits changed')
    cohort={int(x['scan_index']):x for x in csv.DictReader((REFERENCE/'cohort.csv').open())}
    selected_rows=[];fold_rows=[];val_rows=[];histories={};all_prob=[];all_y=[]
    for chosen in selection['folds']:
        fold=chosen['fold'];options=[]
        for candidate in ['legacy','stable','stable_weighted']:
            v=read(root/candidate/f'fold_{fold}'/'validation_result.json');options.append(v)
            vp=list(csv.DictReader((root/candidate/f'fold_{fold}'/'validation_predictions.csv').open()))
            if [int(x['scan_index']) for x in vp]!=splits[fold]['val']:raise ValueError('Wrong validation indices')
            vy=np.array([int(x['true_label']) for x in vp]);prob=np.array([[float(x['p_'+c]) for c in CLASSES] for x in vp]);vm=metrics(vy,prob)
            for k in KEYS:
                if not np.isclose(vm[k],v['validation'][k],atol=1e-10,rtol=0):raise ValueError('Validation metrics disagree')
            train_accuracy=(read(REFERENCE/'BrainPromptPro'/f'fold_{fold}'/'result.json')['final_train']['accuracy']
                            if candidate=='legacy' else v['train']['accuracy'])
            val_rows.append(dict(fold=fold,candidate=candidate,epochs=v['epochs_completed'],train_accuracy=train_accuracy,
                                 **{k:vm[k] for k in KEYS},selected=candidate==chosen['candidate']))
            if candidate!='legacy':
                h=[json.loads(line) for line in (root/candidate/f'fold_{fold}'/'history.jsonl').read_text().splitlines()]
                if any('test' in item for item in h):raise ValueError('Training accessed test')
                ck=torch.load(v['checkpoint'],map_location='cpu',weights_only=True)
                if ck['last_epoch']!=h[-1]['epoch'] or ck['epochs_completed']!=len(h) or ck['split']!=splits[fold]:raise ValueError('Not actual last model')
                if sha256(v['checkpoint'])!=v['checkpoint_sha256']:raise ValueError('Checkpoint changed')
                if ck['metadata_encoder']['fit_ids']!=[cohort[i]['subject_id'] for i in splits[fold]['train']]:raise ValueError('Encoder not trained on TRAIN only')
                # Reconstruct class weights solely from this fold's training labels.
                power=ck['config']['params']['class_weight_power'];weights=ck['config']['params']['class_weights']
                if power:
                    counts=np.bincount([int(cohort[i]['true_label']) for i in splits[fold]['train']],minlength=5)
                    expected=counts.astype(float)**(-power);expected/=np.sum(expected*counts/counts.sum())
                    if not np.allclose(weights,expected,rtol=1e-6):raise ValueError('Weights not fitted from TRAIN')
                histories[(candidate,fold)]=h
                del ck
        winner=max(options,key=lambda r:(r['validation']['f1_macro'],r['validation']['roc_auc_macro'],r['candidate']))
        if winner['candidate']!=chosen['candidate']:raise ValueError('Selection does not follow frozen validation rule')
        out=root/'selected'/f'fold_{fold}';result=read(out/'result.json')
        if result['selection_sha256']!=selection_hash or result['last_epoch']!=winner['last_epoch']:raise ValueError('Selected checkpoint differs')
        if result['checkpoint_sha256']!=sha256(result['checkpoint']):raise ValueError('Selected checkpoint changed')
        predictions=list(csv.DictReader((out/'predictions.csv').open()))
        if [int(x['scan_index']) for x in predictions]!=splits[fold]['test']:raise ValueError('Wrong test indices')
        for row in predictions:
            reference=cohort[int(row['scan_index'])]
            if row['subject_id']!=reference['subject_id'] or row['true_label']!=reference['true_label']:raise ValueError('Test identity/label mismatch')
            row.update(fold=fold,candidate=chosen['candidate'])
        y=np.array([int(row['true_label']) for row in predictions]);p=np.array([[float(row['p_'+c]) for c in CLASSES] for row in predictions])
        if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-6):raise ValueError('Invalid probabilities')
        m=metrics(y,p)
        for k in KEYS:
            if not np.isclose(m[k],result['test'][k],atol=1e-10,rtol=0):raise ValueError('Independent test metric mismatch')
        if m['confusion_matrix']!=result['test']['confusion_matrix']:raise ValueError('Confusion mismatch')
        if not np.array_equal(p.argmax(1),[int(x['predicted_label']) for x in predictions]):raise ValueError('Predictions mismatch')
        fold_rows.append(dict(fold=fold,candidate=chosen['candidate'],epochs=result['epochs_completed'],**{k:m[k] for k in KEYS},**{'predicted_'+c:m['predicted_counts'][i] for i,c in enumerate(CLASSES)}))
        selected_rows.extend(predictions)
    selected_rows.sort(key=lambda r:int(r['scan_index']))
    if [int(x['scan_index']) for x in selected_rows]!=list(range(598)):raise ValueError('OOF coverage incomplete')
    y=np.array([int(x['true_label']) for x in selected_rows]);p=np.array([[float(x['p_'+c]) for c in CLASSES] for x in selected_rows])
    pooled=metrics(y,p)
    meanstd={k:dict(mean=float(np.mean([r[k] for r in fold_rows])),std=float(np.std([r[k] for r in fold_rows],ddof=0))) for k in KEYS}
    reference=read(REFERENCE/'comparison_summary.json')['models']
    summary=dict(models={'BrainPrompt':reference['BrainPrompt'],'BrainPromptPro_original':reference['BrainPromptPro'],
                         'BrainPromptPro_optimized':dict(**meanstd,oof=pooled)},selection_counts=dict(Counter(r['candidate'] for r in fold_rows)),
                 aggregation='Ten-fold mean ± population SD (ddof=0)',scope='Same ADNI_1–3, 598 scans / 398 subjects',
                 protocol='Same outer scan splits, last checkpoint; per-fold validation candidate selection')
    write(root/'comparison_summary.json',summary)
    csv_write(root/'fold_metrics.csv',fold_rows);csv_write(root/'validation_candidates.csv',val_rows)
    csv_write(root/'oof_predictions.csv',selected_rows);csv_write(root/'per_class_metrics.csv',pooled['per_class'])
    csv_write(root/'comparison_mean_std.csv',[dict(model=name,metric=k,**values[k]) for name,values in summary['models'].items() for k in KEYS])
    original_sources=read(REFERENCE/'source_sha256.json')
    unchanged=all(sha256(path)==value for path,value in original_sources.items() if path.startswith('/home/panyao/AD/BrainPrompt/'))
    if not unchanged:raise ValueError('Original BrainPrompt source changed')
    verification=dict(passed=True,training_jobs=20,selected_folds=10,oof_scans=598,selection_recomputed_from_validation=True,
                      class_weights_train_only=True,age_site_encoder_train_only=True,only_last_checkpoints=True,
                      original_brainprompt_unchanged=True,source_hashes_match_training_plan=True,
                      metrics_recomputed_from_csv=True,selection_sha256=selection_hash,script_sha256=sha256(__file__))
    write(root/'verification.json',verification)
    destination=root/'diagnostics';destination.mkdir(exist_ok=True)
    for name in ['ANALYSIS.md','history_summary.json','last_checkpoint_diagnostics.json','diagnostic_summary.json','branch_gradient_diagnostics.json','tests.log']:
        shutil.copyfile(AUDIT/name,destination/name)
    os.environ['MPLCONFIGDIR']=str(root/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained');x=np.arange(5)
    for i,(name,values) in enumerate(summary['models'].items()):
        ax.bar(x+(i-1)*.25,[100*values[k]['mean'] for k in KEYS],.25,yerr=[100*values[k]['std'] for k in KEYS],capsize=3,label=name.replace('BrainPromptPro_','Pro '))
    ax.set(xticks=x,xticklabels=['Accuracy','Macro F1','Macro precision','Macro recall','Macro OvR AUC'],ylabel='Fold mean ± SD (%)',ylim=(0,100))
    ax.legend();fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,(name,value) in zip(axes,summary['models'].items()):
        matrix=np.array(value['oof']['confusion_matrix']);ax.imshow(matrix,cmap='Blues')
        ax.set(xticks=range(5),yticks=range(5),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='True',title=name.replace('BrainPromptPro_','Pro '))
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>matrix.max()/2 else 'black')
    fig.savefig(root/'confusion_matrices.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,name in zip(axes,['BrainPrompt','BrainPromptPro','optimized']):
        records=(selected_rows if name=='optimized' else list(csv.DictReader((REFERENCE/f'{name}_oof_predictions.csv').open())))
        truth=np.array([int(r['true_label']) for r in records]);prob=np.array([[float(r['p_'+c]) for c in CLASSES] for r in records])
        for i,label in enumerate(CLASSES):
            fpr,tpr,_=roc_curve(truth==i,prob[:,i]);auc=roc_auc_score(truth==i,prob[:,i])
            ax.plot(fpr,tpr,label=f'{label}: {auc:.3f}')
        ax.plot([0,1],[0,1],'k--',lw=.8);ax.legend(fontsize=8)
        ax.set(title=name+' (OOF)',xlabel='False positive rate',ylabel='True positive rate',xlim=(0,1),ylim=(0,1))
    fig.savefig(root/'roc_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,key,title in zip(axes,['f1_macro','roc_auc_macro'],['Validation macro F1','Validation macro OvR AUC']):
        for name in ['legacy','stable','stable_weighted']:
            a=sorted([r for r in val_rows if r['candidate']==name],key=lambda r:r['fold'])
            ax.plot(range(10),[r[key] for r in a],marker='o',label=name)
        ax.set(xlabel='Fold (0-based)',ylabel=title,xticks=range(10),ylim=(0,1));ax.legend(fontsize=8)
    fig.savefig(root/'validation_comparison.png',dpi=180);plt.close(fig)
    names=['BrainPrompt','Pro original','Stable','Stable weighted','Validation selected']
    old_history=read(AUDIT/'history_summary.json')
    train_values=[old_history['BrainPrompt']['mean']['train_accuracy']]
    val_values=[old_history['BrainPrompt']['mean']['val_accuracy']]
    for candidate in ['legacy','stable','stable_weighted',None]:
        subset=[r for r in val_rows if (r['selected'] if candidate is None else r['candidate']==candidate)]
        train_values.append(np.mean([r['train_accuracy'] for r in subset]))
        val_values.append(np.mean([r['accuracy'] for r in subset]))
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained');x=np.arange(5)
    ax.bar(x-.18,np.array(train_values)*100,.36,label='Train (eval mode)')
    ax.bar(x+.18,np.array(val_values)*100,.36,label='Validation')
    ax.set(xticks=x,xticklabels=names,ylabel='Mean accuracy (%)',ylim=(0,100),title='Last-checkpoint train / validation gap')
    ax.legend();fig.savefig(root/'training_validation_gap.png',dpi=180);plt.close(fig)
    text=['# BrainPromptPro 优化与十折验证结果','',
          '**本轮没有实现整体性能提升：宏平均 F1、Precision、Recall 略升，Accuracy 和 ROC-AUC 下降。旧 Pro 保持默认，新结构保留为实验配置，不据此宣称超越原版。**','',
          '已完成两个新候选各十折、共 20 次完整训练。每折仅按该折验证 macro F1（同分再比较 AUC）从两个候选及原 Pro 末轮保留对照中选型，锁定全部选择后才执行新测试。以下比较的是这一验证选型流程。','',
          '| 方法 | Accuracy | F1-Score (macro) | Precision (macro) | Recall (macro) | ROC-AUC (macro OvR) |',
          '|---|---:|---:|---:|---:|---:|']
    for name,values in summary['models'].items():
        cells=[f"{values[k]['mean']:.4f} ± {values[k]['std']:.4f}" if k=='roc_auc_macro' else f"{100*values[k]['mean']:.2f}% ± {100*values[k]['std']:.2f}%" for k in KEYS]
        text.append('| '+name+' | '+' | '.join(cells)+' |')
    text+=['','主表为十折均值 ± 标准差（ddof=0），全部使用最后一轮 checkpoint。F1/Precision/Recall 为宏平均，AUC 为五类 softmax 概率的 macro OvR。',
           '',f"验证选择计数：{summary['selection_counts']}。`legacy` 表示验证集选择保留原 Pro；并未隐瞒不如旧模型的候选。",
           '',f"优化流程合并 598 个折外预测的数量（CN/AD/SMC/EMCI/LMCI）：{pooled['predicted_counts']}。",'',
           '| 类别 | 扫描数 | 预测数 | Precision | Recall | F1 | OvR AUC |','|---|---:|---:|---:|---:|---:|---:|']
    for r in pooled['per_class']:
        text.append(f"| {r['class_name']} | {r['support']} | {r['predicted']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} |")
    text+=['','## 训练/验证对照（不用于报告测试优胜者）','',
           '| 候选 | 平均训练 Accuracy | 平均验证 Accuracy | 平均验证 macro F1 | 平均验证 macro AUC | 入选折数 |',
           '|---|---:|---:|---:|---:|---:|']
    for candidate in ['legacy','stable','stable_weighted']:
        rows=[r for r in val_rows if r['candidate']==candidate]
        values=[float(np.mean([r[k] for r in rows])) for k in ['train_accuracy','accuracy','f1_macro','roc_auc_macro']]
        text.append(f'| {candidate} | {100*values[0]:.2f}% | {100*values[1]:.2f}% | {values[2]:.4f} | {values[3]:.4f} | {sum(r["selected"] for r in rows)} |')
    text+=['','上述每类表由合并的 OOF 预测重算，与十折均值不是同一汇总方式。','','## 原因与修改',
           '', '详见 [完整诊断与优化设计](diagnostics/ANALYSIS.md)：过拟合、近共享稠密脑图、近均匀疾病注意力、未受控的分支尺度和类别不平衡；BN 重估与组批敏感性对照不支持把它们视为此次主要原因。',
           '', '优化保留三个模块：连续 FC 先验与 FC 残差、带残差的疾病查询读出、带软元数据先验的多关系图；加入 LayerNorm 和小初值融合门。宽度/深度缩减后参数由 1,005,126 降至 477,161。疾病辅助系数改为 0.2、dropout=0.25、label smoothing=0.05；类别平方根倒数加权有独立对照。',
           '', '结构改动是一个组合方案，本次没有完成每个结构因素的全部单因素消融，不能将收益量化归属于某一项。原 BrainPrompt 核心源码未修改，旧 Pro 源码归档在 source_before/，旧模型默认与 checkpoint 兼容。',
           '', '入选新结构的五折配对训练诊断显示：跨扫描边权标准差 0.000626→0.045358，注意力归一化熵 0.999606→0.916485，ROI 表征余弦 0.992800→0.287519。内部结构现象得到改变，但这并未转化为整体测试收益。详见 [逐折结构统计](selected_structure.json)；每折 `training_structure.pt` 保存同一诊断批次的图、注意力和关系权重。',
           '', '## 解释范围','',
           '- 输入仍为 ADNI_1–3，598 张扫描/398 人；不是论文完整 ADNI 数据。标签与原输入保持一致。',
           '- 原有扫描分层十折完全相同，仍有 291 张测试扫描的受试者出现在该折训练集，不能宣称未见受试者泛化。',
           '- 保留官方外层调度和末轮规则，优化版增加验证选型并取消拟合阶段的测试监测；这是优化实验，不是原发布入口逐行不变的复现。',
           '- 这批测试数据此前已被查看，结果属于探索性比较；没有据新测试分数更换候选。未证明统计显著性，也未证明临床可用性。',
           '- 类别加权可能提高少数类召回而牺牲总体准确率；最终结果如实列出，不保证所有指标同时上升。',
           '', '## 核验与复运行','',
           '31 项测试通过，真实数据三轮试运行通过。已独立复算验证选择与测试指标，核对 20 个候选末轮、训练集权重/metadata 统计、相同十折索引、598 个 OOF ID 恰好各出现一次以及源码校验值。',
           '', '在 `/home/panyao/AD` 的 brainprompt 环境使用新的 RUN_ID：','', '```bash',
           'python -m BrainPromptPro.optimization_cv prepare --run-id NEW_RUN_ID',
           'python -m BrainPromptPro.optimization_cv train --run-id NEW_RUN_ID --candidate stable_weighted --fold 0 --smoke',
           'python BrainPromptPro/scripts/run_optimization_queue.py --run-id NEW_RUN_ID',
           'CUDA_VISIBLE_DEVICES=5 python -m BrainPromptPro.optimization_cv legacy-validation --run-id NEW_RUN_ID',
           'python -m BrainPromptPro.optimization_cv select --run-id NEW_RUN_ID',
           'CUDA_VISIBLE_DEVICES=5 python -m BrainPromptPro.optimization_cv evaluate --run-id NEW_RUN_ID',
           'python BrainPromptPro/scripts/report_pro_optimization.py --run-id NEW_RUN_ID','```','',
           '队列使用 GPU 5、6、7；运行前检查资源情况。已有实验目录禁止覆盖。', '',
           '- [验证候选对照](validation_candidates.csv) · [选择清单](selection.json) · [逐折测试指标](fold_metrics.csv)',
           '- [598 张扫描的 OOF 概率](oof_predictions.csv) · [每类指标](per_class_metrics.csv) · [均值/标准差](comparison_mean_std.csv)',
           '- [指标对比图](comparison.png) · [混淆矩阵](confusion_matrices.png) · [ROC 曲线](roc_curves.png) · [验证对照图](validation_comparison.png) · [训练/验证差距](training_validation_gap.png)',
           '- [独立核验](verification.json) · [冻结实验方案](plan.json) · [源码哈希](source_sha256.json)',
           '- 检查点路径以 `selection.json` 为准；新候选位于 `stable/fold_*/last.pt` 与 `stable_weighted/fold_*/last.pt`，保留对照引用原始结果目录。','']
    (root/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
