"""Independently verify all locked TEST outputs and export anonymous metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from BrainPromptPro.metrics import classification_metrics
from BrainPromptPro.scripts.adni_1to7.audit_dataset import sha
from BrainPromptPro.scripts.adni_1to7.summarize_validation import METRICS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    lock=json.loads(a.lock.read_text());lockhash=sha(a.lock);root=a.lock.parent
    if not lock['development_complete'] or lock['test_selection_policy']!='NO_TEST_BASED_SELECTION':raise ValueError('Not a frozen study')
    rows=[];identities=set()
    for entry in lock['evaluations']:
        identity=(entry['protocol'],entry['experiment'],entry['seed'])
        if identity in identities:raise ValueError('Duplicate locked evaluation')
        identities.add(identity);run=root/'final_test'/entry['protocol']/entry['experiment']/f"seed{entry['seed']}"
        result=json.loads((run/'result.private.json').read_text())
        if result['role']!='test' or result['status']!='COMPLETE' or not result['test_accessed']:raise ValueError('Incomplete TEST output')
        if result['selection_lock_sha256']!=lockhash or result['evaluator_sha256']!=lock['evaluator_sha256']:raise ValueError('Unsealed evaluation')
        for key in ['protocol','experiment','seed','checkpoint_sha256','dataset_hash','split_hash']:
            if result[key]!=entry[key]:raise ValueError('Evaluation identity mismatch: '+key)
        if sha(Path(entry['run'])/'best_validation.pt')!=entry['checkpoint_sha256']:raise ValueError('Locked checkpoint changed')
        payload=torch.load(run/'predictions.private.pt',map_location='cpu',weights_only=True)
        truth=payload['truth'].numpy();prob=payload['probability'].numpy()
        if not np.isfinite(prob).all() or not np.allclose(prob.sum(-1),1,atol=1e-5):raise ValueError('Invalid probability')
        m=classification_metrics(truth,prob,5)
        for key in METRICS:
            if m[key] is None or result['metrics'][key] is None:
                if m[key]!=result['metrics'][key]:raise ValueError('AUC availability mismatch')
            elif not np.isclose(m[key],result['metrics'][key],atol=1e-10,rtol=0):raise ValueError('Independent metric mismatch')
        if m['confusion_matrix']!=result['metrics']['confusion_matrix']:raise ValueError('Confusion matrix mismatch')
        training=json.loads((Path(entry['run'])/'result.json').read_text());names=['CN','AD','SMC','EMCI','LMCI'];pred=prob.argmax(-1)
        rows.append(dict(protocol=entry['protocol'],experiment=entry['experiment'],seed=entry['seed'],checkpoint_sha256=entry['checkpoint_sha256'],dataset_hash=entry['dataset_hash'],split_hash=entry['split_hash'],metrics={k:m[k] for k in METRICS},
            per_class={name:{k:x[k] for k in ['precision','recall','f1','specificity']} for name,x in zip(names,m['per_class'])},prediction_fraction={name:float(np.mean(pred==i)) for i,name in enumerate(names)},mean_class_probability=dict(zip(names,prob.mean(0).tolist())),
            classification_collapse=len(np.unique(pred))==1,training_parameters=training['parameter_count'],trainable_parameters=training['trainable_parameters'],training_seconds=training['elapsed_seconds'],test_inference_seconds=result['metrics']['inference_seconds']))
    groups=[]
    for protocol,exp in dict.fromkeys((r['protocol'],r['experiment']) for r in rows):
        group=[r for r in rows if r['protocol']==protocol and r['experiment']==exp]
        if sorted(x['seed'] for x in group)!=[42,2026,3407]:raise ValueError('Incomplete three-seed group')
        values={}
        for metric in METRICS:
            data=[x['metrics'][metric] for x in group]
            values[metric]=None if any(x is None for x in data) else dict(mean=float(np.mean(data)),sample_sd=float(np.std(data,ddof=1)))
        groups.append(dict(protocol=protocol,experiment=exp,metrics=values))
    a.output.mkdir(parents=True,exist_ok=False)
    doc=dict(evaluation='locked TEST; no test-based selection or further tuning',isolation_scope=lock['isolation_scope'],verification='PASSED',selection_lock_sha256=lockhash,primary_candidate_from_validation=lock['primary_candidate'],runs=rows,groups=groups)
    (a.output/'test_summary.public.json').write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
    for protocol in dict.fromkeys(x['protocol'] for x in groups):
        lines=[f'# 七批 {protocol} 协议冻结测试结果','','模型/超参数/checkpoint已在TEST访问前锁定；主要候选由验证集选择。不得用下表重新挑选模型或继续调参。','','| 方法 | Accuracy | Macro-F1 | Macro-P | Macro-R / BalAcc | ROC-AUC |','|---|---:|---:|---:|---:|---:|']
        for group in groups:
            if group['protocol']!=protocol:continue
            cells=[]
            for metric in ['accuracy','f1_macro','precision_macro','recall_macro','roc_auc_macro']:
                x=group['metrics'][metric];cells.append('不可计算（缺类）' if x is None else f"{100*x['mean']:.2f} ± {100*x['sample_sd']:.2f}")
            lines.append('| '+group['experiment']+' | '+' | '.join(cells)+' |')
        lines+=['','单位%；均值±三训练seed样本SD，不是患者层面的置信区间。每类指标、预测比例、Weighted-F1、参数量/耗时见匿名JSON；原始混淆矩阵和逐样本证据仅在受控目录。不同协议不得混合。协议间存在样本角色交叉，历史研究也有开发暴露；仅各自协议内的TRAIN/VAL/TEST隔离成立，不能称为全研究范围内未使用或全新外部验证。']
        (a.output/f'Test_Results_{protocol}.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASSED',verified_test_runs=len(rows),selection_lock_sha256=lockhash)))


if __name__=='__main__':main()
