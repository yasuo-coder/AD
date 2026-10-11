"""Compute paired module effects from a complete, independently verified matrix."""
import argparse
import json
from pathlib import Path
import numpy as np

SEEDS = [42, 3407, 2026]
PAIRS = {
    'P2': [('A1','A0'),('A4','A2'),('A5','A3'),('A7','A6')],
    'P3': [('A2','A0'),('A4','A1'),('A6','A3'),('A7','A5')],
    'P5': [('A3','A0'),('A5','A1'),('A6','A2'),('A7','A4')],
}
METRICS = ['accuracy','f1_macro','precision_macro','recall_macro','balanced_accuracy','roc_auc_macro','f1_weighted']
CLASSES = ['CN','AD','SMC','EMCI','LMCI']


def mean_sd(values):
    return dict(mean=float(np.mean(values)), sample_sd=float(np.std(values,ddof=1)), values=list(map(float,values)))


def analyze(rows):
    data={}
    for row in rows:
        if row['experiment'] not in [f'A{i}' for i in range(8)]:continue
        key=(row['experiment'],row['seed'])
        if key in data:raise ValueError('Duplicate experiment/seed')
        data[key]=row
    expected={(f'A{i}',s) for i in range(8) for s in SEEDS}
    if set(data)!=expected:raise ValueError('A0-A7 must each have exactly the three registered seeds')
    result={}
    for module,pairs in PAIRS.items():
        backgrounds=[]
        for on,off in pairs:
            deltas={m:mean_sd([data[(on,s)]['metrics'][m]-data[(off,s)]['metrics'][m] for s in SEEDS]) for m in METRICS}
            recalls={c:mean_sd([data[(on,s)]['per_class_recall'][c]-data[(off,s)]['per_class_recall'][c] for s in SEEDS]) for c in CLASSES}
            backgrounds.append(dict(contrast=on+' - '+off,metrics=deltas,per_class_recall=recalls))
        # Average four backgrounds inside each training seed first. The four
        # correlated contrasts must not be treated as 12 independent subjects.
        marginal={m:mean_sd([np.mean([data[(on,s)]['metrics'][m]-data[(off,s)]['metrics'][m] for on,off in pairs]) for s in SEEDS]) for m in METRICS}
        recalls={c:mean_sd([np.mean([data[(on,s)]['per_class_recall'][c]-data[(off,s)]['per_class_recall'][c] for on,off in pairs]) for s in SEEDS]) for c in CLASSES}
        result[module]=dict(backgrounds=backgrounds,mean_marginal=marginal,mean_recall_marginal=recalls,
                            positive_f1_backgrounds=sum(b['metrics']['f1_macro']['mean']>0 for b in backgrounds))
    full={m:mean_sd([data[('A7',s)]['metrics'][m]-data[('A0',s)]['metrics'][m] for s in SEEDS]) for m in METRICS}
    return dict(seeds=SEEDS,modules=result,full_minus_baseline=full,
                uncertainty='Paired training-seed sample SD, not patient confidence intervals or a significance test')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    report=json.loads(a.summary.read_text())
    if report['verification']!='PASSED' or report.get('protocol','sample')!='sample':raise ValueError('Only verified main sample-protocol results')
    if len({x['common_training_hash'] for x in report['runs']})!=1:raise ValueError('Mixed training protocols')
    result=analyze(report['runs']);result.update(evaluation='validation only; test not evaluated',summary_sha256=__import__('hashlib').sha256(a.summary.read_bytes()).hexdigest())
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'ablation_effects.public.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# 三模块配对消融','','同一冻结划分和相同训练seed；先在每个seed内计算四种背景平均差，再计算三seed均值±样本SD。以下单位为百分点，不是受试者置信区间。不采用test选型。','','| 模块/背景 | ΔAccuracy | ΔMacro-F1 | ΔMacro-P | ΔMacro-R / BalAcc | ΔAUC |','|---|---:|---:|---:|---:|---:|']
    keys=['accuracy','f1_macro','precision_macro','recall_macro','roc_auc_macro']
    for name,item in result['modules'].items():
        for row in item['backgrounds']+[dict(contrast='四背景均值',metrics=item['mean_marginal'])]:
            values=[f"{100*row['metrics'][m]['mean']:+.2f} ± {100*row['metrics'][m]['sample_sd']:.2f}" for m in keys]
            lines.append('| '+name+' / '+row['contrast']+' | '+' | '.join(values)+' |')
    lines+=['','每类Recall边际、各seed差值及A7−A0见同目录JSON。全矩阵完成才计算结果；未完成组不插值、不用其他seed补齐。完整消融用于判断交互与否决模块，不保证三个模块均有效。']
    (a.output/'Ablation_Effects.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'status':'PASSED','main_runs':24,'test_accessed':False}))


if __name__=='__main__':main()
