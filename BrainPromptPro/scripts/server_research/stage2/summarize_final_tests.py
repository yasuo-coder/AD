"""Aggregate fixed final evaluation, never selects a model from test scores."""
import csv,json
from pathlib import Path
import numpy as np
r=Path('/home/panyao/AD/BrainPromptPro/outputs/research/final_optimization_20261007');s=r/'stage2_20261008';out=s/'final_results';out.mkdir(exist_ok=True)
keys=['accuracy','f1_macro','precision_macro','recall_macro','roc_auc_macro','f1_weighted','balanced_accuracy'];class_names=['CN','AD','SMC','EMCI','LMCI'];perseed=[];sources=[];perclass=[];entries={};collapse=[]
for base in (r,s):
 for entry in json.loads((base/'selection_lock.json').read_text())['experiments']:
  path=Path(entry['experiment']);x=json.loads((path/'test_result.json').read_text());prefix='' if base==r else 'stage2/';name=prefix+x['model'];seed=x['seed'];v=x['test']
  entries.setdefault(name,{})[seed]=x
  perseed.append(dict(model=name,seed=seed,**{k:v[k] for k in keys},entropy=v['entropy_mean'],confidence=v['confidence_mean'],margin=v['margin_mean'],checkpoint_sha256=x['checkpoint_sha256'],experiment=str(path)))
  collapse.append(dict(model=name,seed=seed,true_distribution=v['true_distribution'],prediction_distribution=v['prediction_distribution'],classification_collapse=v['classification_collapse'],confusion_matrix=v['confusion_matrix']))
  for c in v['per_class']:perclass.append(dict(model=name,seed=seed,class_name=class_names[c['class_index']],**c))
  for source,sv in x['by_source'].items():sources.append(dict(model=name,seed=seed,source=source,n=sv['samples'],class_support=json.dumps([c['support'] for c in sv['per_class']]),**{k:sv[k] for k in keys}))
def save(name,rows):
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
means=[]
for name,values in entries.items():
 assert set(values)=={42,3407,2026}
 row=dict(model=name,n_seeds=3)
 for k in keys:
  v=[values[seed]['test'][k] for seed in [42,3407,2026]];assert all(z is not None for z in v)
  row[k+'_mean']=float(np.mean(v));row[k+'_std']=float(np.std(v,ddof=1))
 means.append(row)
source_means=[]
for name in entries:
 for source in sorted({x['source'] for x in sources}):
  rows=[x for x in sources if x['model']==name and x['source']==source];assert len(rows)==3
  row=dict(model=name,source=source,n=rows[0]['n'],class_support=rows[0]['class_support'])
  for k in keys:
   values=[x[k] for x in rows];available=all(x is not None for x in values)
   row[k+'_mean']=float(np.mean(values)) if available else None;row[k+'_std']=float(np.std(values,ddof=1)) if available else None
  source_means.append(row)
class_means=[]
for name in entries:
 for cls in class_names:
  rows=[x for x in perclass if x['model']==name and x['class_name']==cls];assert len(rows)==3
  row=dict(model=name,class_name=cls,support=rows[0]['support'])
  for k in ['precision','recall','f1','predicted_count','specificity']:
   values=[x[k] for x in rows];row[k+'_mean']=float(np.mean(values));row[k+'_std']=float(np.std(values,ddof=1))
  class_means.append(row)
save('test_three_seed.csv',means);save('test_per_seed.csv',perseed);save('test_by_source_per_seed.csv',sources);save('test_by_source_three_seed.csv',source_means);save('test_per_class_per_seed.csv',perclass);save('test_per_class_three_seed.csv',class_means)
(out/'prediction_distributions.json').write_text(json.dumps(collapse,indent=2)+'\n')
(out/'test_summary.json').write_text(json.dumps(dict(role='Final fixed holdout test; no model selection',sample_count=100,unique_acquisitions_across_seeds=100,seeds=[42,3407,2026],metrics=means,by_source=source_means,per_class=class_means),indent=2)+'\n')
md=['# Final fixed test results','','Three training seeds on the same 100 test acquisitions; mean ± sample SD. F1/P/R are fixed five-class macro, probability AUC is macro OvR. No threshold tuning or reselection after test.','','| Model | Accuracy % | Macro F1 % | Macro Precision % | Macro Recall % | ROC-AUC |','|---|---:|---:|---:|---:|---:|']
for row in means:
 md.append('| '+row['model']+' | '+' | '.join(f"{100*row[k+'_mean']:.2f}±{100*row[k+'_std']:.2f}" if i<4 else f"{row[k+'_mean']:.4f}±{row[k+'_std']:.4f}" for i,k in enumerate(keys[:5]))+' |')
(out/'Final_Test_Results.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
