import csv,json
from pathlib import Path
import numpy as np
r=Path('/home/panyao/AD/BrainPromptPro/outputs/research/final_optimization_20261007');s=r/'stage2_20261008';keys=['accuracy','f1_macro','precision_macro','recall_macro','roc_auc_macro'];seeds=[42,3407,2026];entries={}
for base,prefix in [(r,''),(s,'stage2/')]:
 for p in sorted((base/'experiments').glob('*/sqrt_inverse_frequency/seed_*/result.json')):
  x=json.loads(p.read_text());entries.setdefault(prefix+x['model'],{})[x['seed']]=(x,p)
means={};sd={};perseed=[]
for name,values in entries.items():
 for seed,(x,p) in values.items():perseed.append(dict(model=name,seed=seed,**{k:x['val'][k] for k in keys},best_epoch=x['best_epoch'],epochs_completed=x['epochs_completed'],elapsed_seconds=x['elapsed_seconds'],parameter_count=x['parameter_count'],result_path=str(p)))
 if set(values)!=set(seeds):continue
 v=np.array([[values[z][0]['val'][k] for k in keys] for z in seeds]);means[name]=v.mean(0);sd[name]=v.std(0,ddof=1)
rows=[]
for name,m in means.items():
 row=dict(model=name,n_seeds=3)
 for i,k in enumerate(keys):row[k+'_mean']=float(m[i]);row[k+'_std']=float(sd[name][i])
 for ref,label in [('BrainPrompt','primary_BP'),('stage2/BP_control','contemporary_control')]:
  if ref not in means:continue
  d=m-means[ref]
  for i,k in enumerate(keys):row[k+'_delta_'+label]=float(d[i])
  n=int((d>1e-12).sum());row['rank_'+label]='Strong' if n==5 else 'Acceptable' if n>=4 and d.min()>=-.01 else 'Weak' if n<=2 and d.min()<-.01 else 'Mixed'
 rows.append(row)
out=s/'summary';out.mkdir(exist_ok=True)
def write(name,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for x in rows for k in x))
 with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
write('validation_three_seed.csv',rows);write('validation_per_seed.csv',perseed)
state=dict(role='VALIDATION ONLY, not final test',required_seeds=seeds,complete_model_count=len(means),pending={k:sorted(set(seeds)-set(v)) for k,v in entries.items() if set(v)!=set(seeds)},models=rows)
(out/'validation_summary.json').write_text(json.dumps(state,indent=2)+'\n')
md=['# 三随机种子验证集汇总','','检查点只由验证集选择；均值±样本标准差，seed=42/3407/2026；测试集未参与。仅三seed齐全的模型进入均值表。','','| Model | Accuracy % | Macro F1 % | Macro Precision % | Macro Recall % | ROC-AUC |','|---|---:|---:|---:|---:|---:|']
for name,m in means.items():md.append('| '+name+' | '+' | '.join(f'{100*m[i]:.2f}±{100*sd[name][i]:.2f}' if i<4 else f'{m[i]:.4f}±{sd[name][i]:.4f}' for i in range(5))+' |')
md+=['','原BrainPrompt是事先固定的主基线；BP_control是同架构重复运行敏感性对照，两者均保留，不选择更低者作为唯一比较基准。','','尚未齐三seed：'+json.dumps(state['pending'],ensure_ascii=False)]
(out/'Validation_Results.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
