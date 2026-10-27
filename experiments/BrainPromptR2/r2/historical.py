"""Read archived predictions/history only. Never select new recipes from TEST."""
import csv,json,sys,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/panyao/AD/BrainPromptNComms2025')
STUDY=OLD/'out/adni7_ncomms_subject10_20261021'
sys.path.insert(0,str(OLD))
from bpstudy.report import replay_history,verify_selection
from bpstudy.data import aggregate
from bpstudy.metrics import compute

def load(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def scale(p):
 p=np.asarray(p,dtype=float);lp=np.log(np.maximum(p,1e-300));centered=lp-lp.mean(1,keepdims=True)
 return dict(probability_min=float(p.min()),probability_max=float(p.max()),
  max_probability_mean=float(p.max(1).mean()),entropy_mean=float(-(p*lp).sum(1).mean()),
  centered_log_probability_min=float(centered.min()),centered_log_probability_max=float(centered.max()),
  centered_log_probability_std=float(centered.std()),raw_logits='NA: not archived; log(p) is only shift-invariant equivalent',
  mean_probability_by_class=p.mean(0).tolist(),std_probability_by_class=p.std(0).tolist())
def main():
 out=ROOT/'public/diagnostics';out.mkdir(parents=True,exist_ok=True)
 data=torch.load(STUDY/'cohort.private.pt',weights_only=True,map_location='cpu')
 plans=load(STUDY/'splits.private.json');summary=load(STUDY/'results/summary.json')
 rows=[];candidates=[];e4=[];chosen=Counter();comparisons=[]
 for seed in [42,2024,3407]:
  buckets=defaultdict(list)
  for fold,plan in enumerate(plans[str(seed)]):
   unit=STUDY/'units'/f'seed_{seed}'/f'fold_{fold:02d}';result=load(unit/'result.json')
   payloads=torch.load(unit/'predictions.private.pt',weights_only=True,map_location='cpu')
   verify_selection(result,payloads,data,plan)
   for phase in ['E2','E3']:
    name=phase+'_selected';v=result['variants'][name];p=payloads[name]
    choice=result['selections'][phase];chosen[phase+':'+choice]+=1
    vy,vp,_=aggregate(data,plan['stop'],p['validation_probabilities'].numpy())
    vm=compute(vy,vp);tm=v['metrics']['subject']
    record=dict(seed=seed,outer_fold=fold,phase=phase,selected=choice,
      validation_auroc=vm['macro_auroc_ovr'],validation_ap=vm['macro_pr_auc'],validation_nll=vm['nll'],validation_macro_nll=vm['macro_nll'],
      archived_test_auroc=tm['macro_auroc_ovr'],archived_test_ap=tm['macro_pr_auc'],archived_test_nll=tm['nll'],
      archived_test_macro_nll=tm['macro_nll'],archived_test_subjects=tm['samples'],temperature=float(p['temperature']))
    for level,prob in [('validation',vp),('archived_test',p['subject_probabilities'].numpy())]:
     for k,value in scale(prob).items():record[level+'_'+k]=json.dumps(value) if isinstance(value,list) else value
    rows.append(record);buckets[phase].append(record)
    for cand,s in result['selection_scores'][phase].items():
     candidates.append(dict(seed=seed,outer_fold=fold,phase=phase,candidate=cand,selected=cand==choice,
        validation_auroc=s['macro_auroc_ovr'],validation_ap=s['macro_pr_auc'],validation_nll=s['nll'],validation_macro_nll=s['macro_nll']))
   for name in ['E4','E5_stage1','E5_no_fc_bias_stage1','E5_no_ordinal_stage1','E5_no_prompt_dropout_stage1']:
    d=unit/'components'/name;h=load(d/'history.json');fit=load(d/'result.json');replay=replay_history(h,fit['config'],transfer=fit['transfer'] is not None)
    assert replay['stopped_epoch']==fit['stopped_epoch'] and replay['best_epoch']==fit['best_epoch']
    floor=next(x['epoch'] for x in h if x['lr']<=fit['config']['min_lr']*(1+1e-8));last=h[-1];first=h[0]
    e4.append(dict(seed=seed,outer_fold=fold,component=name,best_epoch=fit['best_epoch'],stopped_epoch=fit['stopped_epoch'],
     first_floor_epoch=floor,minimum_possible_stop_epoch=floor+fit['config']['stop_patience']-1,
     floor_epochs=last['control']['floor_epochs'],auc_bad=last['control']['auc_bad'],nll_bad=last['control']['nll_bad'],
     first_train_loss=first['train_loss'],last_train_loss=last['train_loss'],first_validation_auc=first['val_auc'],
     last_validation_auc=last['val_auc'],first_validation_nll=first['val_nll'],last_validation_nll=last['val_nll'],
     last_predicted_classes=sum(x>0 for x in last['validation']['predicted_counts']) if 'predicted_counts' in last['validation'] else None,best_rule='validation AUROC, tie NLL',
     first_stop_replay=True))
  for phase,rs in buckets.items():
   oof=summary['per_seed'][str(seed)][phase+'_selected']['subject']
   weights=np.array([x['archived_test_subjects'] for x in rs]);record=dict(seed=seed,phase=phase)
   for short,key in [('auroc','macro_auroc_ovr'),('ap','macro_pr_auc'),('nll','nll'),('macro_nll','macro_nll')]:
    values=[x['archived_test_'+short] for x in rs]
    record.update({short+'_fold_mean':float(np.mean(values)),short+'_subject_weighted_fold_mean':float(np.average(values,weights=weights)),
      short+'_pooled_oof':oof[key],short+'_pooled_minus_fold_mean':float(oof[key]-np.mean(values))})
   comparisons.append(record)
 csvwrite(out/'E2_E3_selected_per_fold.csv',rows);csvwrite(out/'E2_E3_validation_candidates.csv',candidates)
 csvwrite(out/'E2_E3_fold_vs_oof.csv',comparisons);csvwrite(out/'E4_epoch84_history.csv',e4)
 e4only=[x for x in e4 if x['component']=='E4']
 info=dict(status='PASSED_READ_ONLY',source_study=str(STUDY),selection_replayed=30,selection_counts=dict(chosen),
  archived_test_used_for_new_selection=False,new_model_inference=False,
  raw_logits_available=False,scale_proxy='centered log(probability), not original logits',
  scale_interpretation='Pooled AUC includes cross-fold pairs; fold-average AUC does not. Differences plus scale dispersion are descriptive and do not identify a unique cause.',
  fold_vs_oof=comparisons,e4=dict(all_30_stop_at_84=all(x['stopped_epoch']==84 for x in e4only),
  all_30_first_floor_at_55=all(x['first_floor_epoch']==55 for x in e4only),
  all_30_earliest_allowed_stop=all(x['stopped_epoch']==x['minimum_possible_stop_epoch'] for x in e4only),
  all_30_first_stop_replayed=True,mean_last_validation_auroc=float(np.mean([x['last_validation_auc'] for x in e4only])),
  last_validation_single_class_folds=sum(x['last_predicted_classes']==1 for x in e4only),
  interpretation='Warmup5 + cosine50 reaches lr floor at epoch55; 30 consecutive floor epochs permits first stop84. Plateau conditions actually hold; 84 is not an explicit epoch cap. Optimization/information-flow validity still requires E4-specific FIT tests.'))
 write(out/'historical_diagnostics.public.json',info)
 print(json.dumps(info,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
