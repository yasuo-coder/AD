"""Independent audit of saved train/validation evidence; never performs inference on test."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,confusion_matrix
sys.dont_write_bytecode=True;sys.path.insert(0,'/home/panyao/AD')
from BrainPromptPro.research_runner import RUN,load_context,selection_key,validate_sample_split,allowed_spec,evaluate_test,dump
from BrainPromptPro.data.metadata import MetadataEncoder,record_id
from BrainPromptPro.data.brainnet_pro import sha256
from BrainPromptPro.train_TUs_graph_classification_llm import model_inputs
cfg,protocol,data,split=load_context();torch.set_num_threads(2)
expected_encoder=MetadataEncoder().fit([data.records[i] for i in split['train']],cfg['data']['site_column'],cfg['data']['missing_site']).state_dict()
assert not (RUN/'selection_lock.json').exists()
assert not list((RUN/'experiments').glob('**/test_result.json'))
assert not list((RUN/'experiments').glob('**/test_payload.pt'))
try:evaluate_test(RUN/'experiments/nonexistent','cpu')
except FileNotFoundError:pass
else:raise AssertionError('Test entry failed closed-lock protection')
try:allowed_spec('UnreviewedCandidate')
except ValueError:pass
else:raise AssertionError('Unregistered candidate accepted')
# Disjoint scans but the same biological subject across visits are allowed.
r=[dict(subject=str(i//3)) for i in range(6)]
assert validate_sample_split(dict(train=[0,1],val=[2,3],test=[4,5]),r,[0,1,0,1,0,1])=={'train_val':1,'train_test':0,'val_test':1}
try:validate_sample_split(dict(train=[0,1],val=[1,2,3],test=[4,5]),r,[0,1,0,1,0,1])
except ValueError:pass
else:raise AssertionError('Duplicate FC index accepted')
verified=[]
for file in sorted((RUN/'experiments').glob('*/*/*/result.json')):
 out=file.parent;res=json.loads(file.read_text());ecfg=json.loads((out/'config.json').read_text())
 assert res['test']=='NOT_EVALUATED'
 ck=torch.load(out/'best_validation.pt',weights_only=True,map_location='cpu')
 last=torch.load(out/'last.pt',weights_only=True,map_location='cpu')
 hist=[json.loads(s) for s in (out/'history.jsonl').read_text().splitlines()]
 assert len(hist)==res['epochs_completed']==last['epoch']+1
 assert res['best_epoch']==ck['epoch']+1
 assert res['checkpoint_sha256']==sha256(out/'best_validation.pt')
 assert ck['split_sha256']==res['split_sha256']==protocol['split_sha256']
 assert ck['bundle_sha256']==cfg['data']['bundle_sha256']
 assert ck['metadata_encoder']==expected_encoder
 assert ck['training_indices_used']==split['train']
 assert ck['config']==ecfg
 for path,h in res['source_sha256'].items():
  assert sha256(path)==h,path
  archived=out/'source'/Path(path).relative_to('/home/panyao/AD')
  assert sha256(archived)==h,str(archived)
 # Replay scheduler, lexicographic selection, and the FIRST prescribed stop.
 opt=torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))],lr=ecfg['params']['init_lr'],weight_decay=ecfg['params']['weight_decay'])
 scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=25)
 best=None;best_epoch=-1
 for i,h in enumerate(hist):
  assert h['epoch']==i+1
  assert not any('test' in k.lower() for k in h), 'Per-epoch test monitoring forbidden'
  key=selection_key(h['validation'])
  if best is None or key>best:best=key;best_epoch=i
  assert h['best_epoch']==best_epoch+1
  scheduler.step(h['validation']['loss']);assert h['lr']==opt.param_groups[0]['lr']
  reason=None
  if h['lr']<ecfg['params']['min_lr']:reason='lr_floor'
  elif i+1>=ecfg['params']['min_epochs'] and i-best_epoch>=ecfg['params']['early_stopping_patience']:reason='validation_patience'
  elif i+1==ecfg['params']['epochs']:reason='epoch_cap'
  assert h['stop_reason']==reason
  assert (reason is not None)==(i==len(hist)-1)
 assert res['best_epoch']==best_epoch+1
 assert scheduler.state_dict()==last['scheduler']
 assert hist[best_epoch]['validation']==ck['validation']
 weights=ck['class_weights']
 if res['supporting_profile']=='unweighted':assert weights is None
 else:
  counts=torch.bincount(data.labels[split['train']],minlength=5).float();w=counts.pow(-.5);w/=w.mean()
  torch.testing.assert_close(w,weights,atol=0,rtol=0)
 for role in ['train','val']:
  rows=list(csv.DictReader((out/(role+'_predictions.csv')).open()))
  assert [r['Sample_ID'] for r in rows]==[record_id(data.records[i]) for i in split[role]]
  y=np.array([int(r['true_label']) for r in rows]);p=np.array([[float(r['p_'+c]) for c in data.class_names] for r in rows]);pred=p.argmax(1)
  np.testing.assert_array_equal(y,data.labels[split[role]].numpy());np.testing.assert_allclose(p.sum(1),1,atol=2e-7)
  assert all(int(r['predicted_label'])==k for r,k in zip(rows,pred))
  pr,re,f1,_=precision_recall_fscore_support(y,pred,labels=range(5),average='macro',zero_division=0)
  metric=dict(accuracy=accuracy_score(y,pred),precision_macro=pr,recall_macro=re,f1_macro=f1,roc_auc_macro=roc_auc_score(y,p,multi_class='ovr',average='macro',labels=range(5)))
  for k,v in metric.items():assert abs(v-res[role][k])<1e-10,(file,role,k)
  assert confusion_matrix(y,pred,labels=range(5)).tolist()==res[role]['confusion_matrix']
  assert np.bincount(pred,minlength=5).tolist()==res[role]['prediction_distribution']
 verified.append(dict(model=res['model'],profile=res['supporting_profile'],seed=res['seed'],epochs=res['epochs_completed'],best_epoch=res['best_epoch'],checkpoint_sha256=res['checkpoint_sha256']))
# Snapshot helper functions also used indirectly, beyond the model-class snapshot.
extra={str(p):sha256(p) for p in [Path('/home/panyao/AD/BrainPromptPro/layers/gcn_layer.py'),Path('/home/panyao/AD/BrainPromptPro/runtime.py'),Path('/home/panyao/AD/BrainPromptPro/nets/load_net.py')]}
summary=dict(passed=True,experiments_verified=len(verified),experiments=verified,independent_probability_metrics=True,train_only_preprocessing_and_class_weights=True,checkpoint_selection_scheduler_and_first_stop_replayed=True,source_hashes_unchanged=True,test_lock_closed=True,test_outputs_absent=True,scan_overlap_rejected_and_biological_overlap_permitted=True,novelty_unregistered_candidate_rejected=True,supplementary_helper_source_sha256=extra,script_sha256=sha256(__file__))
dump(RUN/'verification/development_verification.json',summary)
print(json.dumps(summary,indent=2))
