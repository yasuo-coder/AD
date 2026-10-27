"""Trusted preparation only; training receives a physically FIT-only view."""
from collections import Counter
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from .common import *
def main():
 source=OLD/'out/adni7_ncomms_subject10_20261021';output=ROOT/'prepared'
 output.mkdir(exist_ok=True)
 if (output/'manifest.json').exists():
  m=read(output/'manifest.json');assert digest(output/'data.private.pt')==m['data_sha256'];return
 source_data=torch.load(source/'cohort.private.pt',weights_only=True,map_location='cpu')
 split=read(source/'splits.private.json')['42'][0];ids=split['fit'];groups=np.asarray(source_data['groups'])
 assert not(set(groups[ids])&set(groups[split['stop']])) and not(set(groups[ids])&set(groups[split['test']]))
 unique=sorted(set(groups[ids]));codes={g:n for n,g in enumerate(unique)}
 # No identifiers, original row indices, outer VAL or TEST observations in worker view.
 data={k:source_data[k][ids].clone() for k in ['fc','subject_prompt','y']}
 data.update(semantic=source_data['semantic'].clone(),disease=source_data['disease'].clone(),groups=[codes[g] for g in groups[ids]])
 records=[source_data['records'][i] for i in ids]
 sites=sorted({str(r['site_id']) for r in records});site_map={s:i for i,s in enumerate(sites)}
 data['metadata']=torch.tensor([[float(r['age']),float(str(r['sex']).upper()=='M'),site_map[str(r['site_id'])]] for r in records],dtype=torch.float32)
 save(output/'data.private.pt',data)
 sg=np.array(sorted(set(data['groups'])));gy=np.array([int(data['y'][data['groups'].index(int(g))]) for g in sg])
 expand=lambda gs:[i for i,g in enumerate(data['groups']) if g in set(gs)]
 plans=[]
 for rep,seed in enumerate([42,2024,3407]):
  splitter=StratifiedKFold(3,shuffle=True,random_state=seed)
  for fold,(fit,val) in enumerate(splitter.split(sg,gy)):
   fit2,stop=next(StratifiedShuffleSplit(1,test_size=.15,random_state=seed+1009+fold).split(sg[fit],gy[fit]))
   parts=dict(train=expand(sg[fit][fit2]),stop=expand(sg[fit][stop]),validation=expand(sg[val]))
   assert all(set(int(data['y'][i]) for i in x)==set(range(5)) for x in parts.values())
   assert len(set(sum(parts.values(),[])))==len(data['y'])
   plans.append(dict(repeat=rep,seed=seed,fold=fold,**parts))
 write(output/'plans.private.json',plans)
 public_plans=[]
 for p in plans:
  counts={}
  for part in ['train','stop','validation']:
   labs={data['groups'][i]:int(data['y'][i]) for i in p[part]}
   counts[part]={'subjects':len(labs),'scans':len(p[part]),'class_subject_counts':np.bincount(list(labs.values()),minlength=5).tolist()}
  public_plans.append({k:p[k] for k in ['repeat','seed','fold']}|{'partitions':counts})
 m=dict(status='PREPARED',outer_seed=42,outer_fold=0,development_source='outer FIT only; original VAL and TEST excluded',
  subjects=len(unique),scans=len(ids),subject_class_counts=np.bincount(gy,minlength=5).tolist(),
  data_sha256=digest(output/'data.private.pt'),plans_sha256=digest(output/'plans.private.json'),
  original_source_sha256=digest(source/'cohort.private.pt'),original_split_sha256=digest(source/'splits.private.json'),
  partition_audit=public_plans,label_policy='original_diagnosis MCI -> EMCI; true LMCI preserved',
  patient_metadata='structured age/sex/site; no diagnostic label used as input',
  outer_val_usage='none; held aside',outer_test_usage='none; evaluator absent')
 write(output/'manifest.json',m);write(ROOT/'public/data_isolation.public.json',m)
 print(json.dumps({k:m[k] for k in ['subjects','scans','subject_class_counts']},ensure_ascii=False))
if __name__=='__main__':main()
