import csv,hashlib,json,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from scipy.stats import chi2_contingency
sys.path.insert(0,'/home/panyao/AD')
from BrainPromptPro.data.brainnet_pro import sha256
from BrainPromptPro.runtime import ROOT
from BrainPromptPro.data.labels import normalize_diagnosis,CLASS_NAMES

out=ROOT/'outputs/research/final_optimization_20261007/audits/data'
out.mkdir(parents=True,exist_ok=True)
bundle=ROOT/'outputs/data/adni12345_download_mci_to_lmci_v1_20261007/cohort.pt'
v=torch.load(bundle,weights_only=True,map_location='cpu');records=v['records'];fc=v['fc'].numpy()
def dump(name,value):
    (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write(name,rows):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[];raw_counts={};metadata_missing=[]
for source in [f'ADNI_{i}' for i in range(1,6)]+['Combined']:
    idx=[i for i,r in enumerate(records) if source=='Combined' or r['source_batch']==source]
    rr=[records[i] for i in idx];ages=np.array([r['age'] for r in rr]);timepoints=[r['timepoints'] for r in rr]
    subject_counts=Counter(r['subject'] for r in rr);counts=Counter(r['diagnosis'] for r in rr)
    missing={k:sum(str(r.get(k,'')).strip().lower() in ('','nan','none','unknown') for r in rr) for k in ('subject','raw_visit','fmri_image_id','original_diagnosis','age','sex','site_id')}
    stats=dict(Source=source,FC_Samples=len(idx),Unique_Subjects=len(subject_counts),Multiple_Visit_Subjects=sum(c>1 for c in subject_counts.values()),
               **{c:counts[c] for c in CLASS_NAMES},AD_Percent=100*counts['AD']/len(idx),age_mean=float(ages.mean()),age_std=float(ages.std()),
               age_min=float(ages.min()),age_max=float(ages.max()),Female=sum(r['sex']=='F' for r in rr),Male=sum(r['sex']=='M' for r in rr),
               Sites=len({r['site_id'] for r in rr}),Visit_Labels=len({r['raw_visit'] for r in rr}),Missing_Metadata=sum(missing.values()),
               Invalid_FC=sum(not np.isfinite(fc[i]).all() or fc[i].shape!=(116,116) or not np.allclose(fc[i],fc[i].T,atol=1e-6) or not np.allclose(np.diag(fc[i]),1,atol=1e-6) for i in idx),
               timepoints_min=min(timepoints),timepoints_max=max(timepoints))
    summary.append(stats);metadata_missing.append(dict(source=source,**missing))
    raw_counts[source]=dict(diagnosis=dict(Counter(r['original_diagnosis'] for r in rr)),visits=dict(Counter(r['raw_visit'] for r in rr)),sites=dict(Counter(r['site_id'] for r in rr)))
write('dataset_summary.csv',summary);write('missing_metadata.csv',metadata_missing);dump('distributions.json',raw_counts)
mapping=[]
for raw in sorted({r['original_diagnosis'] for r in records}):
    norm=normalize_diagnosis(raw);k=CLASS_NAMES.index(norm)
    mapping.append(dict(raw_diagnosis=raw,metadata_label=raw,unified_diagnosis=norm,dataset_label=k,glabel_equivalent=k,class_index=k,disease_prompt_row=k,classifier_output_column=k))
write('label_mapping.csv',mapping)
fc_hashes=defaultdict(list)
for i,r in enumerate(records):fc_hashes[hashlib.sha256(fc[i].tobytes()).hexdigest()].append(r['Sample_ID'])
duplicate_fc={h:ids for h,ids in fc_hashes.items() if len(ids)>1}
confounds={}
for a,b in [('source_batch','site_id'),('source_batch','diagnosis'),('site_id','diagnosis')]:
    aa=sorted({r[a] for r in records});bb=sorted({r[b] for r in records});table=np.zeros((len(aa),len(bb)),int)
    for r in records:table[aa.index(r[a]),bb.index(r[b])]+=1
    chi,p,dof,expected=chi2_contingency(table)
    cv=float(np.sqrt(chi/(len(records)*min(len(aa)-1,len(bb)-1))))
    rows=[dict(row_name=x,**{y:int(table[i,j]) for j,y in enumerate(bb)}) for i,x in enumerate(aa)]
    write(a+'_x_'+b+'.csv',rows)
    confounds[a+'_x_'+b]=dict(cramers_v=cv,chi2=float(chi),p_value=float(p),dof=int(dof),expected_cells_below_5=int((expected<5).sum()),
                               inference='Descriptive association; sparse expected cells limit chi-square interpretation; not proof of causal confounding')
prefix_map=defaultdict(set);site_map=defaultdict(set)
for r in records:prefix_map[r['subject'][:3]].add(r['site_id']);site_map[r['site_id']].add(r['subject'][:3])
off=~np.eye(116,dtype=bool)
audit=dict(bundle_sha256=sha256(bundle),samples=len(records),unique_biological_subjects=len({r['subject'] for r in records}),
           exact_duplicate_fc=duplicate_fc,age_finite=bool(np.isfinite([r['age'] for r in records]).all()),
           all_fc_finite=bool(np.isfinite(fc).all()),fc_min=float(fc.min()),fc_max=float(fc.max()),max_asymmetry=float(np.abs(fc-fc.transpose(0,2,1)).max()),
           offdiagonal_negative_fraction=float((fc[:,off]<0).mean()),offdiagonal_abs_mean=float(np.abs(fc[:,off]).mean()),
           site_to_subject_prefix={s:sorted(p) for s,p in site_map.items()},subject_prefix_to_site={p:sorted(s) for p,s in prefix_map.items()},
           site_provenance='Existing aggregate metadata site_id joined by exact Image Data ID; not the ADNI_1--5 storage partition. Scanner/institution identity is not independently adjudicated here.',
           associations=confounds,summary=summary)
dump('extended_audit.json',audit)
print(json.dumps(dict(summary=summary,associations=confounds,duplicate_fc_groups=len(duplicate_fc)),ensure_ascii=False))
