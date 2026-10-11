"""Seal an audited seven-batch cohort and new sample/group holdouts; no training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from BrainPromptPro.scripts.adni_1to7.audit_dataset import sha,write_json,write_csv
from BrainPromptPro.data.metadata import record_id
from BrainPromptPro.research_runner import validate_sample_split


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--historical-bundle',type=Path,required=True);p.add_argument('--historical-config',type=Path,required=True)
    p.add_argument('--historical-split',type=Path,required=True)
    p.add_argument('--encoder',type=Path,default=Path('/data2/panyao_data/ADNI/ADNI_2/models/Llama-encoder-1.0B'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--split-seed',type=int,default=1729)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    hashes=json.loads((a.audit/'artifact_hashes.json').read_text())
    for name,value in hashes.items():assert sha(a.audit/name)==value
    data=torch.load(a.audit/'audited_fc.private.pt',weights_only=True,map_location='cpu')
    old=torch.load(a.historical_bundle,weights_only=True,map_location='cpu')
    cfg=json.loads(a.historical_config.read_text());recs=data['records'];labels=np.array([r['label'] for r in recs]);idx=np.arange(len(recs))
    rest,test=train_test_split(idx,test_size=.1,random_state=a.split_seed,stratify=labels)
    train,val=train_test_split(rest,test_size=1/9,random_state=a.split_seed,stratify=labels[rest])
    split={k:sorted(v.tolist()) for k,v in dict(train=train,val=val,test=test).items()}
    overlap=validate_sample_split(split,recs,labels)
    groups=np.array([r['subject'] for r in recs])
    folds=[t.tolist() for _,t in StratifiedGroupKFold(10,shuffle=True,random_state=a.split_seed).split(idx,labels,groups)]
    group_split=dict(train=sorted(set(idx.tolist())-set(folds[0])-set(folds[1])),val=sorted(folds[1]),test=sorted(folds[0]))
    assert not any(set(groups[group_split[x]])&set(groups[group_split[y]]) for x,y in [('train','val'),('train','test'),('val','test')])
    counts=lambda sp:{k:dict(samples=len(v),subjects=len(set(groups[v])),classes={c:int(sum(labels[v]==j)) for j,c in enumerate(data['class_names'])}) for k,v in sp.items()}
    oldsplit=json.loads(a.historical_split.read_text())
    oldsets={k:{old['records'][i]['fmri_image_id'] for i in v} for k,v in oldsplit.items()}
    exposure={k:{h:sum(recs[i]['fmri_image_id'] in ss for i in v) for h,ss in oldsets.items()} for k,v in split.items()}
    write_json(a.output/'sample_split.private.json',split);write_json(a.output/'subject_split.private.json',group_split)
    for name,v in split.items():write_csv(a.output/(name+'.private.csv'),[recs[i] for i in v])
    cache={r['prompt']:e for r,e in zip(old['records'],old['subject_prompt'])}
    missing=sorted({r['prompt'] for r in recs}-set(cache));print('New unique metadata prompts:',len(missing),flush=True)
    if missing:
        from transformers import AutoTokenizer
        from llm2vec.models import LlamaBiModel
        tok=AutoTokenizer.from_pretrained(str(a.encoder),local_files_only=True)
        model=LlamaBiModel.from_pretrained(str(a.encoder),local_files_only=True).to(a.device).eval()
        for j,text in enumerate(missing):
            batch={k:v.to(a.device) for k,v in tok(text,return_tensors='pt').items()}
            with torch.no_grad():emb=model(**batch).last_hidden_state.squeeze(0)[:-1].mean(0).float().cpu()
            assert emb.shape==(2048,) and torch.isfinite(emb).all();cache[text]=emb
            if (j+1)%50==0:print('Encoded',j+1,flush=True)
    prompt=torch.stack([cache[r['prompt']] for r in recs]);assert torch.equal(prompt[:len(old['records'])],old['subject_prompt'])
    provenance=dict(label_policy=data['label_policy'],sha256={k:sha(cfg['data'][k]) for k in ['roi_prompt','roi_text','disease_prompt','disease_text']},
        audit_hash=sha(a.audit/'audit_summary.json'),manifest_hash=data['summary']['manifest_hash'],
        preparation_script=sha(__file__),encoder_files={p.name:sha(p) for p in sorted(a.encoder.iterdir()) if p.suffix in ['.json','.safetensors','.model']},
        historical_bundle=sha(a.historical_bundle),clinical_source_authority='Download CSV convention; not independent clinical adjudication')
    cohort=dict(fc=data['fc'],records=recs,class_names=data['class_names'],roi_names=data['roi_names'],label_policy=data['label_policy'],
        subject_prompt=prompt,subject_ids=[record_id(r) for r in recs],provenance=provenance)
    torch.save(cohort,a.output/'cohort.private.pt')
    protocol=dict(dataset_version='ADNI_Combined_1to7_v1',sample_count=len(recs),subject_count=len(set(groups)),class_distribution=data['summary']['class_distribution'],
        label_mapping={c:i for i,c in enumerate(data['class_names'])},preprocessing_version=data['summary']['fc_method'],roi_order=data['summary']['roi_order_sha256'],
        split_seed=a.split_seed,training_seeds=[42,3407,2026],mask_seed_rule='training_seed + 1000003; stored generator state',loader_seed_rule='training_seed + 2000003; stored generator state',
        sample_split=counts(split),subject_overlap=overlap,subject_disjoint_supplement=counts(group_split),
        subject_disjoint_missing_classes={k:[c for c,n in v['classes'].items() if n==0] for k,v in counts(group_split).items()},
        subject_disjoint_rule='StratifiedGroupKFold(10); fold0=test, fold1=val, rest=train; longitudinal labels retained; not exact 80/10/10',
        historical_usage_overlap=exposure,external_unseen_cohort=False,history_caveat='Old1-5 samples participated in development; earlier1-4 CV exposure. Same acquisitions across seeds are not new observations.',
        split_hash=sha(a.output/'sample_split.private.json'),subject_split_hash=sha(a.output/'subject_split.private.json'),manifest_hash=provenance['manifest_hash'],dataset_hash=sha(a.output/'cohort.private.pt'),
        original_audit_hash=provenance['audit_hash'],status='DATA_FROZEN_NO_TRAINING')
    write_json(a.output/'protocol.json',protocol)
    cfg['data'].update(cohort_name='ADNI_Combined_1to7_v1',scope='adni1to7',required_batches=[f'ADNI_{i}' for i in range(1,8)],bundle=str(a.output/'cohort.private.pt'),bundle_sha256=protocol['dataset_hash'])
    cfg['new_research']=dict(dataset_version=protocol['dataset_version'],split_version='sample_1729_v1',split_path=str(a.output/'sample_split.private.json'),split_hash=protocol['split_hash'])
    # Paths to frozen server prompts/baseline remain explicit; no old output is reused.
    cfg['output_dir']=str(a.output.parent/'experiments')
    write_json(a.output/'base_config.json',cfg)
    print(json.dumps({k:protocol[k] for k in ['sample_count','subject_count','sample_split','subject_overlap','historical_usage_overlap','split_hash']}))


if __name__=='__main__':main()
