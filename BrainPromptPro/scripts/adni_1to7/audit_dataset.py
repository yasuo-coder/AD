"""Audit seven ADNI batches. Private manifests stay outside the public repository.

No imaging preprocessing is run. Saved covariance-shrinkage correlations are
checked but explicit Pearson FC is reconstructed from existing cleaned AAL series.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.io import loadmat
import torch

from BrainPromptPro.data.labels import CLASS_NAMES, LABEL_POLICY, normalize_diagnosis
from BrainPromptPro.scripts.prepare_adni_combined import canonical, kind


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path, delimiter=','):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def write_csv(path, values):
    if not values:
        return
    keys=list(dict.fromkeys(k for v in values for k in v))
    with Path(path).open('w', newline='') as f:
        w=csv.DictWriter(f, keys);w.writeheader();w.writerows(values)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, default=Path('/data2/panyao_data/ADNI'))
    ap.add_argument('--historical-bundle', type=Path, required=True)
    ap.add_argument('--historical-config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    out=args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    historical=torch.load(args.historical_bundle, weights_only=True, map_location='cpu')
    cfg=json.loads(args.historical_config.read_text())
    roi_names=historical['roi_names']
    assert len(roi_names)==116 and list(CLASS_NAMES)==historical['class_names']
    old_by_image={r['fmri_image_id']:(i,r) for i,r in enumerate(historical['records'])}
    agpath=args.data_root/'ADNI_2/pipeline_logs/ADNI_metadata.csv'
    aggregate=defaultdict(list)
    for r in rows(agpath):aggregate[r['Image Data ID']].append(r)
    provenance={'audit_script':sha(__file__), 'historical_bundle':sha(args.historical_bundle),
                'historical_config':sha(args.historical_config), 'aggregate_metadata':sha(agpath)}
    records=[];fcs=[];coverage=[];conflicts=[];duplicates=[];batch_summary={}
    image_seen={};scan_seen={};ts_seen={};fc_seen={};hist_match=0
    for n in range(1,8):
        batch=f'ADNI_{n}';base=args.data_root/batch
        csv_files=list((base/'data').glob('*.csv'))
        assert len(csv_files)==1
        raw=rows(csv_files[0]);provenance[batch+'_csv']=sha(csv_files[0])
        functional=[r for r in raw if kind(r)=='bold']
        pairs=defaultdict(list)
        for r in raw:pairs[(canonical(r['Subject']),re.sub('[^A-Za-z0-9]+','',r['Visit']),r['Acq Date'])].append(r)
        visit_counts=Counter((sub,visit) for sub,visit,date in pairs)
        qc_path=base/'pipeline_logs/final_pipeline_summary.tsv'
        qc={(r['subject'],r['session']):r for r in rows(qc_path,'\t')} if qc_path.exists() else {}
        validation_path=base/'pipeline_logs/final_validation.tsv'
        validation={(r['subject'].removeprefix('sub-'),r['session'].removeprefix('ses-')):r for r in rows(validation_path,'\t')} if validation_path.exists() else {}
        for p in [qc_path,validation_path,base/'pipeline_scripts/pipeline_lib.py',base/'pipeline_scripts/ConnectivityMatrices_ADNI.py']:
            if p.exists():provenance[batch+'_'+p.name]=sha(p)
        alignment={}
        if n==2:
            p=base/'BrainPrompt_data/brainprompt_manifest.csv';provenance[batch+'_alignment']=sha(p)
            for r in rows(p):
                if r['matched_image_id'] in alignment:raise ValueError('Ambiguous historical image mapping')
                alignment[r['matched_image_id']]=r
        stats=[];status=Counter();raw_classes=Counter();pair_differences=Counter();used_sessions=set()
        for (subject,visit,date),pair in sorted(pairs.items()):
            fs=[x for x in pair if kind(x)=='bold'];t1=[x for x in pair if kind(x)=='t1']
            if len(fs)!=1 or len(t1)!=1:
                raise ValueError('Ambiguous acquisition pairing; requires explicit audit')
            row=fs[0];image=row['Image Data ID'];label=normalize_diagnosis(row['Group']);raw_classes[label]+=1
            for k in ['Group','Sex','Age','Acq Date']:
                if row[k]!=t1[0][k]:pair_differences[k]+=1
            item=dict(source_batch=batch,subject=subject,visit=visit,image_id=image)
            coverage.append(item)
            if n==2 and image not in alignment:
                item['status']='MISSING_AUDITED_IMAGE_MAPPING';status[item['status']]+=1;continue
            session=alignment[image]['session'] if n==2 else (datetime.strptime(date,'%m/%d/%Y').strftime('%Y%m%d') if visit_counts[(subject,visit)]>1 else visit)
            if n==2:
                assert alignment[image]['subject']==subject and alignment[image]['final_status']=='READY'
            folder=base/'ADNI_connectivity'/f'sub-{subject}'/f'ses-{session}'
            prefix=f'sub-{subject}_ses-{session}_AAL116'
            tsfile=folder/(prefix+'_features_timeseries.mat')
            fcfile=folder/(prefix+'_correlation_matrix.mat')
            labelsfile=folder/(prefix+'_labels.tsv')
            item.update(timeseries_file=str(tsfile),fc_file=str(fcfile))
            if not all(p.exists() for p in [tsfile,fcfile,labelsfile]):
                item.update(status='MISSING_PREPROCESSED_FC',pipeline_error=qc.get((subject,session),{}).get('error',''))
                status[item['status']]+=1;continue
            used_sessions.add((subject,session))
            try:
                if validation and validation.get((subject,session),{}).get('status')!='PASS':raise ValueError('Pipeline QC not PASS')
                if qc and qc.get((subject,session),{}).get('connectivity_status')!='PASS':raise ValueError('Connectivity QC not PASS')
                names=[x['label'] for x in rows(labelsfile,'\t')]
                if names!=roi_names:raise ValueError('ROI order differs')
                ts=np.asarray(loadmat(tsfile)['data'], dtype=np.float32)
                if ts.ndim!=2 or ts.shape[1]!=116 or ts.shape[0]<20 or not np.isfinite(ts).all():raise ValueError('Invalid ROI time series')
                if np.any(ts.std(0)==0):raise ValueError('Constant ROI')
                saved=np.asarray(loadmat(fcfile)['data'])
                for a in [saved]:
                    if a.shape!=(116,116) or not np.isfinite(a).all():raise ValueError('Invalid stored FC')
                    if not np.allclose(a,a.T,atol=1e-6) or not np.allclose(np.diag(a),1,atol=1e-5):raise ValueError('FC symmetry/diagonal failure')
                    if np.max(abs(a))>1.00001 or np.std(a)==0:raise ValueError('FC range/constant failure')
                fc=np.corrcoef(ts.T).astype(np.float32)
                if not np.isfinite(fc).all():raise ValueError('Invalid Pearson FC')
            except (ValueError,KeyError,AssertionError) as e:
                item.update(status='QC_FAIL',reason=str(e));status[item['status']]+=1;continue
            digest=hashlib.sha256(ts.T.copy().tobytes()).hexdigest();fcd=hashlib.sha256(fc.tobytes()).hexdigest()
            scan=(subject,row['Acq Date'],visit)
            if image in image_seen or scan in scan_seen:
                prior=image_seen.get(image,scan_seen.get(scan))
                if prior['fc_sha256']!=fcd or prior['diagnosis']!=label:raise ValueError('Duplicate scan content/label conflict')
                item.update(status='DUPLICATE_SCAN',kept_sample=prior['Sample_ID']);duplicates.append(dict(item));status[item['status']]+=1;continue
            if digest in ts_seen or fcd in fc_seen:raise ValueError('Identical content with distinct acquisition identity; manual review required')
            matches=aggregate.get(image,[])
            if not matches or {canonical(x['Subject']) for x in matches}!={subject}:raise ValueError('Missing/ambiguous exact-image metadata')
            if len({x['site_id'] for x in matches})!=1 or len({x['Group'] for x in matches})!=1:raise ValueError('Conflicting aggregate metadata')
            ag=matches[0]
            if row['Group']!=ag['Group']:
                conflicts.append(dict(source_batch=batch,image_id=image,field='Group',download=row['Group'],aggregate=ag['Group'],target_differs=label!=normalize_diagnosis(ag['Group'])))
            for k in ['Sex','Age','Acq Date','Visit']:
                if row.get(k)!=ag.get(k):conflicts.append(dict(source_batch=batch,image_id=image,field=k,download=row.get(k),aggregate=ag.get(k)))
            rec=dict(graph_index=len(records),source_batch=batch,subject=subject,session=session,raw_visit=visit,
                fmri_image_id=image,t1_image_id=t1[0]['Image Data ID'],label=list(CLASS_NAMES).index(label),diagnosis=label,
                original_diagnosis=row['Group'],label_policy=LABEL_POLICY,clinical_label_source='download',age=float(row['Age']),
                sex=row['Sex'],site_id=str(int(ag['site_id'])),acquisition_date=row['Acq Date'],timepoints=len(ts),
                timeseries_file=str(tsfile),label_metadata_source=str(csv_files[0]),demographic_metadata_source=str(csv_files[0]),site_source=str(agpath),
                Sample_ID=subject+'/'+session,Subject_ID=subject,Biological_Subject_ID=subject,Visit_ID=visit,Scan_ID=image,Source_Dataset=batch,
                fc_sha256=fcd,timeseries_sha256=digest,stored_fc_sha256=sha(fcfile),labels_sha256=sha(labelsfile))
            rec['prompt']=f'The subject is a {int(round(rec["age"]))}-year-old '+{'F':'female','M':'male'}[rec['sex']]+f', with the image collected from study site {rec["site_id"]}.'
            # All historical accepted FC/labels must remain unchanged, independently rechecked.
            if image in old_by_image:
                j,old=old_by_image[image]
                assert torch.equal(torch.from_numpy(fc),historical['fc'][j])
                assert all(rec[k]==old[k] for k in ['label','diagnosis','subject','session','prompt'])
                hist_match+=1
            if rec['Sample_ID'] in {x['Sample_ID'] for x in records}:raise ValueError('Duplicate Sample_ID')
            records.append(rec);fcs.append(torch.from_numpy(fc));image_seen[image]=rec;scan_seen[scan]=rec;ts_seen[digest]=rec;fc_seen[fcd]=rec
            item['status']='INCLUDED';status['INCLUDED']+=1
            upper=fc[np.triu_indices(116,1)]
            stats.append(dict(mean=float(upper.mean()),std=float(upper.std()),negative_fraction=float((upper<0).mean()),stored_pearson_max_abs=float(abs(saved-fc).max()),timepoints=len(ts)))
        selected=[x for x in records if x['source_batch']==batch]
        batch_summary[batch]=dict(raw_image_rows=len(raw),raw_subjects=len({canonical(x['Subject']) for x in raw}),raw_rsfmri_scans=len(functional),
            stored_fc_files=len(list((base/'ADNI_connectivity').rglob('*correlation_matrix.mat'))),included_fc=len(selected),
            valid_subjects=len({x['subject'] for x in selected}),classes=dict(Counter(x['diagnosis'] for x in selected)),raw_functional_classes=dict(raw_classes),
            status=dict(status),multiple_visit_subjects=sum(c>1 for c in Counter(x['subject'] for x in selected).values()),
            pairing_field_disagreements=dict(pair_differences),label_source_conflicts=sum(x['source_batch']==batch and x['field']=='Group' for x in conflicts),
            normalized_label_conflicts=sum(x['source_batch']==batch and x.get('target_differs',False) for x in conflicts),
            distribution={k:dict(mean=float(np.mean([x[k] for x in stats])),min=float(min(x[k] for x in stats)),max=float(max(x[k] for x in stats))) for k in stats[0]} if stats else {})
    write_csv(out/'coverage.private.csv',coverage)
    write_json(out/'interim_summary.json',batch_summary)
    print(json.dumps({'historical_matches':hist_match,'batches':{b:v['status'] for b,v in batch_summary.items()},'qc_reasons':dict(Counter(x.get('reason','') for x in coverage if x['status']=='QC_FAIL'))}))
    assert hist_match==len(historical['records']), 'Historical samples unexpectedly lost/changed'
    summary=dict(dataset_version='ADNI_Combined_1to7_v1',status='AUDITED_NO_SPLIT',sample_count=len(records),subject_count=len({x['subject'] for x in records}),
        class_distribution={c:sum(r['diagnosis']==c for r in records) for c in CLASS_NAMES},batches=batch_summary,
        historical_fc_and_label_matches=hist_match,duplicate_scans_removed=len(duplicates),
        multiple_visit_subjects=sum(c>1 for c in Counter(x['subject'] for x in records).values()),
        mci_merged=sum(x['original_diagnosis']=='MCI' for x in records),label_mapping={c:i for i,c in enumerate(CLASS_NAMES)},
        label_source='Download CSV Group at paired acquisition/visit; conflicts reported, not clinically adjudicated',
        fc_method='Pearson np.corrcoef(float32 cleaned AAL116 time series.T), float32 output; signed, diagonal one',
        stored_fc_caveat='Existing correlation_matrix.mat uses Nilearn covariance shrinkage, differs from empirical Pearson; only existing ROI series reused, no image preprocessing rerun',
        roi_order_sha256=hashlib.sha256('\n'.join(roi_names).encode()).hexdigest(),provenance=provenance)
    write_csv(out/'manifest.private.csv',records);write_csv(out/'coverage.private.csv',coverage);write_csv(out/'conflicts.private.csv',conflicts);write_csv(out/'duplicates.private.csv',duplicates)
    summary['manifest_hash']=sha(out/'manifest.private.csv')
    write_json(out/'audit_summary.json',summary)
    torch.save(dict(fc=torch.stack(fcs),records=records,class_names=list(CLASS_NAMES),roi_names=roi_names,label_policy=LABEL_POLICY,summary=summary),out/'audited_fc.private.pt')
    write_json(out/'artifact_hashes.json',{'audited_fc.private.pt':sha(out/'audited_fc.private.pt'),'audit_summary.json':sha(out/'audit_summary.json')})
    print(json.dumps({k:summary[k] for k in ['dataset_version','sample_count','subject_count','class_distribution','historical_fc_and_label_matches','duplicate_scans_removed']}))


if __name__=='__main__':
    main()
