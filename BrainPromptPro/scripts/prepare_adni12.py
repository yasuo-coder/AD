#!/usr/bin/env python3
"""Prepare audited ADNI_1+2 FC and uniform, ID-bearing subject embeddings."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scipy.io import loadmat
from BrainPromptPro.data.brainnet_pro import read_csv, sha256
from BrainPromptPro.data.metadata import record_id
from BrainPromptPro.data.labels import LABEL_POLICY, CLASS_NAMES, validate_record_label
from BrainPromptPro.data.splits import make_holdout
from BrainPromptPro.runtime import ROOT, output_path, write_json


def metadata_site_lookup(rows):
    lookup = defaultdict(set)
    for row in rows:
        match = re.fullmatch(r'(\d{3})_S_(\d+)(?:_\d+)?',row['Subject'])
        if not match:
            raise ValueError(f'Unrecognized metadata subject ID: {row["Subject"]}')
        lookup[match[1]+'S'+match[2]].add(str(int(row['site_id'])))
    return lookup


def make_prompt(row):
    sex = {'F':'female','M':'male'}[row['sex']]
    return (f'The subject is a {int(round(float(row["age"])))}-year-old {sex}, '
            f'with the image collected from study site {row["site_id"]}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:1')
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this environment')
    config = json.loads((ROOT/'configs/smoke/BrainPromptPro_adni2_smoke.json').read_text())
    sources = Path('/data2/panyao_data/ADNI')
    output = output_path('data','adni12_'+LABEL_POLICY)
    output.mkdir(exist_ok=False)
    hashes, files, records, fcs, exclusions, raw_hashes = {}, {}, [], [], [], {}
    def pin(key, path):
        files[key] = str(path); hashes[key] = sha256(path)
    metadata = sources/'ADNI_2/pipeline_logs/ADNI_metadata.csv'
    pin('site_metadata',metadata)
    sites = metadata_site_lookup(read_csv(metadata))
    roi_names = [line.split(':',1)[0] for line in Path(config['data']['roi_text']).read_text().splitlines()]
    classes = config['data']['class_names']
    if classes != list(CLASS_NAMES):
        raise ValueError('ADNI class order mismatch')
    pin('label_policy_source', ROOT/'data/labels.py')
    for key in ('roi_prompt','roi_text','disease_prompt','disease_text'):
        pin(key,config['data'][key])
    disease_lines = Path(config['data']['disease_text']).read_text().splitlines()
    assert [re.search(r'network of an? (\w+) subject',s).group(1) for s in disease_lines] == classes
    import dgl
    for name, graph_name, excluded_name in (
        ('ADNI_1','adni_ADNI1_AAL116.bin','brainprompt_excluded.csv'),
        ('ADNI_2','adni_custom_AAL116.bin','brainprompt_manifest_excluded.csv')):
        directory = sources/name/'BrainPrompt_data'
        for key,file in [('manifest','brainprompt_manifest.csv'),('graph',graph_name),('excluded',excluded_name)]:
            pin(name+'_'+key,directory/file)
        rows = sorted(read_csv(directory/'brainprompt_manifest.csv'),key=lambda r:int(r['graph_index']))
        graphs, labels = dgl.load_graphs(str(directory/graph_name))
        if len(rows)!=len(graphs) or [int(r['graph_index']) for r in rows]!=list(range(len(rows))):
            raise ValueError('Source graph count or order mismatch')
        for i,(row,graph) in enumerate(zip(rows,graphs)):
            validate_record_label(row)
            identity = record_id(row)
            if row.get('final_status','READY') != 'READY':
                raise ValueError(f'Unresolved metadata: {identity}')
            site = sites.get(row['subject'],set())
            if len(site)!=1:
                raise ValueError(f'Missing or ambiguous site: {identity}')
            site = next(iter(site))
            if row.get('site_id') and row['site_id']!=site:
                raise ValueError(f'Source site disagreement: {identity}')
            label = int(row['label'])
            if row['diagnosis']!=classes[label] or int(labels['glabel'][i])!=label:
                raise ValueError(f'Source graph label mismatch: {identity}')
            pin(name+'_'+identity+'_timeseries',row['timeseries_file'])
            raw = np.asarray(loadmat(row['timeseries_file'])['data'],dtype=np.float32).T.copy()
            if raw.shape[0]!=116 or not np.isfinite(raw).all() or (raw.std(-1)==0).any():
                raise ValueError(f'Invalid time series: {identity}')
            if not np.array_equal(raw,graph.ndata['N_features'].numpy()):
                raise ValueError(f'Subject/time series mismatch in graph: {identity}')
            import hashlib
            digest = hashlib.sha256(raw.tobytes()).hexdigest()
            if digest in raw_hashes:
                raise ValueError(f'Duplicate time series: {identity}, {raw_hashes[digest]}')
            raw_hashes[digest] = identity
            roi_file = Path(row['timeseries_file'].replace('_features_timeseries.mat','_labels.tsv'))
            pin(name+'_'+identity+'_roi_labels',roi_file)
            with roi_file.open() as stream:
                labels_roi = list(csv.DictReader(stream,delimiter='\t'))
            if [r['label'] for r in labels_roi] != roi_names:
                raise ValueError(f'AAL116 ROI order differs from prompts: {identity}')
            fc = np.corrcoef(raw).astype(np.float32)
            if not np.isfinite(fc).all(): raise ValueError(f'Nonfinite FC: {identity}')
            fcs.append(torch.from_numpy(fc))
            prepared = dict(graph_index=len(records),source_graph_index=i,source_batch=name,
                subject=row['subject'],session=row['session'],label=label,diagnosis=row['diagnosis'],
                original_diagnosis=row['original_diagnosis'],label_policy=LABEL_POLICY,
                age=float(row['age']),sex=row['sex'],site_id=site,timeseries_file=row['timeseries_file'],
                site_source=str(metadata),clinical_metadata_source=str(directory/'brainprompt_manifest.csv'))
            prepared['prompt'] = make_prompt(prepared)
            records.append(prepared)
        for row in read_csv(directory/excluded_name):
            exclusions.append(dict(source_batch=name,subject=row['subject'],session=row['session'],
                diagnosis=row['diagnosis'],original_diagnosis=row.get('original_diagnosis',row['diagnosis']),
                label_policy=LABEL_POLICY,reason=row.get('reason',row.get('final_status','excluded'))))
    ids = [record_id(r) for r in records]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate scan ID across batches')
    split = make_holdout(records,[r['label'] for r in records],seed=41)
    write_json(output/'split_preview.json',{name:dict(samples=len(idx),
        subjects=len({records[i]['subject'] for i in idx}),
        class_counts=dict(Counter(records[i]['diagnosis'] for i in idx))) for name,idx in split.items()})
    print('Verified FC, ROI order, metadata and split:',(output/'split_preview.json').read_text(),flush=True)
    from transformers import AutoTokenizer
    from llm2vec.models import LlamaBiModel
    model_path = sources/'ADNI_2/models/Llama-encoder-1.0B'
    for path in sorted(model_path.iterdir()):
        if path.is_file() and (path.suffix in ('.json','.safetensors','.model')):
            pin('encoder_'+path.name,path)
    pin('preparation_script',Path(__file__))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
    model = LlamaBiModel.from_pretrained(str(model_path),local_files_only=True).to(device).eval()
    embeddings = []
    for i,row in enumerate(records):
        inputs = {k:v.to(device) for k,v in tokenizer(row['prompt'],return_tensors='pt').items()}
        with torch.no_grad(): hidden = model(**inputs).last_hidden_state.squeeze(0)
        embedding = hidden[:-1].mean(0).detach().cpu().float()
        if embedding.shape!=(2048,) or not torch.isfinite(embedding).all():
            raise ValueError('Invalid subject embedding')
        embeddings.append(embedding)
        if (i+1)%50==0: print(f'Encoded {i+1}/{len(records)} metadata prompts',flush=True)
    provenance = dict(label_policy=LABEL_POLICY,sha256=hashes,source_files=files,alignment='ID-bearing bundle from audited source graphs and manifests',
        graph_timeseries_content_verified=True,roi_name_order_verified=True,
        full_fc_source='np.corrcoef(N_features)',
        subject_embedding_generation='local frozen LlamaBiModel, last_hidden_state[:-1].mean(0), uniform prompt, no labels',
        site_alignment='exact canonical subject ID; all metadata visits must agree on site',
        clinical_metadata_policy='age/sex/diagnosis from each curated source manifest; external metadata supplies site only',
        excluded_scans=exclusions,scope='adni12_subset')
    # Detect external source changes during preparation, including the encoder.
    for key,path in files.items():
        if sha256(path)!=hashes[key]: raise ValueError(f'Source changed during preparation: {key}')
    bundle=output/'cohort.pt'
    torch.save(dict(label_policy=LABEL_POLICY,records=records,subject_ids=ids,fc=torch.stack(fcs),subject_prompt=torch.stack(embeddings),
                    class_names=classes,roi_names=roi_names,provenance=provenance),bundle)
    for name,rows in [('manifest.csv',records),('excluded.csv',exclusions)]:
        with (output/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['source_batch','subject','session','diagnosis','original_diagnosis','label_policy','reason']);writer.writeheader();writer.writerows(rows)
    write_json(output/'provenance.json',provenance)
    config['data'] = dict(format='prepared_cohort_v1',cohort_name='ADNI_1+ADNI_2',scope='adni12_subset',
        required_batches=['ADNI_1','ADNI_2'],label_policy=LABEL_POLICY,bundle='@pro/outputs/data/adni12_'+LABEL_POLICY+'/cohort.pt',bundle_sha256=sha256(bundle),
        node_num=116,class_names=classes,site_column='site_id',missing_site='error',split_mode='grouped_holdout',
        index_dir=None,**{k:config['data'][k] for k in ('roi_prompt','roi_text','disease_prompt','disease_text')})
    config['params'].update(device=args.device,epochs=200,early_stopping_patience=30,
        batch_size=20,n_splits=1,threads=2,seed=41)
    destination=ROOT/'configs/adni12/BrainPromptPro_adni12_holdout.json'
    destination.parent.mkdir(exist_ok=True)
    write_json(destination,config)
    print(json.dumps(dict(config=str(destination),samples=len(records),subjects=len({r['subject'] for r in records}),
                         excluded=len(exclusions),bundle_sha256=config['data']['bundle_sha256'])),flush=True)


if __name__=='__main__': main()
