#!/usr/bin/env python3
"""Prepare a new ADNI_1+2+3+4+5 experiment from download Image IDs and AAL116 FC.

Clinical fields use one declared source throughout. Historical batches and
their earlier experiments are never overwritten. Missing original images and
out-of-list historical replacements are explicitly recorded in coverage.csv.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scipy.io import loadmat
from BrainPromptPro.data.brainnet_pro import sha256
from BrainPromptPro.data.labels import LABEL_POLICY, CLASS_NAMES, normalize_diagnosis, validate_record_label
from BrainPromptPro.data.metadata import record_id
from BrainPromptPro.data.splits import make_holdout
from BrainPromptPro.runtime import ROOT, write_json
from BrainPromptPro.data.private_audit import expected_ids


def read(path, delimiter=','):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def csv_out(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def canonical(value):
    match = re.fullmatch(r'(\d{3})_S_(\d+)(?:_\d+)?', value)
    if not match:
        raise ValueError(f'Unrecognized subject ID: {value}')
    return match[1] + 'S' + match[2]


def kind(row):
    desc = re.sub(r'\s+', '', row['Description'].lower())
    if any(t in desc for t in ('rsfmri', 'fcmri', 'resting')):
        return 'bold'
    if any(t in desc for t in ('mprage', 'spgr', 'fspgr')):
        return 't1'
    raise ValueError(f'Unrecognized image description: {desc}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:4')
    parser.add_argument('--label-source', choices=('download', 'aggregate'), default='download')
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    base = Path('/data2/panyao_data/ADNI')
    name = f'adni12345_{args.label_source}_{LABEL_POLICY}_20261007'
    out = ROOT/'outputs/data'/name
    config_path = ROOT/'configs/adni_combined'/f'BrainPromptPro_adni12345_{args.label_source}_merged.json'
    if config_path.exists():
        raise FileExistsError(config_path)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT/'configs/adni12/BrainPromptPro_adni12_holdout.json').read_text())
    classes = list(CLASS_NAMES)
    assert config['data']['class_names'] == classes
    files, hashes = {}, {}

    def pin(key, path):
        files[key] = str(path)
        hashes[key] = sha256(path)

    pin('preparation_script', Path(__file__).resolve())
    pin('label_policy_source', ROOT/'data/labels.py')
    pin('protocol_template', ROOT/'configs/adni12/BrainPromptPro_adni12_holdout.json')
    for key in ('roi_prompt', 'roi_text', 'disease_prompt', 'disease_text'):
        pin(key, config['data'][key])
    roi_names = [line.split(':', 1)[0] for line in Path(config['data']['roi_text']).read_text().splitlines()]
    assert len(roi_names) == 116
    disease_lines = Path(config['data']['disease_text']).read_text().splitlines()
    assert [re.search(r'network of an? (\w+) subject', s)[1] for s in disease_lines] == classes
    aggregate_file = base/'ADNI_2/pipeline_logs/ADNI_metadata.csv'
    pin('aggregate_metadata', aggregate_file)
    aggregate = defaultdict(list)
    for row in read(aggregate_file):
        aggregate[row['Image Data ID']].append(row)
    records, fcs, coverage, conflicts, exclusions = [], [], [], [], []
    demographic_conflicts=[]
    digests, seen_images = {}, set()
    fc_digests = {}
    batch_summary = {}
    for n in (1, 2, 3, 4, 5):
        batch = f'ADNI_{n}'
        root = base/batch
        csv_files = list((root/'data').glob('*.csv'))
        assert len(csv_files) == 1, csv_files
        source = csv_files[0]
        pin(batch+'_download_csv', source)
        raw = read(source)
        pairs = defaultdict(list)
        for row in raw:
            pairs[(canonical(row['Subject']), re.sub(r'[^A-Za-z0-9]+', '', row['Visit']))].append(row)
        assert len(raw) == 400 and len(pairs) == 200
        current_by_image = {}
        if n == 2:
            manifest = root/'BrainPrompt_data/brainprompt_manifest.csv'
            pin(batch+'_historical_image_alignment', manifest)
            for row in read(manifest):
                assert row['matched_image_id'] not in current_by_image
                current_by_image[row['matched_image_id']] = row
        if n in (3, 4, 5):
            qc_path = root/'pipeline_logs/final_pipeline_summary.tsv'
            pin(batch+'_pipeline_qc', qc_path)
            pin(batch+'_pipeline_source', root/'pipeline_scripts/pipeline_lib.py')
            qc = {(r['subject'], r['session']): r for r in read(qc_path, '\t')}
            assert set(qc) == set(pairs)
            assert all(all(r[k] == 'PASS' for k in ('bids_status', 'fmriprep_status', 'connectivity_status')) for r in qc.values())
        batch_images = set()
        for (subject, visit), pair in sorted(pairs.items()):
            fmri = [r for r in pair if kind(r) == 'bold']
            t1 = [r for r in pair if kind(r) == 't1']
            assert len(fmri) == len(t1) == 1 and len(pair) == 2
            row, structural = fmri[0], t1[0]
            assert all(row[k] == structural[k] for k in ('Group', 'Sex', 'Age', 'Acq Date'))
            image = row['Image Data ID']
            assert image not in seen_images
            seen_images.add(image)
            batch_images.add(image)
            matches = aggregate[image]
            assert matches and {canonical(r['Subject']) for r in matches} == {subject}
            assert len({r['Group'] for r in matches}) == 1
            assert len({r['site_id'] for r in matches}) == 1
            if {r['Sex'] for r in matches} != {row['Sex']}:
                assert row['Sex'] in ('M', 'F'), 'Invalid demographic sex'
                demographic_conflicts.append(dict(source_batch=batch,fmri_image_id=image,field='Sex',download_value=row['Sex'],aggregate_value=';'.join(sorted({r['Sex'] for r in matches})),selected_source='download',reason='Use the same declared demographic source for the whole cohort'))
            global_row = matches[0]
            if row['Group'] != global_row['Group']:
                conflicts.append(dict(source_batch=batch, subject=subject, raw_visit=visit,
                    fmri_image_id=image, download_group=row['Group'], aggregate_group=global_row['Group'],
                    download_target=normalize_diagnosis(row['Group']), aggregate_target=normalize_diagnosis(global_row['Group']),
                    target_differs=normalize_diagnosis(row['Group']) != normalize_diagnosis(global_row['Group'])))
            item = dict(source_batch=batch, subject=subject, raw_visit=visit, fmri_image_id=image,
                t1_image_id=structural['Image Data ID'], download_group=row['Group'], aggregate_group=global_row['Group'])
            if n == 2 and image not in current_by_image:
                item.update(status='excluded_missing_original_image_in_preprocessed_manifest', input_session='', timeseries_file='')
                coverage.append(item)
                exclusions.append(dict(item))
                continue
            session = current_by_image[image]['session'] if n == 2 else visit
            ts = root/'ADNI_connectivity'/f'sub-{subject}'/f'ses-{session}'/f'sub-{subject}_ses-{session}_AAL116_features_timeseries.mat'
            if n == 2:
                assert current_by_image[image]['subject'] == subject
                assert current_by_image[image]['final_status'] == 'READY'
                assert ts == Path(current_by_image[image]['timeseries_file'])
            pin(batch+'_'+image+'_timeseries', ts)
            raw_ts = np.asarray(loadmat(ts)['data'], dtype=np.float32).T.copy()
            assert raw_ts.shape[0] == 116 and raw_ts.shape[1] >= 20 and np.isfinite(raw_ts).all()
            assert (raw_ts.std(-1) > 0).all()
            digest = hashlib.sha256(raw_ts.tobytes()).hexdigest()
            assert digest not in digests, (image, digests.get(digest))
            digests[digest] = image
            labels_file = ts.with_name(ts.name.replace('_features_timeseries.mat', '_labels.tsv'))
            pin(batch+'_'+image+'_roi_labels', labels_file)
            assert [r['label'] for r in read(labels_file, '\t')] == roi_names
            if n in (3, 4, 5):
                assert int(qc[(subject, session)]['bold_timepoints']) == raw_ts.shape[1]
            fc = np.corrcoef(raw_ts).astype(np.float32)
            assert np.isfinite(fc).all()
            fc_digest = hashlib.sha256(fc.tobytes()).hexdigest()
            if fc_digest in fc_digests:
                raise ValueError(f'Identical FC detected: {image} and {fc_digests[fc_digest]}; verify acquisition before deduplication')
            fc_digests[fc_digest] = image
            original = row['Group'] if args.label_source == 'download' else global_row['Group']
            diagnosis = normalize_diagnosis(original)
            prepared = dict(graph_index=len(records), source_batch=batch, subject=subject, session=session,
                raw_visit=visit, fmri_image_id=image, t1_image_id=structural['Image Data ID'],
                label=classes.index(diagnosis), diagnosis=diagnosis, original_diagnosis=original,
                label_policy=LABEL_POLICY, clinical_label_source=args.label_source,
                age=float(row['Age']), sex=row['Sex'], site_id=str(int(global_row['site_id'])),
                acquisition_date=row['Acq Date'], timepoints=int(raw_ts.shape[1]), timeseries_file=str(ts),
                label_metadata_source=str(source if args.label_source == 'download' else aggregate_file),
                demographic_metadata_source=str(source), site_source=str(aggregate_file))
            validate_record_label(prepared)
            sex = {'F': 'female', 'M': 'male'}[prepared['sex']]
            prepared['prompt'] = (f'The subject is a {int(round(prepared["age"]))}-year-old {sex}, '
                                  f'with the image collected from study site {prepared["site_id"]}.')
            records.append(prepared)
            fcs.append(torch.from_numpy(fc))
            item.update(status='included', input_session=session, timeseries_file=str(ts))
            coverage.append(item)
        for image, row in sorted(current_by_image.items()):
            if image not in batch_images:
                exclusions.append(dict(source_batch=batch, subject=row['subject'], raw_visit='',
                    fmri_image_id=image, t1_image_id='', download_group='', aggregate_group='',
                    status='excluded_historical_replacement_outside_download_list', input_session=row['session'],
                    timeseries_file=row['timeseries_file']))
        selected = [r for r in records if r['source_batch'] == batch]
        batch_summary[batch] = dict(raw_image_records=len(raw), raw_functional_images=len(pairs),
            included_functional_images=len(selected), subjects=len({r['subject'] for r in selected}),
            class_counts=dict(Counter(r['diagnosis'] for r in selected)))
    # Known missing originals from historical ADNI_2; other exceptions require an explicit audit.
    missing = {r['fmri_image_id'] for r in coverage if r['status'] != 'included'}
    assert missing == expected_ids('missing_original_images'), 'Unexpected missing acquisitions; review private audit'
    replacements = {r['fmri_image_id'] for r in exclusions if 'replacement' in r['status']}
    assert replacements == expected_ids('excluded_historical_replacements'), 'Unexpected replacement acquisitions; review private audit'
    assert len(records) == 998 and len(coverage) == 1000
    ids = [record_id(r) for r in records]
    assert len(ids) == len(set(ids))
    summary = dict(label_policy=LABEL_POLICY, clinical_label_source=args.label_source,
        clinical_source_authority='Experimental source convention; not an independent clinical adjudication',
        batches=batch_summary, samples=len(records), subjects=len({r['subject'] for r in records}),
        classes=dict(Counter(r['diagnosis'] for r in records)),
        mci_merged=sum(r['original_diagnosis'] == 'MCI' for r in records),
        raw_functional_images=len(coverage), missing_original_images=sorted(missing),
        excluded_historical_replacements=sorted(replacements),
        metadata_raw_group_disagreements=len(conflicts), demographic_source_disagreements=len(demographic_conflicts), metadata_target_disagreements=sum(r['target_differs'] for r in conflicts),
        no_duplicate_image_ids=True, no_duplicate_timeseries=True, no_duplicate_fc=True,
        multiple_visit_subjects=sum(v > 1 for v in Counter(r['subject'] for r in records).values()),
        subject_visit_distribution=dict(Counter(Counter(r['subject'] for r in records).values())), merge_policy='Concatenate batches in order, sort subject/visit within each; reject duplicate Image IDs, scan IDs or time-series content; retain distinct longitudinal scans.',
        shuffled_before_split=False, split_created=False,
        ad_samples=sum(r['diagnosis']=='AD' for r in records),
        non_ad_samples=sum(r['diagnosis']!='AD' for r in records),
        ad_subjects=len({r['subject'] for r in records if r['diagnosis']=='AD'}))
    csv_out(out/'manifest.csv', records)
    csv_out(out/'coverage.csv', coverage)
    csv_out(out/'excluded.csv', exclusions)
    csv_out(out/'label_source_conflicts.csv', conflicts)
    csv_out(out/'demographic_source_conflicts.csv',demographic_conflicts)
    csv_out(out/'ad_subject_trace.csv', [r for r in records if r['diagnosis'] == 'AD'])
    for batch, stats in batch_summary.items():
        chosen=[r for r in records if r['source_batch']==batch]
        stats.update(ad_samples=sum(r['diagnosis']=='AD' for r in chosen), ad_subjects=len({r['subject'] for r in chosen if r['diagnosis']=='AD'}))
    write_json(out/'audit_summary.json', summary)
    print('INPUT_AUDIT', json.dumps(summary), flush=True)
    # Verify that the first four batches remain byte-identical at the FC/record level.
    prior=torch.load(ROOT/'outputs/data/adni1234_download_mci_to_lmci_v1_v1/cohort.pt',weights_only=True,map_location='cpu')
    assert records[:798] == prior['records']
    assert torch.equal(torch.stack(fcs[:798]),prior['fc'])
    for r in records:
        r.update(Subject_ID=r['subject'], Visit_ID=r['raw_visit'], Sample_ID=record_id(r))
    csv_out(out/'manifest.csv', records)
    from transformers import AutoTokenizer
    from llm2vec.models import LlamaBiModel
    model_path = base/'ADNI_2/models/Llama-encoder-1.0B'
    for path in sorted(model_path.iterdir()):
        if path.is_file() and path.suffix in ('.json', '.safetensors', '.model'):
            pin('encoder_'+path.name, path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = LlamaBiModel.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
    cache = {r['prompt']: e for r, e in zip(prior['records'], prior['subject_prompt'])}
    embeddings = []
    for i, row in enumerate(records):
        prompt = row['prompt']
        if prompt not in cache:
            inputs = {k: v.to(device) for k, v in tokenizer(prompt, return_tensors='pt').items()}
            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state.squeeze(0)
            embedding = hidden[:-1].mean(0).detach().cpu().float()
            assert embedding.shape == (2048,) and torch.isfinite(embedding).all()
            cache[prompt] = embedding
        embeddings.append(cache[prompt])
        if (i+1) % 100 == 0:
            print(f'Encoded {i+1}/{len(records)} metadata prompts', flush=True)
    assert torch.equal(torch.stack(embeddings[:798]), prior['subject_prompt'])
    provenance = dict(label_policy=LABEL_POLICY, clinical_label_source=args.label_source,
        sha256=hashes, source_files=files, scope='adni12345_subset',
        alignment='download Image ID to historical audited manifest (ADNI_2) or unique subject/visit AAL116 output (ADNI_1/3/4/5)',
        roi_name_order_verified=True, full_fc_source='np.corrcoef(float32 AAL116 time series transposed to [116,T])',
        subject_embedding_generation='local frozen LlamaBiModel, last_hidden_state[:-1].mean(0); age/sex/site only; cached identical text',
        clinical_metadata_policy=f'all diagnosis fields from {args.label_source}; MCI→LMCI; age/sex from download CSV; exact Image ID for aggregate site',
        excluded_scans=exclusions, audit_summary=summary)
    for key, path in files.items():
        assert sha256(path) == hashes[key], f'Source changed during preparation: {key}'
    bundle = out/'cohort.pt'
    torch.save(dict(label_policy=LABEL_POLICY, records=records, subject_ids=ids,
        fc=torch.stack(fcs), subject_prompt=torch.stack(embeddings), class_names=classes,
        roi_names=roi_names, provenance=provenance), bundle)
    write_json(out/'provenance.json', provenance)
    config['data'].update(cohort_name='ADNI_Combined', scope='adni12345_subset',
        required_batches=['ADNI_1', 'ADNI_2', 'ADNI_3', 'ADNI_4', 'ADNI_5'], clinical_label_source=args.label_source,
        bundle='@pro/outputs/data/'+name+'/cohort.pt', bundle_sha256=sha256(bundle))
    config['data']['split_mode']='fixed_sample_stratified_holdout'
    config['params']['device'] = args.device
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    print('PREPARED', json.dumps(dict(config=str(config_path), bundle=str(bundle), sha256=config['data']['bundle_sha256'])), flush=True)


if __name__ == '__main__':
    main()
