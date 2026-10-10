"""Read-only seven-batch inventory and a strict full-ADNI scope contract.

An inventory is evidence of file availability, not evidence that local scans
match the paper's subjects, labels, exclusions or official index ordering.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BATCHES = tuple(f'ADNI_{i}' for i in range(1, 8))


def csv_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_cohort(source_root, index_dir):
    """Inventory fixed batches without reading the large DICOM volumes."""
    root = Path(source_root)
    scans, batches, occurrences = [], [], defaultdict(list)
    for name in BATCHES:
        batch = root / name
        metadata = sorted((batch / 'data').glob('*.csv'))
        records = defaultdict(list)
        for path in metadata:
            for row in csv_rows(path):
                # These source CSVs contain both structural and functional images.
                # Keep one inventory row per ID/visit; retain all source rows.
                subject = 'sub-' + row['Subject'].replace('_', '')
                session = 'ses-' + row['Visit']
                records[subject, session].append((path, row))
        for (subject, session), rows in sorted(records.items()):
            bids = sorted((batch / 'ADNI_BIDS' / subject / session / 'func').glob('*bold.nii.gz'))
            preproc = sorted((batch / 'ADNI_fmriprep' / subject / session / 'func').glob('*desc-preproc_bold.nii.gz'))
            timeseries = sorted((batch / 'ADNI_connectivity' / subject / session).glob('*AAL116*timeseries.mat'))
            identity = f'{subject}/{session}'
            occurrences[identity].append(name)
            scans.append(dict(batch=name, subject=subject, session=session,
                source_csvs='|'.join(sorted({str(p) for p, _ in rows})),
                source_image_rows=len(rows),
                source_diagnoses='|'.join(sorted({r['Group'] for _, r in rows})),
                source_ages='|'.join(sorted({r['Age'] for _, r in rows})),
                source_sexes='|'.join(sorted({r['Sex'] for _, r in rows})),
                bids_bold_count=len(bids), preprocessed_bold_count=len(preproc),
                aal116_timeseries_count=len(timeseries),
                timeseries_files='|'.join(map(str, timeseries))))
        raw_subjects = [p for p in (batch / 'data').glob('*_S_*') if p.is_dir()]
        manifest = batch / 'BrainPrompt_data/brainprompt_manifest.csv'
        manifest_rows = csv_rows(manifest) if manifest.is_file() else []
        batches.append(dict(batch=name, directory_exists=batch.is_dir(),
            raw_subject_directories=len(raw_subjects), metadata_csvs=[str(p) for p in metadata],
            metadata_sha256={str(p):file_hash(p) for p in metadata},
            metadata_image_rows=sum(len(v) for v in records.values()),
            metadata_scan_ids=len(records),
            bids_bold_files=len(list((batch / 'ADNI_BIDS').glob('sub-*/ses-*/func/*bold.nii.gz'))),
            preprocessed_bold_files=len(list((batch / 'ADNI_fmriprep').glob('sub-*/ses-*/func/*desc-preproc_bold.nii.gz'))),
            aal116_timeseries_files=len(list((batch / 'ADNI_connectivity').glob('sub-*/ses-*/*AAL116*timeseries.mat'))),
            graph_files=[str(p) for p in sorted((batch / 'BrainPrompt_data').glob('*.bin'))],
            manifest_rows=len(manifest_rows),
            manifest_class_counts=dict(Counter(r.get('diagnosis', '') for r in manifest_rows))))
    indices = {}
    for role in ('train', 'val', 'test'):
        with (Path(index_dir) / f'{role}.index').open() as stream:
            indices[role] = [list(map(int, row)) for row in csv.reader(stream)]
    if len({len(v) for v in indices.values()}) != 1:
        raise ValueError('Official index files have different fold counts')
    fold_summary = []
    for i in range(len(indices['train'])):
        values = [j for role in indices for j in indices[role][i]]
        fold_summary.append(dict(fold=i, sizes={r:len(v[i]) for r,v in indices.items()},
            unique_samples=len(set(values)), min_index=min(values), max_index=max(values),
            no_duplicate_indices=len(values)==len(set(values))))
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), source_root=str(root),
        required_batches=list(BATCHES), batches=batches, inventory_scan_rows=len(scans),
        unique_scan_ids=len(occurrences),
        unique_subjects=len({r['subject'] for r in scans}),
        cross_batch_duplicate_scans={k:v for k,v in occurrences.items() if len(v)>1},
        official_folds=fold_summary,
        official_index_sha256={r:file_hash(Path(index_dir)/f'{r}.index') for r in indices},
        paper_cohort_identity_verified=False,
        status='inventory_only_not_a_training_manifest')
    return report, scans


def full_cohort_blockers(config):
    """Never silently substitute a subset or guess anonymous prompt ordering."""
    if config.get('scope') != 'full_adni':
        return []
    from ..runtime import ROOT
    problems = []
    if config.get('required_batches') != list(BATCHES):
        problems.append('Full ADNI must account for every batch ADNI_1 through ADNI_7')
    value = config['data_root']
    root = ROOT / value[len('@pro/'):] if value.startswith('@pro/') else Path(value)
    config['data_root'] = str(root)
    keys = ('manifest', 'graph_bin', 'graph_order', 'prompt_order', 'subject_prompt', 'subject_text')
    for key in keys:
        if not (root / config[key]).is_file():
            problems.append(f'Missing unified {key}: {root / config[key]}')
    if config.get('alignment_provenance') != 'audited_legacy_generator_and_order_csv':
        problems.append('Full-cohort graph/metadata/prompt alignment has not been audited')
    if not isinstance(config.get('prompt_site_count'), int) or config['prompt_site_count'] < 1:
        problems.append('Full-cohort prompt site count must be verified from its own metadata')
    required_hashes = set(keys) | set(config.get('generator_sources', {}))
    if not required_hashes <= config.get('expected_sha256', {}).keys() or not config.get('generator_sources'):
        problems.append('Full-cohort files and audited generators need pinned SHA256 provenance')
    if config.get('split_mode') != 'official' or not config.get('index_dir'):
        problems.append('Full paper replication requires its verified official index files')
    evidence = config.get('official_order_evidence')
    if not evidence:
        problems.append('Missing official index-to-subject ordering evidence; file counts are insufficient')
    else:
        path = root / evidence['path']
        if not path.is_file() or file_hash(path) != evidence['sha256']:
            problems.append('Official ordering evidence is missing or has changed')
        else:
            audit = json.loads(path.read_text())
            if not audit.get('source_reference') or not audit.get('alignment_method'):
                problems.append('Official ordering evidence needs its source and alignment method')
            # Bind the reviewed ID mapping to the exact graph order and splits.
            for key in ('manifest', 'graph_order'):
                artifact = root / config[key]
                if not artifact.is_file() or audit.get('sha256', {}).get(key) != file_hash(artifact):
                    problems.append(f'Official ordering evidence does not bind {key}')
            for role in ('train', 'val', 'test'):
                artifact = Path(config['index_dir']) / f'{role}.index'
                if not artifact.is_file() or audit.get('sha256', {}).get(role+'_index') != file_hash(artifact):
                    problems.append(f'Official ordering evidence does not bind {role} indices')
    manifest = root / config['manifest']
    if manifest.is_file():
        rows = csv_rows(manifest)
        if len(rows) != config['expected_samples']:
            problems.append(f'Expected {config["expected_samples"]} verified paper samples, got {len(rows)}')
        if {r.get('source_batch') for r in rows} != set(BATCHES):
            problems.append('Unified manifest must retain source_batch for all seven batches')
    return problems
