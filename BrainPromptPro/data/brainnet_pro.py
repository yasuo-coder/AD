"""Read-only ADNI adapter: full Pearson FC, explicit ID and provenance checks."""
import csv
import hashlib
import re
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from .metadata import align_records, record_id
from .labels import LABEL_POLICY, CLASS_NAMES, require_label_policy, validate_record_label, normalize_numeric_label


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def load_embeddings(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    value = torch.stack(value) if isinstance(value, (list, tuple)) else value
    if value.ndim == 3 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError(f'Expected finite [count,prompt_dim] tensor: {path}')
    return value.float()


class BrainNetProDataset(Dataset):
    """Legacy artifacts are accepted only with ID manifests and verified sources.

    The legacy prompt tensor has no embedded IDs. Its ordering is established by
    the audited encoder script + prompt-order CSV + exact prompt text. This is
    recorded explicitly; an arbitrary anonymous .pt is not accepted.
    """
    def __init__(self, config):
        require_label_policy(config.get('label_policy'), 'data config')
        self.config = config
        root = Path(config['data_root'])
        path = lambda name: root / config[name]
        self.records = read_csv(path('manifest'))
        self.records.sort(key=lambda r: int(r['graph_index']))
        if [int(r['graph_index']) for r in self.records] != list(range(len(self.records))):
            raise ValueError('manifest graph_index must uniquely cover 0..S-1')
        if len({record_id(r) for r in self.records}) != len(self.records):
            raise ValueError('Duplicate scan ID in manifest')
        graph_order = align_records(self.records, read_csv(path('graph_order')), 'graph order')
        prompt_order = align_records(self.records, read_csv(path('prompt_order')), 'prompt order')
        if config.get('alignment_provenance') != 'audited_legacy_generator_and_order_csv':
            raise ValueError('Legacy embedding ordering needs audited generator provenance')
        self.provenance = {'label_policy': LABEL_POLICY, 'alignment': config['alignment_provenance'], 'sha256': {}}
        for key in ('manifest', 'graph_order', 'prompt_order', 'subject_prompt', 'subject_text', 'graph_bin'):
            self.provenance['sha256'][key] = sha256(path(key))
        for name, source in config['generator_sources'].items():
            self.provenance['sha256'][name] = sha256(source)
        for name, expected in config.get('expected_sha256', {}).items():
            if self.provenance['sha256'].get(name) != expected:
                raise ValueError(f'Provenance changed for {name}; re-audit alignment before training')
        prompts = Path(path('subject_text')).read_text().splitlines()
        self.subject_prompt = load_embeddings(path('subject_prompt'))
        if len(prompts) != len(self.records) or len(self.subject_prompt) != len(self.records):
            raise ValueError('Prompt counts disagree with audited manifests')
        import dgl
        graphs, graph_labels = dgl.load_graphs(str(path('graph_bin')))
        if len(graphs) != len(self.records):
            raise ValueError('Graph count mismatch')
        self.class_names = list(config['class_names'])
        if self.class_names != list(CLASS_NAMES):
            raise ValueError('ADNI class order must be CN, AD, SMC, EMCI, LMCI')
        self.num_classes = len(self.class_names)
        self.labels = torch.tensor([int(r['label']) for r in self.records])
        self.node_num = int(config['node_num'])
        fcs = []
        for i, (row, gr, pr, graph) in enumerate(zip(self.records, graph_order, prompt_order, graphs)):
            validate_record_label(row)
            identity = record_id(row)
            if row.get('final_status', 'READY') != 'READY':
                raise ValueError(f'Unresolved metadata: {identity}')
            if int(gr['graph_index']) != i or int(pr['graph_index']) != i:
                raise ValueError(f'Graph/prompt index inconsistent with ID: {identity}')
            label = int(row['label'])
            if not 0 <= label < self.num_classes or row['diagnosis'] != self.class_names[label]:
                raise ValueError(f'Disease class order mismatch: {identity}')
            if int(gr['label']) != label or normalize_numeric_label(graph_labels['glabel'][i]) != label:
                raise ValueError(f'Graph label mismatch: {identity}')
            if float(pr['age_original']) != float(row['age']):
                raise ValueError(f'Prompt age mismatch: {identity}')
            sex_word = {'F': 'female', 'M': 'male'}.get(row['sex'])
            if pr['sex'] != sex_word or pr['site_id'] != row[config['site_column']]:
                raise ValueError(f'Prompt sex/site mismatch: {identity}')
            expected = (f"The subject is a {int(round(float(row['age'])))}-year-old {sex_word}, "
                        f"with the image collected from the {int(float(pr['site_id']))} site "
                        f"out of a total of {config['prompt_site_count']} study locations.")
            if pr['prompt'] != expected or prompts[i] != expected:
                raise ValueError(f'Prompt text/ID provenance mismatch: {identity}')
            raw = np.asarray(loadmat(row['timeseries_file'])['data'], dtype=np.float32).T.copy()
            original = graph.ndata['N_features'].numpy()
            if raw.shape != original.shape or not np.array_equal(raw, original):
                raise ValueError(f'Graph has wrong subject/time series: {identity}')
            if raw.shape[0] != self.node_num or not np.isfinite(raw).all() or (raw.std(-1) == 0).any():
                raise ValueError(f'Invalid or constant ROI time series: {identity}')
            # Exact original BrainNet.py Pearson FC, before ANY thresholding.
            fc = np.corrcoef(raw).astype(np.float32)
            if not np.isfinite(fc).all():
                raise ValueError(f'Nonfinite Pearson FC: {identity}')
            fcs.append(torch.from_numpy(fc))
        self.fc = torch.stack(fcs)
        self.roi_prompt = load_embeddings(config['roi_prompt'])
        self.disease_prompt = load_embeddings(config['disease_prompt'])
        roi_text = Path(config['roi_text']).read_text().splitlines()
        self.roi_names = [line.split(':', 1)[0] for line in roi_text]
        disease_text = Path(config['disease_text']).read_text().splitlines()
        names = [re.search(r'network of an? (\w+) subject', line).group(1) for line in disease_text]
        if len(names) != len(set(names)) or len(self.disease_prompt) != len(names):
            raise ValueError('Disease prompt names are duplicate or inconsistent')
        if not set(self.class_names) <= set(names):
            raise ValueError('Disease prompt missing a configured class')
        self.disease_prompt = self.disease_prompt[[names.index(n) for n in self.class_names]]
        if self.roi_prompt.shape[0] != self.node_num or len(self.roi_names) != self.node_num:
            raise ValueError('ROI prompt count does not match atlas')
        for key in ('roi_prompt', 'disease_prompt', 'roi_text', 'disease_text'):
            self.provenance['sha256'][key] = sha256(config[key])
        self.provenance.update(samples=len(self), subjects=len({r['subject'] for r in self.records}),
                               cohort_name=config.get('cohort_name', 'legacy_unspecified'),
                               scope=config.get('scope', 'legacy_unspecified'),
                               class_names=self.class_names, full_fc_source='np.corrcoef(N_features)',
                               graph_timeseries_content_verified=True,
                               subject_ids=[record_id(r) for r in self.records])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return {'fc': self.fc[index], 'label': self.labels[index],
                'subject_prompt': self.subject_prompt[index], 'record': self.records[index]}


class FoldDataset(Dataset):
    def __init__(self, dataset, indices, encoder):
        self.dataset, self.indices = dataset, list(indices)
        self.metadata = encoder.transform([dataset.records[i] for i in self.indices])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = self.dataset[self.indices[index]]
        row = sample.pop('record')
        sample.update({k: v[index] for k, v in self.metadata.items()})
        sample['subject_id'] = record_id(row)
        sample['source_batch'] = row.get('source_batch', 'unspecified')
        return sample
