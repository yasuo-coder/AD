"""Load a hash-pinned, ID-bearing cohort prepared inside BrainPromptPro."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .brainnet_pro import sha256, load_embeddings
from .metadata import record_id
from .labels import LABEL_POLICY, CLASS_NAMES, require_label_policy, validate_record_label
from ..runtime import ROOT


def pro_path(value):
    return ROOT/value[5:] if value.startswith('@pro/') else Path(value)


class PreparedCohortDataset(Dataset):
    def __init__(self, config):
        require_label_policy(config.get('label_policy'), 'data config')
        path = pro_path(config['bundle'])
        if sha256(path) != config['bundle_sha256']:
            raise ValueError('Prepared cohort hash mismatch; rerun the audited preparation')
        value = torch.load(path, map_location='cpu', weights_only=True)
        require_label_policy(value.get('label_policy'), 'prepared bundle')
        require_label_policy(value['provenance'].get('label_policy'), 'bundle provenance')
        self.records = value['records']
        ids = [record_id(r) for r in self.records]
        if len(ids) != len(set(ids)) or ids != value['subject_ids']:
            raise ValueError('Prepared cohort contains duplicate or misaligned IDs')
        if {r['source_batch'] for r in self.records} != set(config['required_batches']):
            raise ValueError('Prepared cohort does not cover the requested batches')
        self.class_names = list(config['class_names'])
        if self.class_names != list(CLASS_NAMES) or self.class_names != value['class_names']:
            raise ValueError('Prepared cohort class order mismatch')
        self.num_classes = len(self.class_names)
        self.node_num = config['node_num']
        self.fc, self.subject_prompt = value['fc'], value['subject_prompt']
        self.labels = torch.tensor([int(r['label']) for r in self.records])
        if self.fc.shape != (len(ids), self.node_num, self.node_num):
            raise ValueError('Prepared FC shape mismatch')
        if self.subject_prompt.ndim != 2 or len(self.subject_prompt) != len(ids):
            raise ValueError('Prepared prompt shape mismatch')
        if not torch.isfinite(self.fc).all() or not torch.isfinite(self.subject_prompt).all():
            raise ValueError('Prepared cohort contains nonfinite values')
        for row in self.records:
            validate_record_label(row)
            if row['diagnosis'] != self.class_names[int(row['label'])]:
                raise ValueError('Prepared diagnosis/label mismatch')
        self.roi_prompt = load_embeddings(config['roi_prompt'])
        self.disease_prompt = load_embeddings(config['disease_prompt'])
        self.roi_names = value['roi_names']
        if self.roi_prompt.shape[0] != self.node_num or len(self.disease_prompt) != self.num_classes:
            raise ValueError('Shared prompt count mismatch')
        for key in ('roi_prompt','roi_text','disease_prompt','disease_text'):
            if sha256(config[key]) != value['provenance']['sha256'][key]:
                raise ValueError(f'Shared prompt provenance changed: {key}')
        self.provenance = dict(value['provenance'])
        self.provenance['sha256'] = dict(value['provenance']['sha256'], prepared_bundle=config['bundle_sha256'])
        self.provenance.update(cohort_name=config['cohort_name'], scope=config['scope'],
                               samples=len(ids), subjects=len({r['subject'] for r in self.records}),
                               class_names=self.class_names, subject_ids=ids)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return dict(fc=self.fc[index], subject_prompt=self.subject_prompt[index],
                    label=self.labels[index], record=self.records[index])
