"""Reproducible grouped CV, plus strict validation of imported index files."""
import csv
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split


def make_holdout(records, labels, seed=41, test_size=0.2, val_size=0.2):
    """Stratify whole subjects once, with an explicit consistent-label contract."""
    if not 0 < test_size < 1 or not 0 < val_size < 1-test_size:
        raise ValueError('Holdout fractions must be positive and leave training data')
    subject_labels = {}
    for row, label in zip(records, labels):
        subject = row['subject']
        if subject in subject_labels and subject_labels[subject] != int(label):
            raise ValueError('Holdout stratification requires one diagnosis per subject; specify a longitudinal protocol')
        subject_labels[subject] = int(label)
    subjects = sorted(subject_labels)
    train_val, test = train_test_split(subjects, test_size=test_size, random_state=seed,
                                       stratify=[subject_labels[s] for s in subjects])
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=seed,
                                  stratify=[subject_labels[s] for s in train_val])
    groups = dict(train=set(train), val=set(val), test=set(test))
    split = {name:[i for i,r in enumerate(records) if r['subject'] in members] for name,members in groups.items()}
    validate_split(split, records)
    expected = set(map(int, labels))
    if any({int(labels[i]) for i in values} != expected for values in split.values()):
        raise ValueError('A holdout partition is missing a class; more subjects or another protocol is needed')
    return split


def validate_split(split, records):
    sets, groups = {}, {}
    for name in ('train', 'val', 'test'):
        values = split[name]
        if not values or len(values) != len(set(values)) or any(i < 0 or i >= len(records) for i in values):
            raise ValueError(f'Invalid {name} indices')
        sets[name] = set(values)
        groups[name] = {records[i]['subject'] for i in values}
    if set.union(*sets.values()) != set(range(len(records))):
        raise ValueError('Split must cover every scan exactly once')
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if sets[left] & sets[right] or groups[left] & groups[right]:
            raise ValueError(f'Subject leakage between {left} and {right}')


def make_splits(records, labels, seed=41, n_splits=10, index_dir=None):
    if index_dir:
        from pathlib import Path
        imported = {name: [list(map(int, r)) for r in csv.reader((Path(index_dir) / (name + '.index')).open())]
                    for name in ('train', 'val', 'test')}
        if any(len(rows) != n_splits for rows in imported.values()):
            raise ValueError('Wrong number of imported folds')
        folds = [{name: imported[name][i] for name in imported} for i in range(n_splits)]
    else:
        groups = [r['subject'] for r in records]
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tests = [idx.tolist() for _, idx in skf.split(np.zeros(len(records)), labels, groups)]
        all_idx = set(range(len(records)))
        folds = [dict(test=tests[i], val=tests[(i + 1) % n_splits],
                      train=sorted(all_idx - set(tests[i]) - set(tests[(i + 1) % n_splits]))) for i in range(n_splits)]
    for fold in folds:
        validate_split(fold, records)
    test_indices = [i for fold in folds for i in fold['test']]
    if sorted(test_indices) != list(range(len(records))):
        raise ValueError('CV test folds must partition the dataset')
    return folds
