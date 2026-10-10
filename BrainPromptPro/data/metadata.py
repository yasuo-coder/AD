"""ID matching and preprocessing fitted only on a fold's training records."""
import math
import numpy as np
import torch


def record_id(row):
    subject, session = row['subject'].strip(), row['session'].strip()
    if not subject or not session:
        raise ValueError('Empty subject/session ID')
    return subject + '/' + session


def align_records(reference, other, description):
    """Return other in reference order; reject duplicate/missing/extra IDs."""
    ids = [record_id(r) for r in reference]
    lookup = {record_id(r): r for r in other}
    if len(set(ids)) != len(ids) or len(lookup) != len(other):
        raise ValueError(f'Duplicate subject/session ID in {description}')
    if set(ids) != set(lookup):
        raise ValueError(f'Subject/session ID mismatch in {description}')
    return [lookup[i] for i in ids]


class MetadataEncoder:
    """0 is explicit unknown, F=1/M=2; sites learned once from train only."""
    def fit(self, train_records, site_column='site_id', missing_site='error'):
        if missing_site not in ('error', 'unknown'):
            raise ValueError('missing_site must be error or unknown')
        self.site_column, self.missing_site = site_column, missing_site
        age = np.array([float(r['age']) for r in train_records], dtype=np.float64)
        if not len(age) or not np.isfinite(age).all():
            raise ValueError('Training ages must be nonempty and finite')
        self.age_mean = float(age.mean())
        self.age_std = max(float(age.std()), 1e-8)
        sites = [self._site(r) for r in train_records]
        self.site_mapping = {s: i + 1 for i, s in enumerate(sorted(set(sites) - {None}))}
        self.fit_ids = [record_id(r) for r in train_records]
        return self

    def _site(self, row):
        site = str(row.get(self.site_column, '')).strip()
        if site.lower() in ('', 'nan', 'none', 'unknown'):
            if self.missing_site == 'error':
                raise ValueError(f'Missing {self.site_column}; provide verified metadata or explicitly set missing_site=unknown')
            return None
        return site

    def transform(self, records):
        ages, sexes, sites = [], [], []
        for row in records:
            age = float(row['age'])
            if not math.isfinite(age):
                raise ValueError('Nonfinite age')
            sex = row['sex'].strip().upper()
            if sex not in ('F', 'M'):
                raise ValueError('sex must be documented F/M, not guessed numeric coding')
            ages.append((age - self.age_mean) / self.age_std)
            sexes.append({'F': 1, 'M': 2}[sex])
            sites.append(self.site_mapping.get(self._site(row), 0))
        return {'age': torch.tensor(ages, dtype=torch.float32),
                'sex': torch.tensor(sexes, dtype=torch.long), 'site': torch.tensor(sites, dtype=torch.long)}

    def state_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(); obj.__dict__.update(state)
        return obj
