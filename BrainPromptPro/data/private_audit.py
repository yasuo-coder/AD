"""Load private, previously adjudicated acquisition exceptions outside Git.

This contains no patient identifiers. The public copy fails closed when the
private audit is absent; it never silently relaxes historical cohort checks.
"""
import hashlib
import json
import os
from pathlib import Path

AUDIT_SHA256 = '4a87176d74b17b53f4b71a8b51b573006941da4c43acfbf13545737629f3ed5d'
EXPECTED_COUNTS = {'missing_original_images': 2, 'excluded_historical_replacements': 1, 'adni4_sex_conflicts': 1}

def expected_ids(key):
    path = os.environ.get('BRAINPROMPT_PRIVATE_AUDIT_JSON')
    if not path:
        raise RuntimeError('Set BRAINPROMPT_PRIVATE_AUDIT_JSON to the private adjudicated audit file')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != AUDIT_SHA256:
        raise ValueError('Private audit checksum differs from the adjudicated snapshot')
    if key not in EXPECTED_COUNTS:
        raise ValueError('Unreviewed exception key')
    obj = json.loads(raw)
    values = obj.get(key)
    if not isinstance(values, list) or not values or any(not isinstance(x,str) or not x.strip() for x in values):
        raise ValueError('Private audit requires a nonempty string list for '+key)
    if len(set(values)) != len(values):
        raise ValueError('Duplicate private audit exceptions for '+key)
    if len(values) != EXPECTED_COUNTS[key]:
        raise ValueError('Private audit exception count differs from reviewed cohort')
    return set(values)
