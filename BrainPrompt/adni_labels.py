"""ADNI label policy confirmed by the author via the user's email (2026-10-04)."""
LABEL_POLICY = 'mci_to_lmci_v1'
CLASS_NAMES = ('CN', 'AD', 'SMC', 'EMCI', 'LMCI')
LABEL_MAP = {name: index for index, name in enumerate(CLASS_NAMES)}


def normalize_diagnosis(value):
    name = str(value).strip().upper()
    return 'LMCI' if name == 'MCI' else name


def label_for_diagnosis(value):
    return LABEL_MAP[normalize_diagnosis(value)]


def normalize_numeric_label(value):
    label = int(value)
    return 4 if label == 5 else label
