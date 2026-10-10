"""Regression coverage for the author-confirmed MCI -> LMCI cohort migration."""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
from BrainPromptPro.data.brainnet_pro import read_csv
from BrainPromptPro.data.data import LoadData
from BrainPromptPro.data.labels import LABEL_POLICY, label_for_diagnosis, normalize_diagnosis, validate_record_label
from BrainPromptPro.data.prepared_cohort import PreparedCohortDataset
from BrainPromptPro.data.splits import make_holdout, make_splits, validate_split
from BrainPromptPro.runtime import ROOT


def expect_error(fn, text):
    try:
        fn()
    except ValueError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError('Expected rejection: '+text)


def test_both_projects_normalize_mci_without_changing_emci():
    spec = importlib.util.spec_from_file_location('baseline_labels', '/home/panyao/AD/BrainPrompt/adni_labels.py')
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    for raw, name, label in [('MCI','LMCI',4),(' mci ','LMCI',4),('LMCI','LMCI',4),
                             ('EMCI','EMCI',3),('CN','CN',0),('AD','AD',1),('SMC','SMC',2)]:
        assert normalize_diagnosis(raw) == baseline.normalize_diagnosis(raw) == name
        assert label_for_diagnosis(raw) == baseline.label_for_diagnosis(raw) == label
    assert baseline.normalize_numeric_label(5) == 4
    assert baseline.normalize_numeric_label(3) == 3
    bad = dict(original_diagnosis='MCI',diagnosis='EMCI',label=3,label_policy=LABEL_POLICY)
    expect_error(lambda: validate_record_label(bad), 'Original/canonical')


def test_rebuilt_sources_include_all_mci_and_preserve_original_records():
    import dgl
    source = Path('/data2/panyao_data/ADNI')
    all_rows = []
    for batch, graph, dataset, expected in [
        ('ADNI_1','adni_ADNI1_AAL116','adni_ADNI1_AAL116',41),
        ('ADNI_2','adni_custom_AAL116','adni_custom_AAL116',26)]:
        rows = read_csv(source/batch/'BrainPrompt_data/brainprompt_manifest.csv')
        graphs, labels = dgl.load_graphs(str(source/batch/'BrainPrompt_data'/(graph+'.bin')))
        assert len(rows) == len(graphs) == len(labels['glabel'])
        for row in rows: validate_record_label(row)
        assert sum(r['original_diagnosis']=='MCI' for r in rows) == expected
        assert labels['glabel'].tolist() == [int(r['label']) for r in rows]
        make_splits(rows,labels['glabel'].numpy(),index_dir='/home/panyao/AD/BrainPrompt/data/'+dataset)
        all_rows.extend(rows)
    assert len(all_rows) == 399
    assert Counter(r['diagnosis'] for r in all_rows) == dict(CN=253,AD=8,SMC=23,EMCI=24,LMCI=91)
    excluded1 = read_csv(source/'ADNI_1/BrainPrompt_data/brainprompt_excluded.csv')
    excluded2 = read_csv(source/'ADNI_2/BrainPrompt_data/brainprompt_manifest_excluded.csv')
    assert excluded1 == [] and len(excluded2) == 1
    assert excluded2[0]['diagnosis']=='CN' and excluded2[0]['final_status']=='EXCLUDE_DUPLICATE_BOLD'


def test_prepared_cohort_policy_and_reject_historical_inputs():
    config = json.loads((ROOT/'configs/adni12/BrainPromptPro_adni12_holdout.json').read_text())['data']
    data = LoadData(config)
    assert len(data)==399 and data.provenance['label_policy']==LABEL_POLICY
    assert sum(r['original_diagnosis']=='MCI' for r in data.records)==67
    split = make_holdout(data.records,data.labels.numpy(),seed=41)
    validate_split(split,data.records)
    for indices in split.values():
        assert {int(data.labels[i]) for i in indices}==set(range(5))
    old = copy.deepcopy(config)
    old.pop('label_policy')
    expect_error(lambda: LoadData(old),'historical or missing label policy')
    old = json.loads((ROOT/'outputs/results/adni12_holdout_seed41_20261002/config.json').read_text())['data']
    old['label_policy'] = LABEL_POLICY
    # Renaming a config cannot turn the old filtered bundle into a new cohort.
    expect_error(lambda: PreparedCohortDataset(old),'prepared bundle: historical')


def test_subject_leakage_still_rejected():
    rows = [dict(subject=s) for s in ('A','A','B','C')]
    expect_error(lambda: validate_split(dict(train=[0,2],val=[3],test=[1]),rows), 'Subject leakage')
