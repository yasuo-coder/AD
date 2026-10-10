# All identifiers and demographics in this test are synthetic fixtures, not ADNI records.
import numpy as np
from BrainPromptPro.data.splits import make_holdout, validate_split
from BrainPromptPro.metrics import classification_metrics, subject_metrics
from BrainPromptPro.scripts.prepare_adni12 import metadata_site_lookup, make_prompt


def test_grouped_holdout_is_reproducible_complete_and_disjoint():
    records, labels = [], []
    for cls in range(5):
        for subject in range(10):
            for session in range(2):
                records.append(dict(subject=f'{cls}-{subject}',session=str(session)))
                labels.append(cls)
    split=make_holdout(records,labels)
    assert split==make_holdout(records,labels)
    validate_split(split,records)
    assert [len(split[k]) for k in ('train','val','test')]==[60,20,20]
    assert all({labels[i] for i in idx}==set(range(5)) for idx in split.values())
    labels[1]=1
    try:
        make_holdout(records,labels)
    except ValueError as error:
        assert 'one diagnosis per subject' in str(error)
    else:
        raise AssertionError('Longitudinal label changes require an explicit protocol')


def test_metrics_match_hand_calculation_and_keep_absent_classes():
    y=np.array([0,0,1,1])
    p=np.array([[.9,.1,0],[.2,.8,0],[.3,.7,0],[.6,.4,0]])
    m=classification_metrics(y,p,3)
    assert m['confusion_matrix']==[[1,1,0],[1,1,0],[0,0,0]]
    assert m['accuracy']==.5 and m['f1_weighted']==.5 and m['recall_micro']==.5
    assert np.isclose(m['f1_macro'],1/3) and m['balanced_accuracy']==.5
    assert m['roc_auc_macro'] is None and m['per_class'][2]['support']==0
    assert m['per_class'][0]['specificity']==.5


def test_subject_metrics_pool_scans_once_per_person():
    p=np.array([[.1,.9],[.9,.1],[.2,.8]])
    m=subject_metrics(['A/one','A/two','B/one'],[0,0,1],p,2)
    assert m['samples']==2 and m['accuracy']==1.
    m=subject_metrics(['A/one','A/two','B/one'],[0,1,1],p,2)
    assert not m['available']


def test_site_alignment_uses_full_subject_id_and_prompts_exclude_labels():
    sites=metadata_site_lookup([dict(Subject='002_S_0001',site_id='1'),
                               dict(Subject='002_S_0001_2',site_id='1'),
                               dict(Subject='002_S_0002',site_id='7')])
    assert sites=={'002S0001':{'1'},'002S0002':{'7'}}
    row=dict(age=67.4,sex='F',site_id='1',diagnosis='CN',label=0)
    original=make_prompt(row)
    row.update(diagnosis='AD',label=1)
    assert make_prompt(row)==original
