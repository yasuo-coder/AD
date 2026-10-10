import copy
import json
import torch
from BrainPromptPro.runtime import ROOT
from BrainPromptPro.data.data import LoadData
from BrainPromptPro.data.splits import make_splits
from BrainPromptPro.data.metadata import MetadataEncoder


def test_real_data_alignment_and_rebuilt_grouped_splits():
    config=json.loads((ROOT/'configs/smoke/BrainPromptPro_adni2_smoke.json').read_text())
    data=LoadData(config['data'])
    assert data.provenance['graph_timeseries_content_verified']
    assert data.fc.shape==(len(data),data.node_num,data.node_num)
    torch.testing.assert_close(data.fc,data.fc.transpose(-1,-2))
    folds = make_splits(data.records,data.labels.numpy(),
                        index_dir=config['baseline_root']+'/data/adni_custom_AAL116')
    assert len(folds) == 10
    changed=copy.deepcopy(config['data']);changed['expected_sha256']['subject_prompt']='0'*64
    try:
        LoadData(changed)
    except ValueError as error:
        assert 'Provenance changed' in str(error)
    else:
        raise AssertionError('Changed anonymous embedding must fail provenance validation')


def test_missing_site_is_explicit():
    records=[dict(subject='S',session='init',age='60',sex='F')]
    try:
        MetadataEncoder().fit(records)
    except ValueError as error:
        assert 'Missing' in str(error)
    else:
        raise AssertionError('Missing site must not silently become zero')
    encoder=MetadataEncoder().fit(records,missing_site='unknown')
    assert encoder.transform(records)['site'].tolist()==[0]
