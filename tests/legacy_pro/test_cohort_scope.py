# All identifiers and demographics in this test are synthetic fixtures, not ADNI records.
import copy
import csv
import json
import tempfile
from pathlib import Path
from BrainPromptPro.data.cohort import BATCHES, inspect_cohort
from BrainPromptPro.data.data import LoadData
from BrainPromptPro.runtime import ROOT


def test_full_default_rejects_adni2_substitution():
    full = json.loads((ROOT/'configs/adni_AAL116/BrainPromptPro_adni_AAL116.json').read_text())['data']
    smoke = json.loads((ROOT/'configs/smoke/BrainPromptPro_adni2_smoke.json').read_text())['data']
    assert full['required_batches'] == list(BATCHES)
    assert full['scope'] == 'full_adni' and full['split_mode'] == 'official'
    assert smoke['scope'] == 'smoke_subset'
    substituted = copy.deepcopy(full)
    for key in ('data_root','manifest','graph_bin','graph_order','prompt_order','subject_prompt',
                'subject_text','alignment_provenance','generator_sources','expected_sha256'):
        substituted[key] = smoke[key]
    try:
        LoadData(substituted)
    except ValueError as error:
        message = str(error)
        assert 'no subset fallback' in message
        assert f'Expected 1326 verified paper samples, got {len(list(csv.DictReader((Path(smoke["data_root"])/smoke["manifest"]).open())))}' in message
        assert 'all seven batches' in message
        assert 'index-to-subject ordering evidence' in message
    else:
        raise AssertionError('ADNI_2 cannot stand in for full paper ADNI')


def test_inventory_keeps_all_batches_duplicates_and_unmapped_diagnoses():
    cache = ROOT/'outputs/cache'
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        root = Path(directory)
        for batch in BATCHES:
            (root/batch/'data').mkdir(parents=True)
        # A repeated subject/session must be reported, not silently deduplicated.
        for batch in BATCHES[:2]:
            with (root/batch/'data/images.csv').open('w',newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['Subject','Visit','Group','Age','Sex'])
                writer.writeheader()
                writer.writerow(dict(Subject='001_S_0001',Visit='sc',Group='MCI',Age='60',Sex='F'))
        index = root/'indices'
        index.mkdir()
        for role,value in (('train',0),('val',1),('test',2)):
            (index/f'{role}.index').write_text(str(value)+'\n')
        report, rows = inspect_cohort(root,index)
        assert len(report['batches']) == 7 and len(rows) == 2
        assert report['unique_scan_ids'] == 1
        assert report['cross_batch_duplicate_scans'] == {'sub-001S0001/ses-sc':['ADNI_1','ADNI_2']}
        assert all(r['source_diagnoses']=='MCI' for r in rows)
        assert report['batches'][-1]['aal116_timeseries_files'] == 0
        assert not report['paper_cohort_identity_verified']
