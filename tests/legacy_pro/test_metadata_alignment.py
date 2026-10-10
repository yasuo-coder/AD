import torch
from BrainPromptPro.data.metadata import align_records, MetadataEncoder


def test_alignment_is_by_id_and_statistics_train_only():
    rows = [dict(subject=f'S{i}', session='init', age=str(40 + 10*i), sex='F', site_id=str(i)) for i in range(4)]
    assert align_records(rows, rows[::-1], 'fixture') == rows
    wrong = [dict(r) for r in rows]; wrong[0]['subject'] = 'wrong'
    try:
        align_records(rows, wrong, 'fixture')
    except ValueError:
        pass
    else:
        raise AssertionError('Must reject same-length wrong IDs')
    encoder = MetadataEncoder().fit(rows[:2])
    assert encoder.age_mean == 45 and encoder.age_std == 5
    values = encoder.transform(rows[2:])
    torch.testing.assert_close(values['age'], torch.tensor([3., 5.]))
    assert values['site'].tolist() == [0, 0]
    assert encoder.age_mean == 45
