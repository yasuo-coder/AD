import itertools
import torch
from .common import fixture


def test_required_shapes_and_all_ablations():
    for flags in itertools.product((False,True),repeat=3):
        model,inputs,_=fixture(flags=flags)
        output=model(**inputs)
        assert output['A_brain'].shape==(4,116,116)
        assert output['roi_features'].shape==(4,116,16)
        assert output['subject_repr'].shape==(4,16)
        assert output['fused_repr'].shape==(4,16)
        assert output['logits'].shape==(4,5)
        if flags[1]:
            assert output['roi_attention'].shape==(4,5,116)
        if flags[2]:
            assert output['relation_weights'].shape==(4,3)
            assert all(a.shape==(4,4) for a in output['population_adjacencies'].values())
        output['logits'].square().mean().backward()


def test_single_subject_dynamic_class_count_and_global_gate():
    for classes in (2,7):
        m,x,_=fixture(batch=1,nodes=8,classes=classes,gate='global')
        out=m(**x)
        assert out['logits'].shape==(1,classes)
        assert out['roi_attention'].shape==(1,classes,8)
        torch.testing.assert_close(out['relation_weights'].sum(-1),torch.ones(1))
        out['logits'].sum().backward()
