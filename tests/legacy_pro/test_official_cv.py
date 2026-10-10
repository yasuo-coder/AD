import unittest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from BrainPromptPro.official_cv import audit_splits, epoch_pass, make_official_splits, stop_reason


class OfficialProtocolTest(unittest.TestCase):
    def test_official_splits_cover_scans_but_audit_subject_overlap(self):
        labels = np.repeat(np.arange(5), 40)
        records = [dict(subject=str(i//2), label=int(y)) for i,y in enumerate(labels)]
        splits = make_official_splits(labels)
        self.assertEqual(splits, make_official_splits(labels))
        audit = audit_splits(splits, records)
        self.assertEqual(sorted(i for s in splits for i in s['test']), list(range(200)))
        self.assertTrue(sum(a['test_scans_with_train_subject'] for a in audit) > 0)
        broken = [{k:list(v) for k,v in s.items()} for s in splits]
        broken[0]['test'][0] = broken[0]['train'][0]
        with self.assertRaises(ValueError):
            audit_splits(broken, records)

    def test_strict_lr_floor_and_no_validation_patience_stop(self):
        params = dict(min_lr=1e-6, max_time=60, epochs=1000)
        self.assertIsNone(stop_reason(299, 1e-6, 300, params))
        self.assertEqual(stop_reason(299, 5e-7, 300, params), 'learning_rate_below_minimum')
        self.assertEqual(stop_reason(999, 5e-5, 300, params), 'maximum_epochs')
        self.assertEqual(stop_reason(10, 5e-5, 21601, params), 'official_per_fold_time_limit')

    def test_loss_is_batch_mean_not_sample_weighted(self):
        class Fixed(nn.Module):
            config = dict(use_adaptive_brain_graph=False)
            def forward(self, fc, **kwargs):
                return dict(logits=fc, disease_logits=fc, population_adjacencies={})
        loader = []
        for logits in [torch.tensor([[2.,0.],[2.,0.]]), torch.tensor([[0.,2.]])]:
            loader.append(dict(fc=logits, label=torch.zeros(len(logits),dtype=torch.long),
                               subject_prompt=logits, age=logits, sex=logits, site=logits))
        weights = dict(lambda_disease=1., lambda_brain_sparse=0., lambda_pop_sparse=0., lambda_relation_div=0.)
        result, _ = epoch_pass(Fixed(), loader, 'cpu', weights)
        losses = [2 * F.cross_entropy(b['fc'], b['label']).item() for b in loader]
        self.assertAlmostEqual(result['loss'], sum(losses)/2)
        self.assertNotAlmostEqual(result['loss'], (2*losses[0]+losses[1])/3)
        self.assertAlmostEqual(result['accuracy'], 2/3)


if __name__ == '__main__':
    unittest.main()
