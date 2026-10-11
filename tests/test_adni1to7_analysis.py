import copy
import random
import unittest
from BrainPromptPro.scripts.adni_1to7.analyze_ablation import analyze, METRICS, CLASSES


class PairedAblationTests(unittest.TestCase):
    def matrix(self):
        switches=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
        rows=[]
        for seed,offset in [(42,0),(3407,.05),(2026,-.07)]:
            for i,(p2,p3,p5) in enumerate(switches):
                value=.5+offset+.02*p2-.01*p3+.03*p5-.04*p2*p3
                rows.append(dict(experiment=f'A{i}',seed=seed,metrics={m:value for m in METRICS},per_class_recall={c:value for c in CLASSES}))
        random.Random(13).shuffle(rows)
        return rows

    def test_interaction_and_seed_pairing(self):
        result=analyze(self.matrix())
        for module,effect in [('P2',0.),('P3',-.03),('P5',.03)]:
            metric=result['modules'][module]['mean_marginal']['f1_macro']
            self.assertAlmostEqual(metric['mean'],effect,places=12)
            self.assertAlmostEqual(metric['sample_sd'],0.,places=12)
        self.assertEqual(result['modules']['P2']['positive_f1_backgrounds'],2)
        self.assertAlmostEqual(result['full_minus_baseline']['f1_macro']['mean'],0.,places=12)

    def test_incomplete_or_duplicate_is_not_a_complete_matrix(self):
        rows=self.matrix()
        with self.assertRaises(ValueError):analyze(rows[:-1])
        with self.assertRaises(ValueError):analyze(rows+[copy.deepcopy(rows[0])])

    def test_unregistered_seed_cannot_replace_missing_seed(self):
        rows=self.matrix();rows[0]['seed']=99
        with self.assertRaises(ValueError):analyze(rows)


if __name__=='__main__':unittest.main()
