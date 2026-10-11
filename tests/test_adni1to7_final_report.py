import unittest
import numpy as np
from BrainPromptPro.scripts.adni_1to7.report_final import independent_metrics


class IndependentMetricsTests(unittest.TestCase):
    def test_perfect_five_class_predictions(self):
        metrics, matrix, per_class = independent_metrics(np.arange(5), .9*np.eye(5)+.02)
        np.testing.assert_array_equal(matrix, np.eye(5))
        for value in metrics.values():
            self.assertAlmostEqual(value, 1.)
        for values in per_class.values():
            np.testing.assert_allclose(values, 1.)

    def test_majority_only_is_not_high_macro_performance(self):
        y = np.array([0, 0, 0, 0, 1, 2, 3, 4])
        probability = np.tile([.96, .01, .01, .01, .01], (len(y), 1))
        metrics, matrix, per_class = independent_metrics(y, probability)
        self.assertAlmostEqual(metrics['accuracy'], .5)
        self.assertAlmostEqual(metrics['precision_macro'], .1)
        self.assertAlmostEqual(metrics['recall_macro'], .2)
        self.assertAlmostEqual(metrics['balanced_accuracy'], .2)
        self.assertAlmostEqual(metrics['f1_macro'], 2/15)
        self.assertAlmostEqual(metrics['f1_weighted'], 1/3)
        self.assertAlmostEqual(metrics['roc_auc_macro'], .5)
        self.assertEqual(np.count_nonzero(matrix.sum(0)), 1)

    def test_invalid_probabilities_or_missing_class_are_rejected(self):
        y = np.arange(5)
        for probability in [np.full((5, 5), np.nan), np.zeros((5, 5)), np.ones((5, 4)),
                            np.tile([1.1, -.1, 0, 0, 0], (5, 1))]:
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                independent_metrics(y, probability)
        with self.assertRaises(ValueError):
            independent_metrics(np.zeros(5, dtype=int), .9*np.eye(5)+.02)


if __name__ == '__main__':
    unittest.main()
