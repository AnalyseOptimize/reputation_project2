import unittest

import numpy as np
import pandas as pd

from utils.education_weighting import critic_weights
from utils.index_eda import bootstrap_critic, evaluate_weights, ranking_metrics, weight_scenarios


class IndexEdaTests(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame(np.random.default_rng(1).uniform(size=(15, 5)),
                              index=[f"C{i}" for i in range(15)], columns=list("abcde"))

    def test_rank_metrics_ties_and_reversal(self):
        a = [1, 2, 2, 4]
        same = ranking_metrics(a, a, top_k=2)
        self.assertAlmostEqual(same["spearman"], 1)
        self.assertEqual(same["top_k_jaccard"], 1)
        reverse = ranking_metrics(a, [-v for v in a], top_k=2)
        self.assertAlmostEqual(reverse["kendall_tau_b"], -1)
        self.assertEqual(reverse["top_k_jaccard"], .5)
        self.assertTrue(np.isnan(ranking_metrics(a, [0] * 4)["spearman"]))

    def test_weights_and_reference_alignment(self):
        w, labels = weight_scenarios(self.x.columns, draws=5)
        np.testing.assert_allclose(w.sum(axis=1), 1)
        self.assertTrue((w >= 0).all().all())
        equal = pd.DataFrame([[.2] * 5], columns=list("edcba"))
        metrics = evaluate_weights(self.x, equal, self.x.mean(axis=1).iloc[::-1])
        self.assertAlmostEqual(metrics.spearman.iloc[0], 1)
        self.assertEqual(metrics.max_abs_rank_shift.iloc[0], 0)
        with self.assertRaises(ValueError):
            evaluate_weights(self.x.where(self.x < .9), w)

    def test_bootstrap_resamples_n_and_evaluates_original_companies(self):
        result = bootstrap_critic(self.x, repetitions=12, seed=7)
        self.assertEqual(result["audit"].sample_size.unique().tolist(), [15])
        self.assertTrue(result["audit"].unique_companies.le(15).all())
        self.assertTrue(result["audit"].status.eq("ok").all())
        self.assertEqual(len(result["company_summary"]), 15)
        indices = np.random.default_rng(7).integers(0, 15, size=15)
        expected, _ = critic_weights(self.x.iloc[indices])
        np.testing.assert_allclose(result["weights"].iloc[0], expected)
        metrics = ranking_metrics(self.x.mean(axis=1), self.x.dot(expected))
        self.assertAlmostEqual(result["metrics"].iloc[0].spearman_vs_equal, metrics["spearman"])


if __name__ == "__main__":
    unittest.main()
