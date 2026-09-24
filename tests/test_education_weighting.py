"""Проверки нормировки, CRITIC и глобального максимума дисперсии."""

import unittest

import numpy as np
import pandas as pd

from utils.education_features import FEATURE_COLUMNS
from utils.education_weighting import (
    bootstrap_weights, capped_simplex_vertices, critic_weights, education_matrix,
    experiment, fixed_bounds, max_variance_weights, prepare_matrix,
)


class NormalizationTests(unittest.TestCase):
    def test_fixed_bounds_do_not_depend_on_sample_extremes(self):
        x = pd.DataFrame({"publications_count": [10, 20, 200], "has_mba_emba": [0, 1, 0]})
        result = prepare_matrix(x)
        np.testing.assert_allclose(result.normalized.publications_count, [.1, .2, 1])
        self.assertEqual(result.audit.loc["publications_count", "above_upper_count"], 1)
        changed = prepare_matrix(x.iloc[:2])
        np.testing.assert_allclose(changed.normalized, result.normalized.iloc[:2])

    def test_log_uses_the_same_fixed_upper_bound(self):
        x = pd.DataFrame({"publications_count": [0, 10, 100, 1000]})
        result = prepare_matrix(x, transform="log1p")
        np.testing.assert_allclose(result.normalized.publications_count, [0, np.log1p(10)/np.log1p(100), 1, 1])

    def test_missing_preserved_and_theoretical_violation_rejected(self):
        result = prepare_matrix(pd.DataFrame({"publications_count": [0, 10, np.nan]}))
        self.assertTrue(pd.isna(result.normalized.iloc[2, 0]))
        with self.assertRaises(ValueError):
            prepare_matrix(pd.DataFrame({"has_mba_emba": [0, 2]}))

    def test_clip_before_company_average(self):
        # Первые два человека одной компании: mean(clip(0,200)/100)=0.5,
        # а clip(mean(0,200))/100=1 — другой показатель.
        x = pd.DataFrame({"publications_count": [0, 200, 10]})
        result = prepare_matrix(x).normalized.groupby(pd.Series(["A", "A", "B"])).mean()
        self.assertEqual(result.loc["A", "publications_count"], .5)

    def test_cumulative_degree_coding(self):
        people = pd.DataFrame(0., index=range(4), columns=FEATURE_COLUMNS)
        for i, name in enumerate(["education_secondary", "education_bachelor", "education_master", "education_doctoral"]):
            people.loc[i, name] = 1
        x = education_matrix(people)
        self.assertEqual(x.shape[1], 21)
        np.testing.assert_equal(x.education_bachelor_or_higher.to_numpy(), [0, 1, 1, 1])
        np.testing.assert_equal(x.education_master_or_higher.to_numpy(), [0, 0, 1, 1])
        self.assertNotIn("education_level_unknown", x)


class CriticTests(unittest.TestCase):
    def test_original_paper_numeric_example(self):
        # Diakoulaki et al. (1995), tables 1 and 3; scientific numeric data.
        # Это проверка опубликованного примера, не выборочная нормировка проекта.
        raw = pd.DataFrame({"profitability": [61,20.7,16.3,9,5.4,4,-6.1,-34.6],
                            "market_share": [1.08,.26,1.98,3.29,2.77,4.12,3.52,3.31],
                            "productivity": [4.33,4.34,2.53,1.65,2.33,1.21,2.1,.98]})
        z = (raw - pd.Series({"profitability": -34.6, "market_share": .26, "productivity": .98})).divide(
            pd.Series({"profitability": 95.6, "market_share": 3.86, "productivity": 3.36}))
        weights, _ = critic_weights(z)
        np.testing.assert_allclose(weights, [.202, .481, .317], atol=.001)

    def test_signed_correlations_not_absolute_and_constants_excluded(self):
        z = pd.DataFrame({"a": [0,.5,1], "b": [0,.5,1], "c": [1,.5,0], "constant": [1,1,1]})
        w, details = critic_weights(z)
        np.testing.assert_allclose(w, [.25,.25,.5,0])
        self.assertEqual(details.loc["a", "conflict"], 2)

    def test_no_conflict_raises(self):
        with self.assertRaisesRegex(ValueError, "конфликт"):
            critic_weights(pd.DataFrame({"a": [0,1,2], "b": [0,1,2]}))


class VarianceTests(unittest.TestCase):
    def setUp(self):
        self.z = pd.DataFrame(np.random.default_rng(12).uniform(size=(40, 6)), columns=list("abcdef"))

    def test_simplex_solution_equals_largest_variance(self):
        weights, details = max_variance_weights(self.z)
        self.assertEqual(weights.sum(), 1)
        self.assertEqual(weights.gt(0).sum(), 1)
        self.assertEqual(weights.idxmax(), self.z.var().idxmax())
        self.assertAlmostEqual(details["variance"], self.z.var().max())

    def test_capped_global_solution_dominates_random_feasible_weights(self):
        weights, details = max_variance_weights(self.z, upper_bound=.4)
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertLessEqual(weights.max(), .4)
        self.assertAlmostEqual((self.z @ weights).var(), details["variance"])
        rng = np.random.default_rng(5)
        random = rng.dirichlet(np.ones(6), size=5000)
        random = random[random.max(axis=1) <= .4]
        values = np.einsum("ij,jk,ik->i", random, self.z.cov(), random)
        self.assertGreaterEqual(details["variance"] + 1e-12, values.max())
        self.assertEqual(details["vertices_evaluated"], 60)

    def test_infeasible_and_large_search(self):
        with self.assertRaises(ValueError):
            capped_simplex_vertices(3, .2)
        with self.assertRaises(ValueError):
            capped_simplex_vertices(40, .1)

    def test_ties_are_reported(self):
        _, diagnostic = max_variance_weights(pd.DataFrame({"a": [0,1], "b": [1,0]}))
        self.assertEqual(diagnostic["optimal_vertices"], 2)

    def test_reproducible_bootstrap(self):
        first, _ = bootstrap_weights(self.z, repetitions=5, seed=42)
        second, _ = bootstrap_weights(self.z, repetitions=5, seed=42)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
