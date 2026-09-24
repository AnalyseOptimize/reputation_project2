import unittest

import numpy as np
import pandas as pd
from scipy import stats

from utils.index_regression import FINANCE_NAMES, bh_adjust, fit_ols, prepare_finance


class RegressionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        self.data = pd.DataFrame({'x': rng.normal(size=30), 'z': rng.normal(size=30)})
        self.data['y'] = .5 + .2 * self.data.x + rng.normal(size=30) * (.5 + self.data.x.abs())

    def test_hc3_and_press_against_original_formula_and_refits(self):
        d = self.data
        fit = fit_ols(d, 'y', ['x', 'z'])
        x = np.column_stack([np.ones(len(d)), d[['x', 'z']]])
        y = d.y.to_numpy()
        bread = np.linalg.inv(x.T @ x)
        beta = bread @ x.T @ y
        h = np.einsum('ij,jk,ik->i', x, bread, x)
        e = y - x @ beta
        vcov = bread @ (x.T @ (x * (e / (1 - h))[:, None] ** 2)) @ bread
        np.testing.assert_allclose(fit['table'].coef, beta, rtol=1e-10)
        np.testing.assert_allclose(fit['table'].se_hc3, np.sqrt(np.diag(vcov)), rtol=1e-10)
        np.testing.assert_allclose(fit['table'].p_value, 2 * stats.t.sf(np.abs(beta / np.sqrt(np.diag(vcov))), 27))
        predictions = [x[i] @ np.linalg.lstsq(np.delete(x, i, axis=0), np.delete(y, i), rcond=None)[0]
                       for i in range(len(d))]
        np.testing.assert_allclose(fit['observations'].loo_prediction, predictions, atol=1e-10)

    def test_identification_and_missingness(self):
        d = self.data.assign(constant=1., duplicate=self.data.x)
        d.loc[0, 'z'] = np.nan
        fit = fit_ols(d, 'y', ['x', 'z', 'constant'])
        self.assertEqual(fit['n'], 29)
        self.assertEqual(fit['dropped_constant'], ['constant'])
        with self.assertRaises(ValueError):
            fit_ols(d, 'y', ['x', 'duplicate'])

    def test_singleton_dummy_does_not_fabricate_hc3_or_loocv(self):
        d = self.data.assign(group=['singleton'] + ['other'] * 29)
        fit = fit_ols(d, 'y', ['x'], ['group'])
        self.assertTrue(fit['table'].se_hc3.isna().all())
        self.assertTrue(np.isnan(fit['cv_r2']))
        self.assertNotEqual(fit['inference_status'], 'ok')

    def test_bh_preserves_missing_and_corrects_family(self):
        np.testing.assert_allclose(bh_adjust([.01, np.nan, .04, .03]), [.03, np.nan, .04, .04], equal_nan=True)

    def test_finance_tickers_preserve_distinct_companies(self):
        d = pd.DataFrame(1., index=range(4), columns=list(FINANCE_NAMES))
        d['Тикер биржевой'] = ['HEAD', 'HNFG', 'T, TCSG', 'YDEX, YNDX']
        d['Краткое наименование'] = ['HeadHunter', 'Henderson', 'T', 'Yandex']
        d['Вид деятельности/отрасль'] = 'Test'
        d.loc[0, '2024, Активы, рубли'] = 0
        result = prepare_finance(d)
        self.assertEqual(set(result.index), {'HEAD', 'HNFG', 'T', 'YDEX'})
        self.assertTrue(np.isnan(result.loc['HEAD', 'log_assets_2024']))
        d.loc[1, 'Тикер биржевой'] = 'HEAD'
        with self.assertRaises(ValueError):
            prepare_finance(d)


if __name__ == '__main__':
    unittest.main()
