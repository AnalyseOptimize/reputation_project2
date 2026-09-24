"""Финансовые спецификации исходного ноутбука: OLS, HC3, PRESS и FDR."""

import numpy as np
import pandas as pd
from scipy import stats

from .index_matching import canonical_ticker

CONTROLS = ["log_assets_2024", "age", "leverage_2024", "fixed_assets_ratio_2024", "state_2024"]
OUTCOMES = {"roa_2025": "roa_2024", "stock_return_2025": "stock_return_2024"}
FINANCE_NAMES = {
    "Краткое наименование": "company_finance", "Возраст компании, лет": "age",
    "Вид деятельности/отрасль": "industry_raw", "Тикер биржевой": "ticker_raw",
}
for _year in (2024, 2025):
    for _source, _target in {
        "Среднесписочная численность работников": "employees", "Основные средства, рубли": "fixed_assets",
        "Активы, рубли": "assets", "Совокупный долг, рубли": "debt",
        "Рентабельность активов (ROA)": "roa", "Доходность акций": "stock_return",
        "Дамми на гос. Собственность": "state", "Финансовый рычаг": "leverage", "ОС/Активы": "fixed_assets_ratio",
    }.items():
        FINANCE_NAMES[f"{_year}, {_source}"] = f"{_target}_{_year}"


def prepare_finance(frame):
    missing = set(FINANCE_NAMES) - set(frame)
    if missing:
        raise ValueError(f"Не найдены финансовые колонки: {sorted(missing)}")
    data = frame.rename(columns=FINANCE_NAMES).copy()
    if data.ticker_raw.isna().any():
        raise ValueError("Пустой тикер в финансовом файле")
    data["ticker"] = data.ticker_raw.map(canonical_ticker)
    if data.ticker.duplicated().any() or data.ticker.str.contains(",", regex=False).any():
        raise ValueError("Неоднозначные или повторяющиеся финансовые тикеры")
    for col in ["age", *[v for v in FINANCE_NAMES.values() if v.endswith(("2024", "2025"))]]:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if np.isinf(data[col]).any():
            raise ValueError(f"Бесконечное значение: {col}")
    for year in (2024, 2025):
        if not data[f"state_{year}"].dropna().isin([0, 1]).all():
            raise ValueError("Государственная собственность должна быть 0/1")
    for source in ("assets", "employees"):
        data[f"log_{source}_2024"] = np.log(data[f"{source}_2024"].where(data[f"{source}_2024"] > 0))
    return data.set_index("ticker").sort_index()


def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    valid = np.isfinite(p)
    adjusted = np.full(len(p), np.nan)
    values = p[valid]
    if not len(values):
        return adjusted
    if ((values < 0) | (values > 1)).any():
        raise ValueError("p-value вне 0–1")
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    restored = np.empty_like(ranked)
    restored[order] = ranked
    adjusted[valid] = restored
    return adjusted


def fit_ols(data, outcome, predictors, categorical=()):
    """Как исходный ноутбук, но SVD вместо X'X; вырожденные случаи не маскируются.

    HC3 и t_{n-rank}; PRESS применим при h_ii < 1 и фиксированной матрице X.
    Постоянные регрессоры удаляются и явно перечисляются. Прочая коллинеарность — ошибка.
    """
    required = [outcome, *predictors, *categorical]
    if len(required) != len(set(required)):
        raise ValueError("Повторяющиеся переменные в спецификации")
    sample = data[required].dropna().copy()
    if not len(sample):
        raise ValueError("Нет полных наблюдений")
    x = pd.get_dummies(sample[[*predictors, *categorical]], columns=list(categorical), drop_first=True, dtype=float)
    dropped = [col for col in x if x[col].nunique() <= 1]
    x = x.drop(columns=dropped).astype(float)
    x.insert(0, "Intercept", 1.)
    a, y = x.to_numpy(), sample[outcome].to_numpy(float)
    if not np.isfinite(a).all() or not np.isfinite(y).all():
        raise ValueError("Бесконечное значение в регрессии")
    beta, _, rank, _ = np.linalg.lstsq(a, y, rcond=None)
    n, p = a.shape
    if rank < p:
        raise ValueError(f"Линейная зависимость регрессоров: N={n}, rank={rank}, p={p}")
    if n <= rank + 2:
        raise ValueError(f"Слишком мало наблюдений: N={n}, rank={rank}")
    inverse = np.linalg.pinv(a)
    residuals = y - a @ beta
    leverage = np.einsum("ij,ji->i", a, inverse)
    safe = (1 - leverage) > 1e-10
    df_resid = n - rank
    se = np.full(p, np.nan)
    press = np.full(n, np.nan)
    if safe.all():
        press = residuals / (1 - leverage)
        covariance = (inverse * press[None, :] ** 2) @ inverse.T
        se = np.sqrt(np.maximum(np.diag(covariance), 0))
    t = np.divide(beta, se, out=np.full(p, np.nan), where=se > 0)
    critical = stats.t.ppf(.975, df_resid)
    sse = float(residuals @ residuals)
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - sse / sst if sst > 0 else np.nan
    cv_sse = np.sum(press ** 2)
    table = pd.DataFrame({"term": x.columns, "coef": beta, "se_hc3": se, "t": t,
                          "p_value": 2 * stats.t.sf(np.abs(t), df_resid),
                          "ci_low": beta - critical * se, "ci_high": beta + critical * se})
    observations = pd.DataFrame({"actual": y, "fitted": a @ beta, "residual": residuals,
                                 "leverage": leverage, "loo_prediction": y - press}, index=sample.index)
    return {"table": table, "observations": observations, "n": n, "rank": rank, "df_resid": df_resid,
            "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / df_resid,
            "sse": sse, "cv_r2": 1 - cv_sse / sst if sst > 0 else np.nan,
            "cv_rmse": np.sqrt(cv_sse / n), "dropped_constant": dropped,
            "inference_status": "ok" if safe.all() else "undefined_hc3_and_press_leverage_one",
            "max_leverage": float(leverage.max()), "condition_number": float(np.linalg.cond(a))}


def regression_specs(outcome, lag, terms):
    employee_controls = ["log_employees_2024" if c == "log_assets_2024" else c for c in CONTROLS]
    return [
        (outcome, "M1", terms, []),
        (outcome, "M2", [*terms, lag], []),
        (outcome, "M3", [*terms, lag, *CONTROLS], []),
        (outcome, "M4", [*terms, lag, *CONTROLS], ["industry_group"]),
        (outcome, "M5", [*terms, lag, *employee_controls], []),
        (lag, "C2024", [*terms, *CONTROLS], []),
        (outcome + "_w", "W2025", [*terms, lag, *CONTROLS], []),
    ]
