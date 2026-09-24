"""Объективные веса для образовательной компоненты: CRITIC и max Var(Xw)."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import comb, floor

import numpy as np
import pandas as pd

from .education_features import FEATURE_COLUMNS

QUALITY_COLUMNS = ["education_level_unknown", "publications_count_is_lower_bound", "patents_count_is_lower_bound"]
DEGREE_COLUMNS = ["education_secondary", "education_bachelor", "education_master", "education_doctoral"]
QUANTITATIVE_COLUMNS = [
    "publications_count", "research_experience_years", "patents_count",
    "experience_years_same_sector", "experience_years_same_industry",
    "experience_years_same_industry_group", "professional_experience_years",
    "current_company_experience_years",
]
# Искусственные верхние границы заданы ДО просмотра дисперсий/подбора весов.
# Стаж складывает пересекающиеся эпизоды, поэтому 60/100 — НЕ календарный предел.
ARTIFICIAL_UPPER_BOUNDS = {
    "publications_count": 100.0,
    "patents_count": 20.0,
    "research_experience_years": 40.0,
    "experience_years_same_sector": 60.0,
    "experience_years_same_industry": 60.0,
    "experience_years_same_industry_group": 60.0,
    "professional_experience_years": 100.0,
    "current_company_experience_years": 50.0,
}
FEATURE_LABELS = {
    "education_bachelor_or_higher": "Бакалавриат и выше",
    "education_master_or_higher": "Магистратура / специалитет и выше",
    "education_secondary": "Среднее образование (высший уровень)",
    "education_bachelor": "Бакалавриат (высший уровень)",
    "education_master": "Магистратура / специалитет (высший уровень)",
    "education_doctoral": "Учёная степень",
    "has_mba_emba": "MBA / EMBA",
    "has_foreign_education": "Зарубежное образование",
    "has_professional_retraining": "Профессиональная переподготовка",
    "has_law_education": "Юридическое образование",
    "has_economics_education": "Экономическое образование",
    "publications_count": "Количество публикаций",
    "has_research_experience": "Наличие исследовательского опыта",
    "research_experience_years": "Годы исследовательского опыта",
    "patents_count": "Количество патентов",
    "experience_years_same_sector": "Годы в том же секторе",
    "experience_years_same_industry": "Годы в той же отрасли",
    "experience_years_same_industry_group": "Годы в той же отраслевой группе",
    "professional_experience_years": "Общий профессиональный стаж",
    "has_experience_same_sector": "Опыт в том же секторе",
    "has_experience_same_industry": "Опыт в той же отрасли",
    "has_experience_same_industry_group": "Опыт в той же отраслевой группе",
    "has_other_company_management": "Управление другой компанией",
    "current_company_experience_years": "Стаж в текущей компании",
}


def education_matrix(frame: pd.DataFrame, *, encoding="cumulative") -> pd.DataFrame:
    """Матрица критериев по компаниям; метаданные и старый expertise_score исключены."""
    columns = [col for col in FEATURE_COLUMNS if col not in QUALITY_COLUMNS]
    x = frame[columns].apply(pd.to_numeric, errors="raise").astype(float)
    if encoding == "original":
        return x
    if encoding != "cumulative":
        raise ValueError("encoding должен быть cumulative или original")
    degrees = pd.DataFrame({
        "education_bachelor_or_higher": x[["education_bachelor", "education_master", "education_doctoral"]].sum(axis=1, min_count=3),
        "education_master_or_higher": x[["education_master", "education_doctoral"]].sum(axis=1, min_count=2),
        "education_doctoral": x.education_doctoral,
    }, index=x.index)
    return pd.concat([degrees, x.drop(columns=DEGREE_COLUMNS)], axis=1)


@dataclass
class PreparedMatrix:
    normalized: pd.DataFrame
    audit: pd.DataFrame


def fixed_bounds(columns, *, multiplier=1.0) -> pd.DataFrame:
    if multiplier <= 0:
        raise ValueError("Множитель искусственных границ должен быть положительным")
    rows = []
    for col in columns:
        if col not in FEATURE_LABELS:
            raise ValueError(f"Не заданы границы показателя {col}")
        artificial = col in ARTIFICIAL_UPPER_BOUNDS
        rows.append({"feature": col, "lower": 0.0,
                     "upper": ARTIFICIAL_UPPER_BOUNDS[col] * multiplier if artificial else 1.0,
                     "bound_type": "artificial" if artificial else "theoretical"})
    return pd.DataFrame(rows).set_index("feature")


def prepare_matrix(x: pd.DataFrame, *, bounds=None, transform="none", missing="preserve", tolerance=1e-12) -> PreparedMatrix:
    """Шкала 0–1 по ФИКСИРОВАННЫМ границам, до усреднения людей по компании.

    Выборочные экстремумы не используются. Пропуски можно сохранить для
    последующего среднего по наблюдаемым людям; коэффициенты не заполняются ими.
    """
    if len(x) < 2 or x.columns.has_duplicates or x.index.has_duplicates:
        raise ValueError("Нужны >=2 независимых наблюдения и уникальные названия строк/критериев")
    if transform not in {"none", "log1p"} or missing not in {"preserve", "zero", "raise"}:
        raise ValueError("Неизвестный способ преобразования/обработки пропусков")
    values = x.astype(float).copy()
    if np.isinf(values.to_numpy()).any() or values.lt(0).any().any():
        raise ValueError("Критерии образования должны быть конечными и неотрицательными")
    audit = (fixed_bounds(x.columns) if bounds is None else bounds.reindex(x.columns)).copy()
    if audit[["lower", "upper"]].isna().any().any() or not np.isfinite(audit[["lower", "upper"]]).all().all():
        raise ValueError("Не для всех критериев заданы конечные границы")
    if not audit.upper.gt(audit.lower).all() or audit.lower.lt(0).any():
        raise ValueError("Нужны 0 <= lower < upper")
    audit.index.name = "feature"
    audit["missing_count"] = values.isna().sum()
    if missing == "raise" and audit.missing_count.gt(0).any():
        raise ValueError("Есть пропуски в матрице критериев")
    if missing == "zero":
        values = values.fillna(0)
    wholly_missing = values.isna().all()
    audit["below_lower_count"] = values.lt(audit.lower).sum()
    audit["above_upper_count"] = values.gt(audit.upper).sum()
    # Выход за теоретическую границу означает ошибку данных, а не выброс.
    theoretical = audit.bound_type.eq("theoretical")
    if ((audit.below_lower_count + audit.above_upper_count).gt(0) & theoretical).any():
        raise ValueError("Значения нарушают теоретические границы")
    values = values.clip(lower=audit.lower, upper=audit.upper, axis=1)
    lower, upper = audit.lower.copy(), audit.upper.copy()
    audit["transform"] = "identity"
    if transform == "log1p":
        selected = values.columns.intersection(QUANTITATIVE_COLUMNS)
        values[selected] = np.log1p(values[selected])
        lower.loc[selected] = np.log1p(lower.loc[selected])
        upper.loc[selected] = np.log1p(upper.loc[selected])
        audit.loc[selected, "transform"] = "log1p"
    spread = upper - lower
    normalized = (values - lower).divide(spread).clip(0, 1)
    active = normalized.std(ddof=1).gt(tolerance) & ~wholly_missing
    audit["lower_transformed"] = lower
    audit["upper_transformed"] = upper
    audit["active"] = active
    audit["exclusion_reason"] = np.where(wholly_missing, "all_missing", np.where(active, "", "constant"))
    audit["std_normalized"] = normalized.std(ddof=1)
    if not active.any():
        raise ValueError("Нет варьирующих критериев: объективные веса не определены")
    return PreparedMatrix(normalized, audit)


def _active_matrix(z: pd.DataFrame, tolerance=1e-12):
    if z.shape[0] < 2 or not np.isfinite(z.to_numpy(dtype=float)).all():
        raise ValueError("Нужна конечная матрица с минимум двумя наблюдениями")
    active = z.std(ddof=1).gt(tolerance)
    if not active.any():
        raise ValueError("Все критерии постоянны")
    return z.loc[:, active]


def critic_weights(z: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """C_j = sigma_j * sum_k(1-r_jk); Pearson, НЕ |r|. См. Diakoulaki et al., 1995."""
    active = _active_matrix(z)
    correlation = active.corr(method="pearson").clip(-1, 1)
    matrix = correlation.to_numpy(copy=True)
    np.fill_diagonal(matrix, 1.0)
    correlation = pd.DataFrame(matrix, index=active.columns, columns=active.columns)
    std = active.std(ddof=1)
    conflict = (1 - correlation).sum(axis=1)
    information = std * conflict
    if information.sum() <= 1e-12:
        raise ValueError("CRITIC не определён: отсутствует конфликт критериев (sum C_j = 0)")
    weights = (information / information.sum()).reindex(z.columns, fill_value=0.0).rename("critic")
    details = pd.DataFrame({"std": std, "conflict": conflict, "information": information,
                            "weight": weights}).reindex(z.columns).fillna(0)
    details.index.name = "feature"
    return weights, details


@lru_cache(maxsize=16)
def capped_simplex_vertices(n_features: int, upper_bound: float) -> np.ndarray:
    """Все вершины {w>=0, sum(w)=1, w<=u}; глобальный максимум выпуклой квадратичной формы.

    В вершине k=floor(1/u) координат равны u, одна равна остатку (если он
    ненулевой), остальные — нулю. Ограничение размера защищает от взрыва перебора.
    """
    if not 0 < upper_bound <= 1 or n_features * upper_bound < 1 - 1e-12:
        raise ValueError("Невыполнимое ограничение верхнего веса")
    k = floor(1 / upper_bound + 1e-12)
    residual = 1 - k * upper_bound
    if abs(residual) < 1e-12:
        residual = 0.0
    count = comb(n_features, k) * (n_features - k if residual > 0 else 1)
    if count > 500_000:
        raise ValueError(f"Перебор {count:,} вершин слишком велик; измените верхнюю границу/число критериев")
    vertices = np.zeros((count, n_features), dtype=float)
    row = 0
    for chosen in combinations(range(n_features), k):
        free = [j for j in range(n_features) if j not in chosen] if residual > 0 else [None]
        for j in free:
            vertices[row, list(chosen)] = upper_bound
            if j is not None:
                vertices[row, j] = residual
            row += 1
    vertices.setflags(write=False)
    return vertices


def max_variance_weights(z: pd.DataFrame, *, upper_bound=1.0) -> tuple[pd.Series, dict]:
    active = _active_matrix(z)
    covariance = active.cov().to_numpy()
    vertices = capped_simplex_vertices(active.shape[1], float(upper_bound))
    objectives = np.einsum("ij,jk,ik->i", vertices, covariance, vertices, optimize=True)
    maximum = objectives.max()
    winner = int(np.argmax(objectives))
    ties = np.flatnonzero(np.isclose(objectives, maximum, rtol=1e-10, atol=1e-12))
    weights = pd.Series(vertices[winner], index=active.columns).reindex(z.columns, fill_value=0.0)
    weights.name = "max_variance" if upper_bound == 1 else "max_variance_capped"
    return weights, {
        "variance": float(maximum), "upper_bound": upper_bound,
        "vertices_evaluated": len(vertices), "optimal_vertices": len(ties),
        "tie_rule": "first in input feature order; ties reported, not averaged",
        "global_optimum": True,
    }


def fit_weighting(z: pd.DataFrame, *, upper_bound=.20) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    active = _active_matrix(z)
    equal = pd.Series(0.0, index=z.columns)
    equal.loc[active.columns] = 1 / active.shape[1]
    critic, critic_details = critic_weights(z)
    max_var, diagnostic = max_variance_weights(z)
    capped, capped_diagnostic = max_variance_weights(z, upper_bound=upper_bound)
    weights = pd.DataFrame({"equal": equal, "critic": critic, "max_variance": max_var, "max_variance_capped": capped})
    weights.index.name = "feature"
    return weights, critic_details, {"max_variance": diagnostic, "max_variance_capped": capped_diagnostic}


def prepare_company_education(people: pd.DataFrame, *, ceo_only=False, encoding="cumulative",
                              transform="none", bounds=None, missing="preserve"):
    """Общая подготовка для ноутбука и производственного расчёта CRITIC."""
    if people.duplicated(["ticker", "name"]).any():
        raise ValueError("Повторяющиеся люди в компании")
    selected = people.loc[people.is_ceo] if ceo_only else people
    x = education_matrix(selected.reset_index(drop=True), encoding=encoding)
    prepared = prepare_matrix(x, bounds=bounds, transform=transform, missing=missing)
    normalized = prepared.normalized.groupby(selected.ticker.to_numpy()).mean()
    normalized.index.name = "ticker"
    coverage = prepared.normalized.groupby(selected.ticker.to_numpy()).count()
    coverage.index.name = "ticker"
    # Не удалять/не заполнять молча компании без наблюдений по целому критерию.
    if normalized.isna().any().any():
        raise ValueError("У компаний есть полностью отсутствующие критерии; задайте явный missing='zero' либо исправьте данные")
    return {"normalized": normalized, "person_normalized": prepared.normalized,
            "coverage": coverage, "preprocessing": prepared.audit}


def education_critic_component(people: pd.DataFrame, *, ceo_only=False, bounds=None):
    """Нормировка -> средние компаний -> CRITIC; тот же основной сценарий, что в ноутбуке."""
    result = prepare_company_education(people, ceo_only=ceo_only, bounds=bounds)
    weights, details = critic_weights(result["normalized"])
    result.update(weights=weights, critic_details=details,
                  score=(100 * result["normalized"].dot(weights)).rename("expertise_score"))
    return result


def experiment(people: pd.DataFrame, *, ceo_only=False, encoding="cumulative", transform="none",
               upper_bound=.20, bounds=None, missing="preserve"):
    """Фиксированная нормировка людей -> средние компаний -> веса компаний."""
    prepared = prepare_company_education(people, ceo_only=ceo_only, encoding=encoding,
                                        transform=transform, bounds=bounds, missing=missing)
    normalized = prepared["normalized"]
    weights, critic_details, diagnostics = fit_weighting(normalized, upper_bound=upper_bound)
    scores = 100 * normalized.dot(weights)
    scores.index.name = "ticker"
    summary = pd.DataFrame({
        "index_variance": scores.var(ddof=1), "index_std": scores.std(ddof=1),
        "nonzero_weights": weights.gt(1e-12).sum(), "max_weight": weights.max(),
        "effective_features": 1 / weights.pow(2).sum(),
    })
    summary.index.name = "method"
    return {**prepared,
            "weights": weights, "critic_details": critic_details, "diagnostics": diagnostics,
            "scores": scores, "summary": summary, "rank_correlations": scores.corr(method="spearman")}


def bootstrap_weights(z: pd.DataFrame, *, repetitions=200, seed=42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Устойчивость CRITIC и свободного simplex-max при ФИКСИРОВАННОЙ нормировке."""
    rng = np.random.default_rng(seed)
    draws, failures = [], []
    for iteration in range(repetitions):
        sample = z.iloc[rng.integers(0, len(z), len(z))]
        for method, fitter in (("critic", critic_weights), ("max_variance", max_variance_weights)):
            try:
                weights, _ = fitter(sample)
                draws.extend({"iteration": iteration, "method": method, "feature": feature, "weight": float(value)}
                             for feature, value in weights.items())
            except ValueError as exc:
                failures.append({"iteration": iteration, "method": method, "reason": str(exc)})
    return pd.DataFrame(draws), pd.DataFrame(failures, columns=["iteration", "method", "reason"])
