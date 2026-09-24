"""Ранговая устойчивость и бутстрап весов на фиксированной матрице компаний."""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

from .education_weighting import critic_weights


def validate_matrix(x):
    if len(x) < 3 or x.shape[1] < 2 or not x.index.is_unique or not x.columns.is_unique:
        raise ValueError("Нужны >=3 компании, >=2 компоненты и уникальные ключи")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Для сравнения рейтингов нужна полная конечная матрица")
    if not x.ge(0).all().all() or not x.le(1).all().all():
        raise ValueError("Ожидается фиксированная шкала 0–1")


def ranking_metrics(reference, candidate, *, top_k=10):
    """Ранг 1 — лучший; средние ранги при равенстве, топ включает граничные ничьи."""
    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    if reference.ndim != 1 or candidate.shape != reference.shape or len(reference) < 2:
        raise ValueError("Нужны два вектора одинаковой длины >=2")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Пропуски в сравниваемых рейтингах")
    if top_k < 1:
        raise ValueError("top_k должен быть положительным")
    ranks_ref, ranks_candidate = rankdata(-reference), rankdata(-candidate)
    defined = np.ptp(reference) > 0 and np.ptp(candidate) > 0
    top_ref = rankdata(-reference, method="min") <= min(top_k, len(reference))
    top_candidate = rankdata(-candidate, method="min") <= min(top_k, len(candidate))
    shifts = np.abs(ranks_ref - ranks_candidate)
    return {
        "spearman": float(spearmanr(reference, candidate).statistic) if defined else np.nan,
        "kendall_tau_b": float(kendalltau(reference, candidate, variant="b").statistic) if defined else np.nan,
        "mean_abs_rank_shift": float(shifts.mean()),
        "max_abs_rank_shift": float(shifts.max()),
        "top_k_jaccard": float((top_ref & top_candidate).sum() / (top_ref | top_candidate).sum()),
    }


def weight_scenarios(columns, *, draws=1000, alphas=(10., 1., .2), seed=42):
    """Dirichlet вокруг равных весов + однофакторная сетка, включая исключения."""
    columns = list(columns)
    p = len(columns)
    if p < 2 or draws < 1 or any(a <= 0 for a in alphas):
        raise ValueError("Некорректная конфигурация сценариев")
    rng = np.random.default_rng(seed)
    weights, labels = [], []
    for alpha in alphas:
        sampled = rng.dirichlet(np.full(p, alpha), size=draws)
        for i, row in enumerate(sampled):
            weights.append(row)
            labels.append({"family": f"dirichlet_{alpha:g}", "varied_component": "", "target_weight": np.nan})
    for j, col in enumerate(columns):
        for share in np.unique(np.r_[np.linspace(0, 1, 21), 1 / p]):
            row = np.full(p, (1 - share) / (p - 1))
            row[j] = share
            weights.append(row)
            labels.append({"family": "one_at_a_time", "varied_component": col, "target_weight": share})
    w = pd.DataFrame(weights, columns=columns).rename_axis("scenario")
    return w, pd.DataFrame(labels, index=w.index)


def evaluate_weights(x, weights, reference=None, *, top_k=10):
    validate_matrix(x)
    if set(weights.columns) != set(x.columns):
        raise ValueError("Набор компонент весов должен совпадать с матрицей")
    weights = weights.reindex(columns=x.columns)
    array = weights.to_numpy(float)
    if not np.isfinite(array).all() or (array < 0).any() or not np.allclose(array.sum(axis=1), 1):
        raise ValueError("Веса должны быть неотрицательными и суммироваться в 1")
    baseline = x.mean(axis=1) if reference is None else reference.reindex(x.index)
    scores = x.to_numpy() @ array.T
    metrics = pd.DataFrame([ranking_metrics(baseline, scores[:, j], top_k=top_k)
                            for j in range(len(weights))], index=weights.index)
    metrics["l1_from_equal"] = np.abs(array - 1 / x.shape[1]).sum(axis=1)
    metrics["max_weight"] = array.max(axis=1)
    return metrics


def bootstrap_critic(x, *, repetitions=2000, seed=42, top_k=10):
    """n строк с возвращением; веса оцениваются на повторе, ранги — на исходных n."""
    validate_matrix(x)
    if repetitions < 1:
        raise ValueError("Нужно хотя бы одно повторение")
    point, _ = critic_weights(x)
    baseline_equal = x.mean(axis=1)
    baseline_critic = x.dot(point)
    rng = np.random.default_rng(seed)
    draws, audits = [], []
    for iteration in range(repetitions):
        indices = rng.integers(0, len(x), size=len(x))
        audit = {"iteration": iteration, "sample_size": len(x),
                 "unique_companies": len(np.unique(indices)), "status": "ok", "reason": ""}
        try:
            weight, _ = critic_weights(x.iloc[indices])
            draws.append({"iteration": iteration, **weight.to_dict()})
        except ValueError as exc:
            audit.update(status="failed", reason=str(exc))
        audits.append(audit)
    audit = pd.DataFrame(audits).set_index("iteration")
    if not draws:
        raise ValueError("CRITIC не определён ни в одном повторении бутстрапа")
    weights = pd.DataFrame(draws).set_index("iteration")[x.columns]
    scores = pd.DataFrame(100 * x.to_numpy() @ weights.to_numpy().T,
                          index=x.index, columns=weights.index)
    ranks = scores.rank(ascending=False, method="average")
    top = scores.rank(ascending=False, method="min").le(min(top_k, len(x)))
    weight_summary = pd.DataFrame({"point": point, "mean": weights.mean(), "std": weights.std(),
                                  "p025": weights.quantile(.025), "median": weights.median(),
                                  "p975": weights.quantile(.975)})
    weight_summary.index.name = "component"
    company_summary = pd.DataFrame({
        "equal_score": 100 * baseline_equal, "critic_score": 100 * baseline_critic,
        "equal_rank": baseline_equal.rank(ascending=False),
        "critic_rank": baseline_critic.rank(ascending=False),
        "score_p025": scores.quantile(.025, axis=1), "score_p975": scores.quantile(.975, axis=1),
        "rank_p025": ranks.quantile(.025, axis=1), "rank_median": ranks.median(axis=1),
        "rank_p975": ranks.quantile(.975, axis=1), "top_k_probability": top.mean(axis=1),
    })
    metrics_equal = evaluate_weights(x, weights, baseline_equal, top_k=top_k).add_suffix("_vs_equal")
    metrics_point = evaluate_weights(x, weights, baseline_critic, top_k=top_k).add_suffix("_vs_point_critic")
    return {"weights": weights, "audit": audit, "weight_summary": weight_summary,
            "company_summary": company_summary, "metrics": metrics_equal.join(metrics_point)}
