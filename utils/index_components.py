"""Формулы ячейки 6 index_0_finance.ipynb без итогового взвешенного индекса."""

import numpy as np
import pandas as pd

SCORE_COLUMNS = ["media_score", "awards_score", "expertise_score", "face_score", "communication_score"]
LEGACY_COLUMNS = ["notebook_higher_education", "notebook_degree", "notebook_publications",
                  "notebook_research_experience", "notebook_patents"]
NOMINATION_COLUMNS = [f"Nom_{year}" for year in range(2015, 2025)]


def nonnegative(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)


def log_capped_score(series, quantile=0.99):
    transformed = np.log1p(nonnegative(series))
    positive = transformed[transformed > 0]
    if positive.empty:
        return pd.Series(0.0, index=series.index)
    return (transformed / positive.quantile(quantile)).clip(0, 1)


def positive_log_score(series):
    transformed = np.log1p(nonnegative(series))
    return transformed / transformed.max() if transformed.max() > 0 else transformed


def observed_minmax_score(series):
    values = pd.to_numeric(series, errors="coerce")
    observed = values.dropna()
    if observed.empty or observed.max() == observed.min():
        return pd.Series(0.0, index=series.index)
    return ((values - observed.min()) / (observed.max() - observed.min())).fillna(0).clip(0, 1)


def calculate_person_components(people):
    """Одна общая нормировка ДО отбора CEO; все баллы сразу в шкале 0–100."""
    df = people.copy()
    df["yandex_count_original"] = nonnegative(df["yandex_count"])
    df["yandex_count"] = df.groupby("person_id")["yandex_count_original"].transform("max")
    df["media_score"] = 100 * log_capped_score(df["yandex_count"])
    df["awards_raw"] = sum(
        df[col].fillna("").astype(str).str.strip().str.upper().map({"N": 1.0, "W": 2.0}).fillna(0)
        for col in NOMINATION_COLUMNS
    )
    df["awards_score"] = 100 * positive_log_score(df["awards_raw"])
    x = {col: nonnegative(df[col]) for col in LEGACY_COLUMNS}
    df["expertise_score"] = 100 * (
        .30 * x["notebook_higher_education"].clip(0, 1)
        + .25 * x["notebook_degree"].clip(0, 1)
        + .20 * (x["notebook_publications"] / 4).clip(0, 1)
        + .15 * (x["notebook_research_experience"] / 3).clip(0, 1)
        + .10 * (x["notebook_patents"] / 3).clip(0, 1)
    )
    df["face_score"] = 100 * observed_minmax_score(df["facial_trust_appearance_proxy"])
    return df


def communication_components(frame):
    """Использовать вероятности, если есть, иначе фактическую шкалу −1/0/+1."""
    df = frame.copy()
    if {"positive", "negative"} <= set(df):
        positive = pd.to_numeric(df.positive, errors="coerce")
        negative = pd.to_numeric(df.negative, errors="coerce")
        tone = (positive - negative).clip(-1, 1)
    elif "Тональность" in df:
        tone = pd.to_numeric(df["Тональность"], errors="coerce")
        if not tone.dropna().isin([-1, 0, 1]).all():
            raise ValueError("Неизвестные значения Тональность; ожидаются -1, 0, 1")
    else:
        raise ValueError("Нет positive/negative или Тональность")
    df["communication_tone"] = tone
    df["communication_observed"] = tone.notna().astype(int)
    df["communication_score"] = ((tone + 1) / 2).clip(0, 1).fillna(0) * 100
    return df[["ticker", "communication_tone", "communication_observed", "communication_score"]]
