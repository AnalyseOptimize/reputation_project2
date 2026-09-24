"""Расчёт каждой компоненты внутри её источника, без общей таблицы людей."""

import re

import numpy as np
import pandas as pd

from .education_features import company_aliases, company_key
from .index_components import NOMINATION_COLUMNS

DEFAULT_MEDIA_UPPER = 100_000.0
DEFAULT_FACE_LOWER = -3.0
DEFAULT_FACE_UPPER = 3.0
AWARDS_UPPER = 2.0 * len(NOMINATION_COLUMNS)


def fixed_score(values, lower, upper, *, log=False):
    """Нормировка до среднего; пропуски сохраняются, искусственные пороги фиксированы."""
    if not np.isfinite([lower, upper]).all() or lower >= upper or (log and lower < 0):
        raise ValueError("Некорректные границы шкалы")
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if np.isinf(numeric).any():
        raise ValueError("Бесконечный показатель")
    clipped = numeric.clip(lower, upper)
    if log:
        return 100 * (np.log1p(clipped) - np.log1p(lower)) / (np.log1p(upper) - np.log1p(lower))
    return 100 * (clipped - lower) / (upper - lower)


def source_ceo(position, ticker, company):
    """Определение CEO только по должности и компании в текущем источнике.

    Голое CEO/гендиректор относится к компании строки. Если работодатель явно
    указан в должности, он должен совпадать с компанией строки/её алиасом.
    """
    role = str(position or "").casefold().replace("ё", "е")
    aliases = company_aliases({"company_name": str(company)}, ticker)
    pattern = r"\bceo\b|chief executive officer|генеральн\w* директор|председатель правления|главный исполнительный директор"
    for clause in re.split(r"[;|,]", role):
        for match in re.finditer(pattern, clause):
            prefix = clause[:match.start()]
            if re.search(r"заместител|зам\.|deputy|assistant|помощник|советник|аппарата|бывш|former", prefix):
                continue
            suffix = clause[match.end():].strip()
            suffix = re.split(r"\s+(?:с|до|since|until)\s+|\s+[—–]\s+", suffix)[0]
            suffix = re.sub(r"\([^)]*\)", "", suffix).strip(" .-—–")
            if not suffix:
                return True
            suffix = re.sub(r"^(?:группы|групп[аы] компаний)\s+", "", suffix)
            suffix = re.sub(r"^банка\s+", "банк ", suffix)
            if company_key(suffix) in aliases:
                return True
    return False


def source_people(frame, metric_columns):
    """Дубли должностей одного человека схлопываются ТОЛЬКО внутри таблицы."""
    data = frame.copy()
    data["is_ceo"] = [source_ceo(p, t, c) for p, t, c in
                      zip(data.Position.fillna(""), data.ticker, data.Company)]
    grouped = data.groupby(["ticker", "name"], as_index=False, sort=False)
    values = grouped[metric_columns].first()
    roles = grouped.agg(is_ceo=("is_ceo", "max"),
                        source_positions=("Position", lambda x: " | ".join(dict.fromkeys(x.dropna().astype(str)))))
    return values.merge(roles, on=["ticker", "name"], validate="one_to_one")


def calculate_source(frame, component, *, media_upper=DEFAULT_MEDIA_UPPER,
                     face_lower=DEFAULT_FACE_LOWER, face_upper=DEFAULT_FACE_UPPER):
    """Нормированные персональные значения ровно из одного источника."""
    columns = {"media_score": ["yandex_count"], "face_score": ["facial_trust_appearance_proxy"],
               "awards_score": NOMINATION_COLUMNS}[component]
    data = source_people(frame, columns)
    if component == "media_score":
        values = pd.to_numeric(data.yandex_count, errors="raise")
        if values.dropna().lt(0).any():
            raise ValueError("Отрицательное количество медиаупоминаний")
        data[component] = fixed_score(values, 0, media_upper, log=True)
    elif component == "face_score":
        data[component] = fixed_score(data.facial_trust_appearance_proxy, face_lower, face_upper)
    else:
        raw = pd.Series(0., index=data.index)
        for column in columns:
            nominations = data[column].fillna("").astype(str).str.strip().str.upper()
            if not nominations.isin(["", "N", "W"]).all():
                raise ValueError(f"Неизвестное обозначение награды: {column}")
            raw += nominations.map({"": 0, "N": 1, "W": 2})
        data["awards_raw"] = raw
        data[component] = fixed_score(raw, 0, AWARDS_UPPER, log=True)
    return data


def source_average(people, component, *, ceo_only=False):
    selected = people.loc[people.is_ceo] if ceo_only else people
    grouped = selected.groupby("ticker")[component]
    return grouped.mean().rename(component), grouped.count().rename(component + "_observed_count")


def communication_source(frame):
    """Коммуникация уже на уровне компании; пропуск не является отрицательной тональностью."""
    if {"positive", "negative"} <= set(frame):
        positive = pd.to_numeric(frame.positive, errors="raise")
        negative = pd.to_numeric(frame.negative, errors="raise")
        if not positive.dropna().between(0, 1).all() or not negative.dropna().between(0, 1).all():
            raise ValueError("Вероятности тональности вне 0–1")
        tone = positive - negative
    else:
        tone = pd.to_numeric(frame["Тональность"], errors="raise")
        if not tone.dropna().isin([-1, 0, 1]).all():
            raise ValueError("Неизвестная категория тональности")
    data = pd.DataFrame({"ticker": frame.ticker, "communication_score": fixed_score(tone, -1, 1)})
    # Если появятся несколько обращений, каждое наблюдаемое обращение имеет одинаковый вес.
    return data.groupby("ticker").communication_score.mean(), data.groupby("ticker").communication_score.count()
