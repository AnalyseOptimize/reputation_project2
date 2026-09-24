"""Идентификаторы компаний и сопоставление людей внутри одной компании."""

import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from unidecode import unidecode

TICKER_ALIASES = {"TCSG": "T", "YNDX": "YDEX"}
# В исходных таблицах эти варианты записаны отдельными строками одного человека.
SOURCE_NAME_ALIASES = {
    ("NVTK", "Леонид Михельсон"): "Михельсон Леонид Викторович",
    ("PRMD", "Александр Ефремов"): "Александр Игоревич Ефремов",
    ("SGZH", "Artem Zasursky"): "Artem I. Zassoursky",
}
# Раскрытие инициалов подтверждается локальной записью MRKP и ролью IR.
EDUCATION_NAME_ALIASES = {
    ("MRKP", "Сергей Александрович Терников"): "Alexandrovich T. Sergey",
}
# IVAT_person_info.json, ceo.notes: другая персона в CSV явно признана ошибкой.
# Не переносить её фото/медиа на Станислава Иодковского.
SOURCE_EXCLUSIONS = {
    ("IVAT", "Иодковский Эдмунд Феликсович"): "IVAT JSON: ошибочная идентичность, не CEO Станислав Иодковский",
}
TRANSLIT_ALIASES = {
    "alexey": "aleksei", "aleksey": "aleksei", "alexei": "aleksei",
    "sergey": "sergei", "andrey": "andrei", "dmitry": "dmitrii",
    "dmitriy": "dmitrii", "dmitri": "dmitrii", "yevgeniy": "evgenii",
    "evgeniy": "evgenii", "yevgeny": "evgenii", "evgeny": "evgenii",
    "eugene": "evgenii", "yuri": "iurii", "yuriy": "iurii",
    "yurii": "iurii", "iuri": "iurii", "pyotr": "petr", "piotr": "petr",
    "nikolay": "nikolai", "ilya": "ilia", "mihail": "mikhail",
    "artyom": "artem", "gennady": "gennadii", "gennadiy": "gennadii",
    "valery": "valerii", "valeriy": "valerii", "vitaly": "vitalii",
    "vitaliy": "vitalii", "anatoliy": "anatolii", "anatoly": "anatolii",
    "yulia": "iuliia", "yuliya": "iuliia", "julia": "iuliia",
    "natalia": "nataliia", "natalya": "nataliia", "maria": "mariia",
    "mariya": "mariia",
    "alexander": "aleksandr", "aleksander": "aleksandr", "maxim": "maksim",
    "peter": "petr", "georgy": "georgii", "yun": "iun",
}


def clean_name(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def canonical_ticker(value, company="") -> str:
    # В трёх исходных таблицах Henderson уже ошибочно обозначен HEAD.
    # Исправление возможно по независимому полю Company.
    if re.search(r"эйч\s*эф\s*джи|х[эе]ндерсон|henderson", str(company), re.I):
        return "HNFG"
    parts = {TICKER_ALIASES.get(p.strip().upper(), p.strip().upper())
             for p in re.split(r"[,;/|]", str(value)) if p.strip()}
    if len(parts) != 1 or next(iter(parts)) in {"NAN", "NONE", ""}:
        raise ValueError(f"Неоднозначный или пустой тикер: {value!r}")
    return parts.pop()


def name_variants(value) -> list[str]:
    name = clean_name(value)
    variants = [name, *re.split(r"\s*/\s*", name), *re.findall(r"\(([^()]+)\)", name),
                re.sub(r"\([^)]*\)", "", name)]
    return list(dict.fromkeys(v.strip() for v in variants if v.strip()))


def name_tokens(name, *, core=False) -> list[str]:
    text = unidecode(name.replace("ё", "е").replace("Ё", "Е")).lower()
    text = text.replace("'", "").replace("`", "")
    tokens = [TRANSLIT_ALIASES.get(t, t) for t in re.findall(r"[a-z0-9]+", text)]
    if core and len(tokens) >= 3:
        shorter = [t for t in tokens if not re.search(r"(?:ovich|evich|ovitch|evitch|ovna|evna|ichna|kyzy|ogly|oglu)$", t)
                   and t not in {"ilich", "ilyich"}]
        if len(shorter) >= 2:
            tokens = shorter
    return tokens


def name_keys(value, *, core=False) -> set[str]:
    return {" ".join(sorted(name_tokens(v, core=core))) for v in name_variants(value)} - {""}


def name_similarity(left, right) -> float:
    return max((fuzz.ratio(" ".join(sorted(name_tokens(a, core=True))),
                           " ".join(sorted(name_tokens(b, core=True))))
                for a in name_variants(left) for b in name_variants(right)), default=0.0)


def match_people(education: pd.DataFrame, base: pd.DataFrame,
                 threshold=85.0, margin=5.0) -> pd.DataFrame:
    """Взаимно-однозначное сопоставление; неоднозначные пары сохраняются отдельно."""
    report = []
    for ticker, group in education.groupby("ticker", sort=True):
        candidates = base.loc[base.ticker.eq(ticker), "name"].tolist()
        used = set()
        pending = list(group.index)
        for idx in pending[:]:
            name = education.at[idx, "name"]
            alias = EDUCATION_NAME_ALIASES.get((ticker, name))
            if alias in candidates:
                report.append({"ticker": ticker, "education_name": name, "matched_name": alias,
                               "method": "reviewed_alias", "score": 100.0})
                used.add(alias)
                pending.remove(idx)
        for core in (False, True):
            for idx in pending[:]:
                name = education.at[idx, "name"]
                hits = [c for c in candidates if c not in used and name_keys(name, core=core) & name_keys(c, core=core)]
                if len(hits) == 1:
                    candidate = hits[0]
                    # Не выбирать одного из двух однофамильцев по порядку строк.
                    reverse = [i for i in pending if name_keys(education.at[i, "name"], core=core)
                               & name_keys(candidate, core=core)]
                    if len(reverse) != 1:
                        continue
                    report.append({"ticker": ticker, "education_name": name, "matched_name": candidate,
                                   "method": "core_exact" if core else "exact", "score": 100.0})
                    used.add(candidate)
                    pending.remove(idx)
        # Дополнительные западные имена: Eric Hugh John Stoyll / Eric Stoyll.
        def contains_name(left, right):
            return any(len(a) >= 2 and len(b) >= 2 and (a <= b or b <= a)
                       for a in (set(name_tokens(v, core=True)) for v in name_variants(left))
                       for b in (set(name_tokens(v, core=True)) for v in name_variants(right)))
        for idx in pending[:]:
            name = education.at[idx, "name"]
            hits = [c for c in candidates if c not in used and contains_name(name, c)]
            if len(hits) == 1 and sum(contains_name(education.at[i, "name"], hits[0]) for i in pending) == 1:
                report.append({"ticker": ticker, "education_name": name, "matched_name": hits[0],
                               "method": "unique_name_subset", "score": 100.0})
                used.add(hits[0])
                pending.remove(idx)
        remaining = [c for c in candidates if c not in used]
        accepted = {}
        if pending and remaining:
            scores = np.array([[name_similarity(education.at[i, "name"], c) for c in remaining] for i in pending])
            ii, jj = linear_sum_assignment(-scores)
            for i, j in zip(ii, jj):
                score = scores[i, j]
                row_second = max(np.delete(scores[i], j), default=0)
                col_second = max(np.delete(scores[:, j], i), default=0)
                if score >= threshold and score - max(row_second, col_second) >= margin:
                    accepted[pending[i]] = (remaining[j], float(score))
        for idx in pending:
            match, score = accepted.get(idx, ("", 0.0))
            report.append({"ticker": ticker, "education_name": education.at[idx, "name"],
                           "matched_name": match, "method": "fuzzy" if match else "unmatched", "score": score})
    return pd.DataFrame(report)


def ceo_position(value) -> bool:
    """CEO по названию роли; заместители/помощники не становятся CEO."""
    for part in re.split(r"[;|]", clean_name(value).casefold()):
        if re.search(r"заместител|зам\.|deputy|assistant|помощник|советник|бывш|former", part):
            continue
        if re.search(r"\bceo\b|chief executive officer|генеральн\w* директор|председатель правления", part):
            return True
    return False
