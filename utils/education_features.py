"""Расчёт признаков образования и опыта без сторонних зависимостей."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .education_rules import (
    COMPANY_ALIASES, ECONOMICS_PATTERN, FIELD_KEYS, FOREIGN_INSTITUTION_PATTERN,
    JOINT_DOMESTIC_PATTERN, LAW_PATTERN, MBA_PATTERN, RETRAINING_PATTERN,
)

COLLECTION_YEAR = 2024
LEVELS = ("secondary", "bachelor", "master", "doctoral")
LEVEL_MAP = {
    "secondary education": "secondary",
    "bachelor's degree": "bachelor",
    "master's degree": "master",
    "specialist degree": "master",
    "doctoral degree": "doctoral",
}
DIMENSIONS = ("sector", "industry", "industry_group")
FEATURE_COLUMNS = [
    *[f"education_{level}" for level in LEVELS], "education_level_unknown",
    "has_mba_emba", "has_foreign_education", "has_professional_retraining",
    "has_law_education", "has_economics_education",
    "publications_count", "publications_count_is_lower_bound",
    "has_research_experience", "research_experience_years",
    "patents_count", "patents_count_is_lower_bound",
    *[f"experience_years_same_{key}" for key in DIMENSIONS],
    "professional_experience_years",
    *[f"has_experience_same_{key}" for key in DIMENSIONS],
    "has_other_company_management", "current_company_experience_years",
]


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or ""))
                  .casefold().replace("ё", "е")).strip()


def matches(pattern: str, value: Any) -> bool:
    return bool(re.search(pattern, normalize(value)))


def records(value: Any) -> list:
    return [v for v in (value if isinstance(value, list) else [value]) if v not in (None, "", {})]


def record_text(record: Any, keys: tuple = FIELD_KEYS) -> str:
    if not isinstance(record, dict):
        return str(record or "")
    return " | ".join(str(record[k]) for k in keys if record.get(k))


def classify_field(value: str) -> tuple[int, int]:
    """Независимые признаки: смешанная программа может дать две единицы."""
    return int(matches(LAW_PATTERN, value)), int(matches(ECONOMICS_PATTERN, value))


def education_level(record: dict) -> str | None:
    category = normalize(record.get("qualification_degree"))
    text = record_text(record)
    # Нормализованная категория имеет приоритет над текстом пояснений.
    if category in LEVEL_MAP:
        return LEVEL_MAP[category]
    if category not in ("", "unknown"):
        return None
    if matches(r"незакон|незаверш|unfinished|incomplete|completion not confirmed|honorary|почетн", text):
        return None
    if matches(r"(?:кандидат|доктор)\w* .{0,60}наук|\bph\.?\s*d\.?\b|doctoral degree|doctor of (?:philosophy|science)|candidate dissertation", text):
        return "doctoral"
    if matches(MBA_PATTERN, text):
        return None
    if matches(r"магистр|магистрат|master|\bmsc\b|\bma\b|специалист|специалитет", text):
        return "master"
    if matches(r"бакалавр|bachelor|undergraduate degree", text):
        return "bachelor"
    if matches(r"среднее (?:специальное|профессиональное)|secondary education", text):
        return "secondary"
    return None


def foreign_education(record: dict) -> bool:
    country = normalize(record.get("country") or record.get("country_of_study"))
    if country:
        return country not in {"russia", "russian federation", "россия", "рф", "ru", "rus"}
    institution = record_text(record, ("institution", "university"))
    if matches(JOINT_DOMESTIC_PATTERN, institution):
        return False
    return matches(FOREIGN_INSTITUTION_PATTERN, institution)


def education_features(person: dict) -> dict:
    entries = [r for r in records(person.get("education")) + records(person.get("degrees"))
               if isinstance(r, dict)]
    levels = [education_level(r) for r in entries]
    highest = next((level for level in reversed(LEVELS) if level in levels), None)
    law, economics = classify_field(" | ".join(record_text(r) for r in entries))
    # Для bare Management применяем правило отдельно к каждому полю.
    economics = max([economics] + [classify_field(str(r.get(k) or ""))[1]
                                  for r in entries for k in FIELD_KEYS])
    return {
        **{f"education_{level}": int(level == highest) for level in LEVELS},
        "education_level_unknown": int(highest is None),
        "has_mba_emba": int(any(normalize(r.get("qualification_degree")) == "mba"
                                or matches(MBA_PATTERN, record_text(r)) for r in entries)),
        "has_foreign_education": int(any(foreign_education(r) for r in entries)),
        "has_professional_retraining": int(any(matches(
            RETRAINING_PATTERN, record_text(r, FIELD_KEYS + ("description", "details"))
        ) for r in entries)),
        "has_law_education": law,
        "has_economics_education": economics,
    }


def parse_year(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    if re.fullmatch(r"(?:18|19|20)\d{2}(?:\.0)?", text):
        return int(float(text))
    if re.fullmatch(r"(?:18|19|20)\d{2}-\d{2}(?:-\d{2})?", text):
        return int(text[:4])
    return None


def endpoint(record: dict, kind: str) -> int | None:
    for key in (f"{kind}_year", f"{kind}_date", kind):
        year = parse_year(record.get(key))
        if year is not None:
            return year
    return None


@dataclass
class Episode:
    index: int
    record: Any
    start: int | None
    end: int | None
    years: int = 0
    note: str = ""


def experience_episodes(value: Any, *, missing_dates: str = "professional",
                        as_of: int = COLLECTION_YEAR) -> list[Episode]:
    """Включительные интервалы; пересечения НЕ объединяются.

    Хронология — начало, иначе известное окончание. Полностью недатированные
    записи не имеют позиции во времени и не получают выдуманного начала.
    При одинаковом начале эпизоды параллельны; следующий — строго более поздний.
    Начало восстанавливается только по исходному окончанию предыдущей записи.
    """
    if missing_dates not in {"professional", "complete-only"}:
        raise ValueError(f"Unknown missing_dates policy: {missing_dates}")
    result = [Episode(i, r, endpoint(r, "start") if isinstance(r, dict) else None,
                      endpoint(r, "end") if isinstance(r, dict) else None)
              for i, r in enumerate(records(value))]
    ordered = sorted((e for e in result if e.start is not None or e.end is not None),
                     key=lambda e: (e.start if e.start is not None else e.end, e.index))
    original_ends = {e.index: e.end for e in result}
    if missing_dates == "professional":
        for i, e in enumerate(ordered):
            if e.start is None and i:
                previous_end = original_ends[ordered[i - 1].index]
                if previous_end is not None and previous_end <= e.end:
                    e.start = previous_end
                    e.note = "start_from_previous_end"
        starts = sorted({e.start for e in ordered if e.start is not None})
        for e in ordered:
            if e.start is not None and e.end is None:
                e.end = next((year for year in starts if year > e.start), as_of)
                e.note = "end_from_next_start" if e.end != as_of else "end_at_collection_year"
    for e in result:
        if e.start is None or e.end is None:
            e.note = "missing_dates_excluded"
        elif e.start > as_of:
            e.note = "after_collection_year_excluded"
        elif e.end < e.start:
            e.note = "reversed_interval_excluded"
        else:
            e.years = min(e.end, as_of) - e.start + 1
            if e.end > as_of:
                e.note = (e.note + ";end_capped_at_collection_year").strip(";")
    return result


LEGAL_FORM_PATTERN = (
    r"\b(?:международная компания|публичное акционерное общество|"
    r"открытое акционерное общество|акционерное общество|общество с ограниченной ответственностью|"
    r"мкпао|пао|оао|зао|ао|ооо|plc|pjsc|ojsc|jsc|llc|ltd|limited)\b"
)


def company_key(value: Any) -> str:
    text = re.sub(LEGAL_FORM_PATTERN, " ", normalize(value))
    return re.sub(r"[^\w]+", "", text)


def company_aliases(document: dict, ticker: str) -> set[str]:
    name = document.get("company_name") or ""
    # Скобки содержат алиасы/пояснения, но не являются частью юр. названия.
    names = [re.sub(r"\([^)]*\)", "", s).strip() for s in name.split("/")]
    names.extend(COMPANY_ALIASES.get(ticker, []))
    return {company_key(s) for s in names if company_key(s)}


def is_current_company(organization: Any, aliases: set[str]) -> bool:
    if not organization:
        return False
    text = re.sub(r"\([^)]*\)", "", str(organization))
    if company_key(text) in aliases:
        return True
    # Разрешить несколько названий одного работодателя. Пустые части вроде
    # "ОАО/ПАО" пропускаются, но дочернее юрлицо не становится алиасом группы.
    parts = [company_key(part) for part in text.split("/") if company_key(part)]
    return bool(parts) and all(part in aliases for part in parts)


def is_management(role: Any, organization: Any) -> bool:
    if not organization or matches(
        r"министерств|правительств|администраци|университет|институт|академи|"
        r"федеральн\w* служб|государственн\w* дум|совет федерации|university|ministry", organization
    ):
        return False
    if matches(r"^(?:советник|помощник|ассистент|advis[eo]r|assistant)\b", role):
        return False
    return matches(
        r"совет\w* директор|наблюдательн\w* совет|правлени|директор|президент|"
        r"управляющ|руководител|начальник|\bboard\b|\bdirector\b|\bchair\w*|"
        r"\b(?:ceo|cfo|coo|cto|cio|president)\b|chief .{0,35}officer|"
        r"\b(?:general|managing) (?:manager|partner)\b|\bhead of\b", role
    )


# Агрегаты могут содержать итоги и подмножества: "502 работ, включая 29 ...".
# Берём итог до including/включая; самостоятельные категории суммируются.
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "одном": 1, "один": 1, "одной": 1, "две": 2, "двух": 2}


def aggregate_count(text: str, kind: str) -> tuple[int | None, bool]:
    text = normalize(text)
    text = re.split(r"\bincluding\b|включая|из них|of which|including", text)[0]
    nouns = (r"(?:научн\w*\s+и\s+учебно-методическ\w*\s+|научн\w*\s+и\s+методическ\w*\s+|"
             r"научн\w*\s+|scientific\s+)?(?:публикаци\w*|работ\w*|стат\w*|книг\w*|доклад\w*|publications?|articles?|books?)"
             if kind == "publications" else r"(?:патент\w*|patents?|изобретени\w*|inventions?|авторских свидетельств)")
    number = r"\d+|" + "|".join(NUMBER_WORDS)
    hits = re.findall(r"\b(" + number + r")\s+" + nouns, text)
    values = [int(s) if s.isdigit() else NUMBER_WORDS[s] for s in hits]
    return (sum(values) if values else None,
            matches(r"более|больше|свыше|\bover\b|more than|at least|не менее|approximately|около", text))


def scientific_count(value: Any, kind: str) -> tuple[int | None, int]:
    entries = records(value)
    if not entries:
        return 0, 0
    individual = set()
    totals = []
    unknown = False
    for entry in entries:
        title = normalize(entry.get("title")) if isinstance(entry, dict) else ""
        description = normalize(entry.get("description")) if isinstance(entry, dict) else normalize(entry)
        total, approximate = aggregate_count(title, kind)
        if total is None and (not title or matches(r"профиль|profile|сводн|агрегир", title)):
            total, approximate = aggregate_count(description, kind)
        if total is not None:
            totals.append((total, approximate))
        elif title and not matches(r"профиль|profile|selected publications|publication record", title):
            identifier = entry.get("patent_number") or entry.get("number") or title
            individual.add(normalize(identifier))
        elif isinstance(entry, str) and not matches(r"профиль|profile|источник|биограф|сообщает", entry):
            individual.add(normalize(entry))
        elif matches(r"publication record", title):
            individual.add(title)
        else:
            unknown = True
    if totals:
        total, approximate = max(totals, key=lambda item: item[0])
        # Поименные записи обычно входят в агрегат: не складывать их с итогом.
        return max(total, len(individual)), int(approximate or unknown or len(individual) > total)
    if individual:
        return len(individual), 1  # Доступный перечень не подтверждает полный итог.
    return None, int(unknown)


def person_features(person: dict, document: dict, ticker: str,
                    research_missing_dates: str = "professional") -> tuple[dict, list[dict]]:
    features = education_features(person)
    audit = []
    professional = experience_episodes(person.get("professional_experience"))
    research = experience_episodes(person.get("research_experience"), missing_dates=research_missing_dates)
    aliases = company_aliases(document, ticker)
    for kind, episodes in (("professional", professional), ("research", research)):
        for e in episodes:
            audit.append({
                "ticker": ticker, "name": person["name"], "kind": kind, "episode": e.index,
                "organization": e.record.get("organization") if isinstance(e.record, dict) else e.record,
                "start_original": endpoint(e.record, "start") if isinstance(e.record, dict) else None,
                "end_original": endpoint(e.record, "end") if isinstance(e.record, dict) else None,
                "start_used": e.start, "end_used": min(e.end, COLLECTION_YEAR) if e.end is not None else None,
                "years": e.years, "note": e.note,
                "is_current_company": int(is_current_company(e.record.get("organization"), aliases))
                    if isinstance(e.record, dict) and kind == "professional" else None,
            })
    features["professional_experience_years"] = sum(e.years for e in professional)
    features["has_research_experience"] = int(bool(research))
    features["research_experience_years"] = sum(e.years for e in research)
    for key in DIMENSIONS:
        current = normalize(document.get(key))
        relevant = []
        for e in professional:
            if not isinstance(e.record, dict) or (e.start is not None and e.start > COLLECTION_YEAR):
                continue
            label = normalize(e.record.get(key))
            # Для текущей компании известна классификация из корня её файла.
            if not label and is_current_company(e.record.get("organization"), aliases):
                label = current
            if current and label == current:
                relevant.append(e)
        features[f"experience_years_same_{key}"] = sum(e.years for e in relevant)
        features[f"has_experience_same_{key}"] = int(bool(relevant))
    features["current_company_experience_years"] = sum(
        e.years for e in professional if isinstance(e.record, dict)
        and is_current_company(e.record.get("organization"), aliases)
    )
    features["has_other_company_management"] = int(any(
        isinstance(e.record, dict)
        and (e.start is None or e.start <= COLLECTION_YEAR)
        and not is_current_company(e.record.get("organization"), aliases)
        and is_management(e.record.get("role"), e.record.get("organization"))
        for e in professional
    ))
    for kind in ("publications", "patents"):
        features[f"{kind}_count"], features[f"{kind}_count_is_lower_bound"] = scientific_count(person.get(kind), kind)
    return features, audit
