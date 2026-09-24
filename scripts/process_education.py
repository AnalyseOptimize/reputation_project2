#!/usr/bin/env python3
"""python3 scripts/process_education.py — расчёт признаков из локальных JSON."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.education_features import (  # noqa: E402
    FEATURE_COLUMNS, classify_field, education_level, foreign_education,
    person_features, records,
)
from utils.education_rules import FIELD_KEYS  # noqa: E402


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def process(input_dir: Path, output: Path, research_missing_dates: str) -> dict:
    # Нерекурсивный glob исключает backup и .json.tmp.
    files = sorted(input_dir.glob("*_person_info.json"))
    if not files:
        raise ValueError(f"Не найдены *_person_info.json в {input_dir}")
    rows, episodes, education_audit = [], [], []
    fields = Counter()
    seen = set()
    for file in files:
        document = json.loads(file.read_text(encoding="utf-8-sig"))
        ticker = document.get("company_ticker") or file.name.removesuffix("_person_info.json")
        people = records(document.get("ceo")) + records(document.get("tmt"))
        for person in people:
            if not isinstance(person, dict) or not person.get("name"):
                raise ValueError(f"Некорректная запись человека в {file}")
            key = (ticker, person["name"])
            if key in seen:
                raise ValueError(f"Повторное наблюдение {key}: требуется явное объединение записей")
            seen.add(key)
            features, person_episodes = person_features(person, document, ticker, research_missing_dates)
            rows.append({"ticker": ticker, "name": person["name"],
                         "role": person.get("current_role"), **features})
            episodes.extend(person_episodes)
            for kind in ("education", "degrees"):
                for i, record in enumerate(records(person.get(kind))):
                    if not isinstance(record, dict):
                        raise ValueError(f"Некорректное образование: {key}, {kind}[{i}]")
                    for field in FIELD_KEYS:
                        if record.get(field):
                            fields[(field, str(record[field]))] += 1
                    education_audit.append({
                        "ticker": ticker, "name": person["name"], "kind": kind, "record": i,
                        "institution": record.get("institution") or record.get("university"),
                        "qualification_degree": record.get("qualification_degree"),
                        "level_used": education_level(record),
                        "foreign_education": int(foreign_education(record)),
                    })
    write_csv(output, ["ticker", "name", "role", *FEATURE_COLUMNS], rows)
    field_rows = []
    for (field, value), count in sorted(fields.items()):
        law, economics = classify_field(value)
        field_rows.append({"field": field, "value": value, "count": count,
                           "has_law_education": law, "has_economics_education": economics})
    write_csv(output.parent / "education_fields_audit.csv",
              ["field", "value", "count", "has_law_education", "has_economics_education"], field_rows)
    write_csv(output.parent / "experience_audit.csv", [
        "ticker", "name", "kind", "episode", "organization", "start_original", "end_original",
        "start_used", "end_used", "years", "note", "is_current_company",
    ], episodes)
    write_csv(output.parent / "education_records_audit.csv", [
        "ticker", "name", "kind", "record", "institution", "qualification_degree",
        "level_used", "foreign_education",
    ], education_audit)
    summary = {
        "input_files": len(files), "people": len(rows), "companies": len({r["ticker"] for r in rows}),
        "collection_year": 2024, "research_missing_dates": research_missing_dates,
        "output": str(output),
        "experience_notes": dict(Counter(e["note"] for e in episodes if e["note"])),
        "unknown_publication_counts": sum(r["publications_count"] is None for r in rows),
        "unknown_patent_counts": sum(r["patents_count"] is None for r in rows),
    }
    (output.parent / "processing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "education" / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "education" / "eduction_processed.csv")
    parser.add_argument("--research-missing-dates", choices=("professional", "complete-only"),
                        default="professional", help="Правило для неполных дат исследовательского опыта")
    args = parser.parse_args()
    try:
        summary = process(args.input_dir, args.output, args.research_missing_dates)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Ошибка: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
