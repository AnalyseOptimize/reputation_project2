#!/usr/bin/env python3
"""Независимые компоненты по компаниям; образование — нормированные признаки + CRITIC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.education_features import FEATURE_COLUMNS, records
from utils.education_weighting import education_critic_component, education_matrix, fixed_bounds
from utils.index_components import SCORE_COLUMNS
from utils.index_matching import SOURCE_NAME_ALIASES, canonical_ticker, clean_name
from utils.source_components import (
    AWARDS_UPPER, DEFAULT_FACE_LOWER, DEFAULT_FACE_UPPER, DEFAULT_MEDIA_UPPER,
    calculate_source, communication_source, source_average,
)


def read_table(path: Path) -> pd.DataFrame:
    # Медиафайл имеет расширение .xlsx, но фактически содержит CSV.
    if zipfile.is_zipfile(path):
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path, encoding="utf-8-sig")


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Только очистка собственного источника: тикер, имя, дубли."""
    df = frame.copy()
    df["ticker"] = [canonical_ticker(t, c) for t, c in zip(df["Ticker"], df["Company"])]
    df.attrs["source_tickers"] = sorted(df.ticker.unique())
    if "Name" in df:
        df["name"] = df.Name.map(clean_name)
        nameless = df.name.eq("")
        df.attrs["nameless_rows"] = df.loc[nameless, ["ticker", "Company"]].to_dict("records")
        df = df.loc[~nameless].copy()
        # Эти три явно заданных варианта нужны для дублей внутри одного источника.
        df["name"] = [SOURCE_NAME_ALIASES.get((t, n), n) for t, n in zip(df.ticker, df.name)]
    return df


def load_education_people(directory: Path, education_file: Path) -> pd.DataFrame:
    """CSV и его собственные JSON относятся к одной образовательной компоненте."""
    metadata = []
    for path in sorted(directory.glob("*_person_info.json")):
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        ticker = canonical_ticker(document["company_ticker"], document["company_name"])
        for is_ceo, person in [(True, document.get("ceo")),
                               *[(False, p) for p in records(document.get("tmt"))]]:
            if person:
                metadata.append({"ticker": ticker, "name": clean_name(person["name"]), "is_ceo": is_ceo})
    if not metadata:
        raise ValueError(f"Не найдены JSON образования в {directory}")
    people = pd.read_csv(education_file, encoding="utf-8-sig")
    missing = {"ticker", "name", *FEATURE_COLUMNS} - set(people)
    if missing:
        raise ValueError(f"Нет колонок образования: {sorted(missing)}")
    people["ticker"] = people.ticker.map(canonical_ticker)
    people["name"] = people.name.map(clean_name)
    if people.name.eq("").any():
        raise ValueError("Пустое имя в образовании")
    # Это точная связь CSV с его источником, не сопоставление людей разных компонент.
    people = people.drop(columns=["is_ceo"], errors="ignore").merge(
        pd.DataFrame(metadata), on=["ticker", "name"], how="outer", validate="one_to_one", indicator=True
    )
    if not people._merge.eq("both").all():
        raise ValueError("CSV образования и JSON содержат разные наблюдения; пересоздайте CSV образования")
    return people.drop(columns="_merge")


def combine_components(components: dict[str, pd.Series], tickers) -> pd.DataFrame:
    """Единственное межкомпонентное объединение: уже агрегированные значения по тикеру."""
    if set(components) != set(SCORE_COLUMNS):
        raise ValueError("Ожидаются ровно пять компонент")
    if any(not series.index.is_unique for series in components.values()):
        raise ValueError("Повторяющиеся тикеры в компоненте")
    result = pd.DataFrame(components).reindex(sorted(set(tickers)))
    result.index.name = "ticker"
    return result[SCORE_COLUMNS].reset_index()


def build(root: Path, education_file: Path, output_dir: Path, audit_dir: Path, *,
          media_upper=DEFAULT_MEDIA_UPPER, face_lower=DEFAULT_FACE_LOWER,
          face_upper=DEFAULT_FACE_UPPER) -> dict:
    input_paths = {
        "media_score": root / "media" / "ceo_media_coverage_unique.xlsx",
        "awards_score": root / "rewards" / "Managers_Rewards.xlsx",
        "face_score": root / "visual" / "MOEXBMI_2024_with_photos_and_face_metrics.csv",
    }
    frames = {component: prepare(read_table(path)) for component, path in input_paths.items()}
    # Ни одна персональная таблица здесь не объединяется с другой.
    sources = {
        component: calculate_source(frame, component, media_upper=media_upper,
                                    face_lower=face_lower, face_upper=face_upper)
        for component, frame in frames.items()
    }
    education = load_education_people(root / "education" / "data", education_file)
    bounds = fixed_bounds(education_matrix(education).columns)
    education_results = {
        sample: education_critic_component(education, ceo_only=(sample == "ceo"), bounds=bounds)
        for sample in ("all_managers", "ceo")
    }

    comm_dir = next((root / name for name in ("communication", "сommunication")
                     if (root / name).is_dir()), None)
    comm_files = sorted(comm_dir.glob("*.xlsx")) if comm_dir else []
    if len(comm_files) != 1:
        raise ValueError("Ожидается один файл .xlsx в communication/сommunication")
    communication_frame = prepare(read_table(comm_files[0]))
    communication, communication_counts = communication_source(communication_frame)

    tickers = set(education.ticker) | set(communication.index)
    for frame in frames.values():
        tickers.update(frame.attrs["source_tickers"])
    outputs, coverages = {}, {}
    for sample in ("all_managers", "ceo"):
        components, counts = {}, {}
        for component, source in sources.items():
            components[component], counts[component] = source_average(source, component, ceo_only=(sample == "ceo"))
        components["expertise_score"] = education_results[sample]["score"]
        selected = education.loc[education.is_ceo] if sample == "ceo" else education
        counts["expertise_score"] = selected.groupby("ticker").size()
        components["communication_score"] = communication
        counts["communication_score"] = communication_counts
        outputs[sample] = combine_components(components, tickers)
        coverage = pd.DataFrame(counts).reindex(sorted(tickers)).fillna(0).astype(int)
        coverage.index.name = "ticker"
        coverages[sample] = coverage

    # Сохранение только после успешного расчёта всех компонент.
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    outputs["all_managers"].to_csv(output_dir / "index_data.csv", index=False, encoding="utf-8-sig")
    outputs["ceo"].to_csv(output_dir / "index_data_ceo.csv", index=False, encoding="utf-8-sig")
    education.to_csv(audit_dir / "education_people.csv", index=False, encoding="utf-8-sig")
    bounds.to_csv(audit_dir / "education_normalization_bounds.csv", encoding="utf-8-sig")
    for component, source in sources.items():
        source.to_csv(audit_dir / f"{component}_people.csv", index=False, encoding="utf-8-sig")
    for sample, result in education_results.items():
        result["critic_details"].to_csv(audit_dir / f"education_critic_{sample}.csv", encoding="utf-8-sig")
        result["normalized"].to_csv(audit_dir / f"education_normalized_{sample}.csv", encoding="utf-8-sig")
        result["coverage"].to_csv(audit_dir / f"education_feature_coverage_{sample}.csv", encoding="utf-8-sig")
        result["preprocessing"].to_csv(audit_dir / f"education_preprocessing_{sample}.csv", encoding="utf-8-sig")
        coverages[sample].to_csv(audit_dir / f"component_coverage_{sample}.csv", encoding="utf-8-sig")
    summary = {
        "companies": len(tickers), "columns": ["ticker", *SCORE_COLUMNS],
        "aggregation": "Independent sources; company-level join on ticker only; missing scores preserved",
        "normalization": {
            "media": {"lower": 0, "upper": media_upper, "transform": "log1p", "bounds": "artificial"},
            "awards": {"lower": 0, "upper": AWARDS_UPPER, "transform": "log1p", "bounds": "theoretical"},
            "face": {"lower": face_lower, "upper": face_upper, "transform": "linear", "bounds": "artificial"},
            "communication": {"lower": -1, "upper": 1, "transform": "linear", "bounds": "theoretical"},
            "education": "Same fixed person-level bounds and CRITIC as education_index_experiments.ipynb",
        },
        "source_people": {**{key: len(value) for key, value in sources.items()}, "education": len(education)},
        "ceo_selection": "Each source uses its own Position; education uses its own JSON ceo",
        "source_ceos": {**{key: int(value.is_ceo.sum()) for key, value in sources.items()},
                       "education": int(education.is_ceo.sum())},
        "missing_components": {
            sample: {col: int(frame[col].isna().sum()) for col in SCORE_COLUMNS}
            for sample, frame in outputs.items()
        },
        "nameless_rows_excluded": {key: frame.attrs.get("nameless_rows", []) for key, frame in frames.items()},
        "outputs": [str(output_dir / name) for name in ("index_data.csv", "index_data_ceo.csv")],
    }
    (audit_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--education-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--media-upper", type=float, default=DEFAULT_MEDIA_UPPER)
    parser.add_argument("--face-lower", type=float, default=DEFAULT_FACE_LOWER)
    parser.add_argument("--face-upper", type=float, default=DEFAULT_FACE_UPPER)
    args = parser.parse_args()
    summary = build(args.root, args.education_file or args.root / "education" / "eduction_processed.csv",
                    args.output_dir or args.root, args.audit_dir or args.root / "index_audit",
                    media_upper=args.media_upper, face_lower=args.face_lower, face_upper=args.face_upper)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
