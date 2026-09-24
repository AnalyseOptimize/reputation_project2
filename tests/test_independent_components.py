"""Отсутствие связей по людям между компонентами, фиксированные шкалы и CRITIC."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.build_index_data import build, prepare
from utils.education_features import FEATURE_COLUMNS
from utils.education_weighting import ARTIFICIAL_UPPER_BOUNDS, education_critic_component, experiment
from utils.index_components import NOMINATION_COLUMNS, SCORE_COLUMNS
from utils.source_components import calculate_source, communication_source, fixed_score, source_average, source_ceo


def education_fixture():
    rng = np.random.default_rng(42)
    rows = []
    for i in range(12):
        row = {feature: int(rng.integers(0, 2)) for feature in FEATURE_COLUMNS}
        for feature, cap in ARTIFICIAL_UPPER_BOUNDS.items():
            row[feature] = float(rng.uniform(0, cap))
        for feature in ("education_secondary", "education_bachelor", "education_master", "education_doctoral"):
            row[feature] = 0
        row[["education_bachelor", "education_master", "education_doctoral"][i % 3]] = 1
        row.update(ticker=f"C{i}", name=f"Education Person {i}", role="CEO", is_ceo=i < 10)
        rows.append(row)
    return pd.DataFrame(rows)


class IndependentComponentTests(unittest.TestCase):
    def test_fixed_normalization_and_missing(self):
        x = pd.Series([0, 5, 20, np.nan])
        score = fixed_score(x, 0, 20, log=True)
        self.assertAlmostEqual(score[1], 100 * np.log1p(5) / np.log1p(20))
        self.assertEqual(score[2], 100)
        self.assertTrue(pd.isna(score[3]))
        pd.testing.assert_series_equal(fixed_score(x.iloc[:2], 0, 20, log=True), score.iloc[:2])

    def test_source_specific_ceo_and_missing_metric(self):
        frame = prepare(pd.DataFrame({"Ticker": ["X", "X", "X"], "Company": ["Company X"] * 3,
                                      "Name": ["A", "A", "B"], "Position": ["Board member", "CEO", "Board member"],
                                      "facial_trust_appearance_proxy": [0, 0, np.nan]}))
        people = calculate_source(frame, "face_score")
        self.assertEqual(len(people), 2)
        all_score, counts = source_average(people, "face_score")
        ceo_score, _ = source_average(people, "face_score", ceo_only=True)
        self.assertEqual(all_score.loc["X"], 50)
        self.assertEqual(counts.loc["X"], 1)
        self.assertEqual(ceo_score.loc["X"], 50)

    def test_role_in_another_company_does_not_make_ceo(self):
        self.assertTrue(source_ceo("Член СД; генеральный директор ПАО «ЯНДЕКС»", "YDEX", "ПАО ЯНДЕКС"))
        self.assertFalse(source_ceo("Генеральный директор ООО «Яндекс.Такси»", "YDEX", "ПАО ЯНДЕКС"))
        self.assertFalse(source_ceo("Руководитель аппарата генерального директора", "YDEX", "ПАО ЯНДЕКС"))
        self.assertFalse(source_ceo("Заместитель генерального директора", "YDEX", "ПАО ЯНДЕКС"))
        self.assertTrue(source_ceo("Chief Executive Officer", "X", "Company X"))

    def test_unobserved_communication_remains_missing(self):
        result, count = communication_source(pd.DataFrame({"ticker": list("ABCD"), "Тональность": [-1, 0, 1, np.nan]}))
        self.assertEqual(result.iloc[:3].tolist(), [0, 50, 100])
        self.assertTrue(pd.isna(result.loc["D"]))
        self.assertEqual(count.loc["D"], 0)

    def test_education_matches_notebook_for_both_samples(self):
        people = education_fixture()
        for ceo_only in (False, True):
            production = education_critic_component(people, ceo_only=ceo_only)
            notebook = experiment(people, ceo_only=ceo_only)
            np.testing.assert_allclose(production["weights"], notebook["weights"].critic, atol=1e-12)
            np.testing.assert_allclose(production["score"], notebook["scores"].critic, atol=1e-12)

    def test_end_to_end_sources_with_no_common_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ["education/data", "media", "visual", "rewards", "communication"]:
                (root / folder).mkdir(parents=True)
            edu = education_fixture()
            edu_file = root / "education" / "eduction_processed.csv"
            edu.drop(columns="is_ceo").to_csv(edu_file, index=False)
            for person in edu.to_dict("records"):
                doc = {"company_ticker": person["ticker"], "company_name": "Company " + person["ticker"],
                       "ceo": {"name": person["name"]} if person["is_ceo"] else None,
                       "tmt": [] if person["is_ceo"] else [{"name": person["name"]}]}
                (root / "education" / "data" / (person["ticker"] + "_person_info.json")).write_text(json.dumps(doc))
            source_base = pd.DataFrame({"Ticker": edu.ticker, "Company": "Company " + edu.ticker, "Position": "CEO"})
            media = source_base.assign(Name=[f"Media {i}" for i in range(12)], yandex_count=np.arange(12) + 1)
            media.loc[0, "Position"] = "Board member"  # CEO есть только в образовании/наградах/визуале.
            media_path = root / "media" / "ceo_media_coverage_unique.xlsx"
            media.to_csv(media_path, index=False)
            source_base.assign(Name=[f"Photo {i}" for i in range(12)], facial_trust_appearance_proxy=0).to_csv(
                root / "visual" / "MOEXBMI_2024_with_photos_and_face_metrics.csv", index=False)
            source_base.assign(Name=[f"Award {i}" for i in range(12)], **dict.fromkeys(NOMINATION_COLUMNS, "")).to_csv(
                root / "rewards" / "Managers_Rewards.xlsx", index=False)
            source_base.assign(Тональность=1).to_csv(root / "communication" / "tone.xlsx", index=False)
            with patch("utils.index_matching.match_people", side_effect=AssertionError("No cross-source matching allowed")):
                build(root, edu_file, root, root / "audit")
            before = pd.read_csv(root / "index_data.csv")
            ceo = pd.read_csv(root / "index_data_ceo.csv").set_index("ticker")
            self.assertEqual(before.columns.tolist(), ["ticker", *SCORE_COLUMNS])
            self.assertTrue(pd.isna(ceo.loc["C0", "media_score"]))
            self.assertTrue(pd.notna(ceo.loc["C0", "expertise_score"]))
            self.assertTrue(pd.notna(ceo.loc["C11", "media_score"]))
            self.assertTrue(pd.isna(ceo.loc["C11", "expertise_score"]))
            # Полная смена ФИО в одном источнике не меняет ни одно значение.
            media["Name"] = [f"Unrelated person {i}" for i in range(12)]
            media.to_csv(media_path, index=False)
            build(root, edu_file, root, root / "audit")
            pd.testing.assert_frame_equal(before, pd.read_csv(root / "index_data.csv"))


if __name__ == "__main__":
    unittest.main()
