"""Формулы ноутбука, идентичности и агрегация всех людей / CEO."""

import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.build_index_data import prepare, read_table
from utils.education_features import FEATURE_COLUMNS
from utils.index_components import (
    LEGACY_COLUMNS, NOMINATION_COLUMNS, calculate_person_components,
    communication_components, log_capped_score, observed_minmax_score, positive_log_score,
)
from utils.index_matching import canonical_ticker, match_people

ROOT = Path(__file__).resolve().parents[1]


def person_fixture():
    rows = []
    for ticker, name, ceo, media, face in [("A", "A1", True, 3, 2), ("A", "A2", False, 100, 8),
                                           ("B", "B1", False, 0, np.nan)]:
        row = {"ticker": ticker, "name": name, "person_id": name, "is_ceo": ceo,
               "yandex_count": media, "facial_trust_appearance_proxy": face,
               "education_observed": 1, "media_observed": 1, "face_observed": int(pd.notna(face)),
               **dict.fromkeys(FEATURE_COLUMNS, 0), **dict.fromkeys(LEGACY_COLUMNS, 0),
               **dict.fromkeys(NOMINATION_COLUMNS, "")}
        rows.append(row)
    rows[0].update(Nom_2024="W", Nom_2023="N", notebook_higher_education=1,
                   notebook_degree=1, notebook_publications=2, notebook_research_experience=1,
                   notebook_patents=3, has_mba_emba=1)
    return pd.DataFrame(rows)


class ComponentTests(unittest.TestCase):
    def test_normalizations_match_notebook_functions(self):
        notebook = json.loads((ROOT / "notebooks/index_0_finance.ipynb").read_text())
        tree = ast.parse("".join(notebook["cells"][6]["source"]))
        functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
        namespace = {"np": np, "pd": pd}
        exec(compile(functions, "notebook_functions", "exec"), namespace)
        values = pd.Series([np.nan, -2, 0, 1, 5, 20, 10000])
        for function in (log_capped_score, observed_minmax_score, positive_log_score):
            pd.testing.assert_series_equal(function(values), namespace[function.__name__](values))
        self.assertTrue(observed_minmax_score(pd.Series([5, 5, None])).eq(0).all())

    def test_person_formula_and_awards(self):
        result = calculate_person_components(person_fixture())
        self.assertEqual(result.loc[0, "awards_raw"], 3)
        self.assertAlmostEqual(result.loc[0, "expertise_score"], 80)
        self.assertEqual(result.loc[0, "face_score"], 0)
        self.assertEqual(result.loc[1, "face_score"], 100)
        self.assertEqual(result.loc[2, "face_score"], 0)

    def test_media_max_is_shared_across_companies(self):
        frame = person_fixture()
        frame.loc[2, "person_id"] = frame.loc[0, "person_id"]
        result = calculate_person_components(frame)
        self.assertEqual(result.loc[0, "yandex_count"], result.loc[2, "yandex_count"])

    def test_tone_scale_and_missing(self):
        result = communication_components(pd.DataFrame({"ticker": list("ABCD"), "Тональность": [-1, 0, 1, np.nan]}))
        self.assertEqual(result.communication_score.tolist(), [0, 50, 100, 0])
        self.assertEqual(result.communication_observed.tolist(), [1, 1, 1, 0])
        result = communication_components(pd.DataFrame({"ticker": ["A"], "positive": [.7], "negative": [.1]}))
        self.assertAlmostEqual(result.communication_score.iloc[0], 80)

class MatchingTests(unittest.TestCase):
    def test_ticker_corrections(self):
        self.assertEqual(canonical_ticker("T, TCSG"), "T")
        self.assertEqual(canonical_ticker("YDEX, YNDX"), "YDEX")
        self.assertEqual(canonical_ticker("HEAD", 'ПАО "ЭЙЧ ЭФ ДЖИ"'), "HNFG")
        self.assertEqual(canonical_ticker("HEAD", 'МКПАО "Хэдхантер"'), "HEAD")
        with self.assertRaises(ValueError):
            canonical_ticker("SBER, GAZP")

    def test_transliteration_and_distinct_names(self):
        education = pd.DataFrame({"ticker": ["A", "A"], "name": ["Михаил Эдуардович Осеевский", "Иодковский Станислав Эрикович"]})
        base = pd.DataFrame({"ticker": ["A", "A"], "name": ["Mikhail Eduardovitch Oseevskiy", "Иодковский Эдмунд Феликсович"]})
        report = match_people(education, base)
        self.assertEqual(report.loc[report.education_name.str.startswith("Михаил"), "method"].iloc[0], "fuzzy")
        self.assertEqual(report.loc[report.education_name.str.startswith("Иодковский"), "method"].iloc[0], "unmatched")

    def test_ambiguous_short_name_is_not_assigned_arbitrarily(self):
        education = pd.DataFrame({"ticker": ["A", "A"], "name": ["Иван Петрович Иванов", "Иван Сергеевич Иванов"]})
        base = pd.DataFrame({"ticker": ["A"], "name": ["Иван Иванов"]})
        self.assertTrue(match_people(education, base).method.eq("unmatched").all())

    def test_csv_disguised_as_excel_and_duplicate_source_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "media.xlsx"
            path.write_text("Ticker,Company,Name,Position\nNVTK,НОВАТЭК,Леонид Михельсон,Член СД\n", encoding="utf-8")
            result = prepare(read_table(path))
            self.assertEqual(result.name.iloc[0], "Михельсон Леонид Викторович")


if __name__ == "__main__":
    unittest.main()
