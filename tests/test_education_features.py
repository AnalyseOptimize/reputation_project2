"""Проверки содержательных правил расчёта и неоднозначных исходных данных."""

import unittest

from utils.education_features import (
    classify_field, education_features, experience_episodes, foreign_education,
    person_features, scientific_count, is_current_company, company_aliases, is_management,
)


class EducationTests(unittest.TestCase):
    def test_highest_degree_and_independent_mba(self):
        person = {"education": [
            {"qualification_degree": "Specialist degree", "year": 1990},
            {"qualification_degree": "MBA", "year": 2020},
        ]}
        result = education_features(person)
        self.assertEqual((result["education_master"], result["has_mba_emba"]), (1, 1))
        person["degrees"] = [{"degree": "Кандидат технических наук", "year": 2000}]
        result = education_features(person)
        self.assertEqual((result["education_master"], result["education_doctoral"], result["has_mba_emba"]), (0, 1, 1))

    def test_incomplete_phd_and_academic_title_are_not_degrees(self):
        result = education_features({"education": [
            {"qualification_degree": "Incomplete higher education", "program": "unfinished Ph.D. studies"},
            {"program": "Аспирантура"},
        ], "degrees": [{"degree": "Профессор"}]})
        self.assertEqual(result["education_doctoral"], 0)
        self.assertEqual(result["education_level_unknown"], 1)

    def test_fields_and_retraining(self):
        self.assertEqual(classify_field("Economics and law of Russia and Eastern Europe"), (1, 1))
        self.assertEqual(classify_field("Автоматизация и управление технологическими процессами"), (0, 0))
        self.assertEqual(classify_field("Финансы и кредит"), (0, 1))
        self.assertEqual(classify_field("Юриспруденция"), (1, 0))
        result = education_features({"education": [{"institution": "Юридический институт",
                                                    "program": "Повышение квалификации"}]})
        self.assertEqual(result["has_law_education"], 0)
        self.assertEqual(result["has_professional_retraining"], 0)
        result = education_features({"education": [{"qualification": "Профессиональная переподготовка"}]})
        self.assertEqual(result["has_professional_retraining"], 1)

    def test_foreign_institution_and_joint_program(self):
        self.assertTrue(foreign_education({"institution": "Harvard Business School"}))
        self.assertTrue(foreign_education({"institution": "Киевский политехнический институт"}))
        self.assertFalse(foreign_education({"institution": "Moscow State University"}))
        self.assertFalse(foreign_education({"institution": "Kingston University London / РАНХиГС"}))


class ExperienceTests(unittest.TestCase):
    def test_unsorted_next_start_and_inclusive_overlap(self):
        episodes = experience_episodes([
            {"start_year": 2010, "end_year": None},
            {"start_year": 2000, "end_year": None},
            {"start_year": 2005, "end_year": 2012},
        ])
        self.assertEqual([e.years for e in episodes], [15, 6, 8])
        self.assertEqual(sum(e.years for e in episodes), 29)

    def test_missing_start_uses_previous_explicit_end(self):
        episodes = experience_episodes([
            {"start_year": None, "end_year": 2010},
            {"start_year": 2000, "end_year": 2005},
            {"start_year": None, "end_year": None},
        ])
        self.assertEqual([e.years for e in episodes], [6, 6, 0])

    def test_alias_dates_cap_and_invalid_period(self):
        episodes = experience_episodes([
            {"start_year": None, "start": "2020-02", "end_date": "2026-01-01"},
            {"start_year": 2020, "end_year": 2019},
            {"start_year": 2025, "end_year": 2026},
            {"start_year": 2024, "end_year": 2024},
        ])
        self.assertEqual([e.years for e in episodes], [5, 0, 0, 1])

    def test_parallel_same_start_does_not_truncate(self):
        episodes = experience_episodes([{"start_year": 2020}, {"start_year": 2020}])
        self.assertEqual([e.years for e in episodes], [5, 5])

    def test_existence_without_dates_and_file_company(self):
        doc = {"company_name": "ПАО Сбербанк", "sector": "Financials", "industry": "Banks", "industry_group": "Banks"}
        person = {"name": "Тест", "company_name": "Другая компания", "professional_experience": [
            {"organization": "Сбербанк России", "sector": None, "role": "Президент"},
            {"organization": "ПАО Другая компания", "role": "Генеральный директор", "start_year": 2020},
        ]}
        result, _ = person_features(person, doc, "SBER")
        self.assertEqual(result["has_experience_same_sector"], 1)
        self.assertEqual(result["experience_years_same_sector"], 0)
        self.assertEqual(result["has_other_company_management"], 1)
        self.assertEqual(result["current_company_experience_years"], 0)
        person["professional_experience"] = [{"organization": "ПАО Сбербанк", "role": "Президент", "start_year": 2020}]
        result, _ = person_features(person, doc, "SBER")
        self.assertEqual(result["has_other_company_management"], 0)
        self.assertEqual(result["current_company_experience_years"], 5)

    def test_missing_industry_does_not_match_missing_industry(self):
        result, _ = person_features({"name": "Тест", "professional_experience": [
            {"organization": "A", "start_year": 2000},
        ]}, {"company_name": "B"}, "B")
        self.assertEqual(result["has_experience_same_industry"], 0)

    def test_research_policy(self):
        data = [{"start_year": 2020}]
        self.assertEqual(experience_episodes(data)[0].years, 5)
        self.assertEqual(experience_episodes(data, missing_dates="complete-only")[0].years, 0)

    def test_company_aliases_do_not_merge_subsidiaries(self):
        aliases = company_aliases({"company_name": "ПАО Сбербанк"}, "SBER")
        self.assertTrue(is_current_company("ОАО/ПАО Сбербанк / Сбербанк России", aliases))
        self.assertFalse(is_current_company("Сбербанк / ООО Сбербанк Сервис", aliases))
        self.assertFalse(is_current_company("ООО Сбербанк Сервис", aliases))
        self.assertFalse(is_management("Советник генерального директора", "ПАО Другая компания"))
        self.assertTrue(is_management("Заместитель генерального директора", "ПАО Другая компания"))


class ScientificCountTests(unittest.TestCase):
    def test_summary_and_list_are_not_added(self):
        self.assertEqual(scientific_count([
            {"description": "502 scientific publications, including 29 monographs"},
            {"title": "Some research paper", "year": 2020},
        ], "publications"), (502, 0))

    def test_categories_and_lower_bound(self):
        self.assertEqual(scientific_count([{"title": "11 книг и более 50 статей"}], "publications"), (61, 1))
        self.assertEqual(scientific_count(["Профиль РАН сообщает о 34 патентах РФ и одном патенте Республики Корея."], "patents"), (35, 0))

    def test_unknown_counts_and_deduplication(self):
        self.assertEqual(scientific_count([{"title": "ResearchGate author profile"}], "publications"), (None, 1))
        self.assertEqual(scientific_count([{"title": "A"}, {"title": "A"}], "publications"), (1, 1))
        self.assertEqual(scientific_count([], "patents"), (0, 0))


if __name__ == "__main__":
    unittest.main()
