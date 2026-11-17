from __future__ import annotations

import unittest

from app.importance import HIGH, LOW, MEDIUM, classify
from app.summary_schema import SummaryItem


def item(text: str, evidence: str = "мне очень плохо сегодня") -> SummaryItem:
    return SummaryItem(text=text, evidence=evidence)


class ImportanceTest(unittest.TestCase):
    def test_health_is_high(self) -> None:
        self.assertEqual(classify("fact", item("Пользователю очень плохо"))[0], HIGH)
        self.assertEqual(classify("fact", item("Пользователь плохо спал"))[0], HIGH)

    def test_preference_is_high(self) -> None:
        self.assertEqual(classify("fact", item("Пользователь обожает кофе"))[0], HIGH)

    def test_relationship_is_high(self) -> None:
        self.assertEqual(
            classify("fact", item("Пользователю отказали в ответных чувствах"))[0],
            HIGH,
        )

    def test_study_and_work_are_medium(self) -> None:
        self.assertEqual(classify("fact", item("Пользователь изучает C++"))[0], MEDIUM)
        self.assertEqual(classify("fact", item("Пользователь ищет работу"))[0], MEDIUM)

    def test_leisure_is_low(self) -> None:
        self.assertEqual(classify("fact", item("Пользователь играет в Fortnite"))[0], LOW)

    def test_plain_fact_is_medium(self) -> None:
        self.assertEqual(classify("fact", item("Пользователь живёт в Москве"))[0], MEDIUM)

    def test_strong_decision_is_high(self) -> None:
        self.assertEqual(
            classify("decision", item("Пользователь решил бросить курс"))[0],
            HIGH,
        )
        self.assertEqual(
            classify("decision", item("Пользователь выбрал другой курс"))[0],
            HIGH,
        )

    def test_ordinary_decision_is_medium(self) -> None:
        self.assertEqual(
            classify("decision", item("Пользователь подумает о курсе"))[0],
            MEDIUM,
        )

    def test_unfinished_health_is_high(self) -> None:
        self.assertEqual(
            classify("unfinished", item("Пользователь собирается к врачу"))[0],
            HIGH,
        )
        self.assertEqual(
            classify("unfinished", item("Пользователь не закончил проект"))[0],
            MEDIUM,
        )

    def test_confidence_grows_with_evidence(self) -> None:
        short = classify("fact", item("Пользователь любит кофе", "люблю кофе"))[1]
        long = classify(
            "fact",
            item("Пользователь любит кофе", "я очень люблю кофе по утрам"),
        )[1]
        self.assertGreater(long, short)

    def test_speculation_lowers_confidence(self) -> None:
        sure = classify("fact", item("Пользователь любит кофе"))[1]
        maybe = classify(
            "fact",
            item("Пользователь, наверное, любит кофе"),
        )[1]
        self.assertLess(maybe, sure)

    def test_case_and_yo_are_ignored(self) -> None:
        self.assertEqual(classify("fact", item("ПОЛЬЗОВАТЕЛЬ УСТАЛ"))[0], HIGH)


if __name__ == "__main__":
    unittest.main()
