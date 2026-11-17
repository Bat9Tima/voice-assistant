from __future__ import annotations

import unittest

from app.summary_schema import (
    StructuredSummary,
    SummaryFormatError,
    SummaryItem,
    parse_summary,
    render_summary,
)


class ParseSummaryTest(unittest.TestCase):
    def test_confirmed_items_are_kept(self) -> None:
        answer = """
        {
          "facts": [
            {"text": "Пользователю было плохо", "evidence": "мне очень плохо"}
          ],
          "decisions": [
            {"text": "Решили вернуться к теме", "evidence": "давай вернёмся к этому"}
          ],
          "unfinished": [],
          "topics": ["самочувствие"],
          "summary": "Говорили о самочувствии."
        }
        """
        user_text = "мне очень плохо. давай вернёмся к этому завтра"

        summary = parse_summary(answer, user_text)

        self.assertEqual(
            summary.facts,
            [SummaryItem("Пользователю было плохо", "мне очень плохо")],
        )
        self.assertEqual(
            summary.decisions,
            [SummaryItem("Решили вернуться к теме", "давай вернёмся к этому")],
        )
        self.assertEqual(summary.unfinished, [])
        self.assertEqual(summary.topics, ["самочувствие"])
        self.assertEqual(summary.summary, "Говорили о самочувствии.")

    def test_item_without_confirmed_evidence_is_dropped(self) -> None:
        answer = """
        {
          "facts": [
            {"text": "Пользователь любит кофе", "evidence": "обожаю кофе"},
            {"text": "Пользователь устал", "evidence": "я устал"}
          ]
        }
        """
        user_text = "я устал сегодня"

        summary = parse_summary(answer, user_text)

        self.assertEqual(summary.facts, [SummaryItem("Пользователь устал", "я устал")])

    def test_evidence_must_come_from_user(self) -> None:
        answer = """
        {"facts": [{"text": "Пользователь любит чай", "evidence": "я люблю чай"}]}
        """
        # Цитата есть, но это слова ассистента, а не пользователя.
        self.assertEqual(parse_summary(answer, "расскажи про чай"), StructuredSummary())

    def test_short_evidence_is_dropped(self) -> None:
        answer = '{"facts": [{"text": "Пользователь согласился", "evidence": "да"}]}'
        self.assertEqual(parse_summary(answer, "да").facts, [])

    def test_comparison_ignores_case_punctuation_and_yo(self) -> None:
        answer = (
            '{"facts": [{"text": "Пользователь приболел", "evidence": "Я ПРИБОЛЕЛ, но не сильно"}]}'
        )
        summary = parse_summary(answer, "я приболел но не сильно")
        self.assertEqual(len(summary.facts), 1)

    def test_duplicate_items_are_removed(self) -> None:
        answer = """
        {
          "facts": [
            {"text": "Пользователь устал", "evidence": "я устал"},
            {"text": "пользователь устал", "evidence": "мне устал"}
          ]
        }
        """
        summary = parse_summary(answer, "я устал")
        self.assertEqual(len(summary.facts), 1)

    def test_markdown_fence_and_prose_are_tolerated(self) -> None:
        answer = (
            "Вот заметка:\n```json\n"
            '{"facts": [{"text": "Пользователь устал", "evidence": "я устал"}],'
            ' "summary": "Короткий разговор."}\n'
            "```"
        )
        summary = parse_summary(answer, "я устал")
        self.assertEqual(summary.summary, "Короткий разговор.")
        self.assertEqual(len(summary.facts), 1)

    def test_invalid_json_raises(self) -> None:
        with self.assertRaises(SummaryFormatError):
            parse_summary("Говорили о C++ и циклах.", "привет")

    def test_json_array_is_not_a_summary(self) -> None:
        with self.assertRaises(SummaryFormatError):
            parse_summary("[]", "привет")

    def test_topics_are_deduplicated_and_limited(self) -> None:
        answer = (
            '{"topics": ["работа", "Работа", "дом", "отдых", "планы", '
            '"учёба", "семья", "погода", "спорт"]}'
        )
        topics = parse_summary(answer, "").topics
        self.assertEqual(topics, ["работа", "дом", "отдых", "планы", "учёба", "семья", "погода"])

    def test_wrong_types_do_not_break_parsing(self) -> None:
        answer = '{"facts": "нет", "topics": "нет", "summary": 42}'
        summary = parse_summary(answer, "")
        self.assertEqual(summary, StructuredSummary())

    def test_to_dict_is_json_friendly(self) -> None:
        summary = StructuredSummary(
            facts=[SummaryItem("Пользователь устал", "я устал")],
            topics=["самочувствие"],
            summary="Короткий разговор.",
        )
        self.assertEqual(
            summary.to_dict(),
            {
                "facts": [{"text": "Пользователь устал", "evidence": "я устал"}],
                "decisions": [],
                "unfinished": [],
                "topics": ["самочувствие"],
                "summary": "Короткий разговор.",
            },
        )


class RenderSummaryTest(unittest.TestCase):
    def test_renders_sections_in_priority_order(self) -> None:
        summary = StructuredSummary(
            facts=[SummaryItem("Пользователю было плохо", "мне очень плохо")],
            decisions=[SummaryItem("Решили вернуться", "давай вернёмся")],
            unfinished=[SummaryItem("Не договорили о планах", "потом расскажу")],
            topics=["самочувствие", "планы"],
            summary="Говорили о самочувствии.",
        )

        text = render_summary(summary)

        self.assertEqual(
            text,
            "Факты о пользователе:\n"
            "- Пользователю было плохо\n"
            "Решения:\n"
            "- Решили вернуться\n"
            "Осталось:\n"
            "- Не договорили о планах\n"
            "Темы: самочувствие, планы\n"
            "Говорили о самочувствии.",
        )
        self.assertNotIn("мне очень плохо", text)

    def test_empty_summary_renders_empty_text(self) -> None:
        self.assertEqual(render_summary(StructuredSummary()), "")

    def test_summary_only_renders_one_line(self) -> None:
        self.assertEqual(
            render_summary(StructuredSummary(summary="Короткий разговор.")),
            "Короткий разговор.",
        )


if __name__ == "__main__":
    unittest.main()
