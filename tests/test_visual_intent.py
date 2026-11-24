from __future__ import annotations

import unittest

from app.visual_intent import (
    OTHER,
    SCREEN,
    VISUAL_INTENTS,
    VISUAL_QUESTION,
    looks_visual,
)


class LooksVisualTest(unittest.TestCase):
    def test_screen_questions_are_candidates(self) -> None:
        for phrase in (
            "что тут не так",
            "почему это не работает",
            "что за ошибка",
            "где кнопка сохранения",
            "что написано в окне",
            "это нормально",
            "что случилось",
            "что происходит",
            "что в левом верхнем углу",
            "что видно на экране",
            "что там",
            "что произошло",
            "что за фигня",
            "как это исправить",
            "не открывается",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(looks_visual(phrase))

    def test_plain_conversation_is_not_a_candidate(self) -> None:
        for phrase in (
            "как дела",
            "какая погода",
            "включи музыку",
            "расскажи анекдот",
            "что ты помнишь",
            "открой ютуб",
            "который час",
            "как дела на работе",
        ):
            with self.subTest(phrase=phrase):
                self.assertFalse(looks_visual(phrase))

    def test_filler_does_not_trigger_the_model(self) -> None:
        self.assertFalse(looks_visual("так"))
        self.assertFalse(looks_visual("ну и ладно"))

    def test_case_and_punctuation_are_ignored(self) -> None:
        self.assertTrue(looks_visual("ЧТО ТУТ НЕ ТАК?"))

    def test_empty_text_is_not_a_candidate(self) -> None:
        self.assertFalse(looks_visual(""))
        self.assertFalse(looks_visual("   "))


class VisualQuestionTest(unittest.TestCase):
    def test_prompt_has_two_choices(self) -> None:
        self.assertEqual(VISUAL_QUESTION["type"], "choice")
        self.assertEqual(set(VISUAL_QUESTION["criteria"]), {SCREEN, OTHER})
        self.assertEqual(VISUAL_INTENTS, {SCREEN, OTHER})

    def test_prompt_mentions_the_current_screen(self) -> None:
        instructions = VISUAL_QUESTION["instructions"]
        self.assertIn("экране", instructions)
        self.assertIn("Не считай screen", instructions)


if __name__ == "__main__":
    unittest.main()
