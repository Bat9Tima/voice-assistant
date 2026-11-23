from __future__ import annotations

import unittest

from app.visual_router import (
    AMBIGUOUS,
    CONTEXT_ONLY,
    NEW_CAPTURE,
    TEXT_ONLY,
    VisualDecision,
    VisualRouter,
)


class VisualRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.router = VisualRouter({"vision": {"context_ttl_sec": 600}})

    def decide(
        self,
        text: str,
        *,
        has_context: bool = True,
        context_age: float | None = 10.0,
        in_dialog: bool = False,
        explicit_capture: bool = False,
    ) -> VisualDecision:
        return self.router.decide(
            text,
            has_context=has_context,
            context_age=context_age,
            in_dialog=in_dialog,
            explicit_capture=explicit_capture,
        )

    def test_explicit_capture_wins(self) -> None:
        self.assertEqual(
            self.decide("посмотри на экран", explicit_capture=True).mode,
            NEW_CAPTURE,
        )
        self.assertEqual(
            self.decide("что на экране", explicit_capture=True).mode,
            NEW_CAPTURE,
        )

    def test_explicit_questions_use_saved_description(self) -> None:
        for phrase in (
            "что написано на экране",
            "что было на экране",
            "что на фото",
            "что было на снимке",
            "что видно на прошлом скриншоте",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(self.decide(phrase).mode, CONTEXT_ONLY)

    def test_position_and_color_questions_use_description(self) -> None:
        for phrase in (
            "а что в левом верхнем углу",
            "что в правом нижнем углу",
            "что слева",
            "что на фоне",
            "какого цвета кнопка",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(self.decide(phrase).mode, CONTEXT_ONLY)

    def test_deictic_questions_use_description(self) -> None:
        for phrase in (
            "что там было",
            "а он кто",
            "что ты видела",
            "что на нем",
            "что это",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(self.decide(phrase).mode, CONTEXT_ONLY)

    def test_demonstrative_statement_in_dialog_stays_text(self) -> None:
        for phrase in ("это интересно", "это было давно", "это неважно"):
            with self.subTest(phrase=phrase):
                self.assertEqual(self.decide(phrase, in_dialog=True).mode, TEXT_ONLY)

    def test_general_questions_stay_text(self) -> None:
        for phrase in (
            "какая погода",
            "что нового",
            "что такое фотосинтез",
            "что такое угол",
            "что такое фон",
            "какого цвета небо",
            "что в центре города",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(self.decide(phrase).mode, TEXT_ONLY)

    def test_commands_are_not_visual(self) -> None:
        for phrase in (
            "открой фото",
            "нажми на кнопку в левом верхнем углу",
            "перемести окно в левый верхний угол",
            "сохрани снимок",
            "запомни, что на фото был кот",
            "а открой ютуб",
            "ну нажми на кнопку",
            "пожалуйста, закрой окно",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(self.decide(phrase).mode, TEXT_ONLY)

    def test_no_context_reference_is_ambiguous(self) -> None:
        for phrase in (
            "что на фото",
            "что было на экране",
            "что в левом верхнем углу",
            "что там было",
            "какого цвета кнопка",
        ):
            with self.subTest(phrase=phrase):
                decision = self.decide(phrase, has_context=False, context_age=None)
                self.assertEqual(decision.mode, AMBIGUOUS)
                self.assertEqual(decision.reason, "no_context")

    def test_no_context_general_question_stays_text(self) -> None:
        decision = self.decide("какая погода", has_context=False, context_age=None)
        self.assertEqual(decision.mode, TEXT_ONLY)

    def test_stale_implicit_question_is_ambiguous(self) -> None:
        for phrase in ("что в левом верхнем углу", "а он кто"):
            with self.subTest(phrase=phrase):
                self.assertEqual(self.decide(phrase, context_age=601).mode, AMBIGUOUS)

    def test_stale_explicit_reference_still_works(self) -> None:
        for phrase in ("что было на фото", "что написано на экране"):
            with self.subTest(phrase=phrase):
                self.assertEqual(self.decide(phrase, context_age=601).mode, CONTEXT_ONLY)

    def test_zero_ttl_disables_implicit_context(self) -> None:
        router = VisualRouter({"vision": {"context_ttl_sec": 0}})
        decision = router.decide("что в углу", has_context=True, context_age=1.0)
        self.assertEqual(decision.mode, AMBIGUOUS)

    def test_dialog_allows_short_fragments(self) -> None:
        self.assertEqual(
            self.decide("а в левом верхнем углу", in_dialog=True).mode,
            CONTEXT_ONLY,
        )
        self.assertEqual(self.decide("а он кто", in_dialog=True).mode, CONTEXT_ONLY)
        self.assertEqual(self.decide("что там", in_dialog=True).mode, CONTEXT_ONLY)

    def test_short_fragments_outside_dialog(self) -> None:
        self.assertEqual(self.decide("а в левом верхнем углу").mode, TEXT_ONLY)
        self.assertEqual(self.decide("а он кто").mode, CONTEXT_ONLY)
        self.assertEqual(self.decide("а он").mode, TEXT_ONLY)

    def test_casual_deictic_phrases_stay_text(self) -> None:
        for phrase in ("а он", "как там дела", "что там нового у тебя"):
            with self.subTest(phrase=phrase):
                self.assertEqual(self.decide(phrase, in_dialog=True).mode, TEXT_ONLY)

    def test_empty_text(self) -> None:
        self.assertEqual(self.decide("").mode, TEXT_ONLY)


if __name__ == "__main__":
    unittest.main()
