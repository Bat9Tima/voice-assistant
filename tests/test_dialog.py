from __future__ import annotations

import unittest

from app.dialog import DialogContext, NewsItem, clip_text


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_news(index: int = 1) -> NewsItem:
    return NewsItem(index, f"Факт {index}", f"Заголовок {index}", f"https://example.com/{index}")


class ClipTextTest(unittest.TestCase):
    def test_collapses_whitespace(self) -> None:
        self.assertEqual(clip_text("  одна\n\n две\tтри  "), "одна две три")

    def test_short_text_is_unchanged(self) -> None:
        self.assertEqual(clip_text("коротко", 20), "коротко")

    def test_long_text_is_truncated_with_ellipsis(self) -> None:
        text = clip_text("абвгде", 5)
        self.assertEqual(text, "абвг…")
        self.assertEqual(len(text), 5)

    def test_empty_text(self) -> None:
        self.assertEqual(clip_text(""), "")


class DialogTurnTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.context = DialogContext(turn_ttl_sec=60, clock=self.clock)

    def test_turn_is_valid_at_ttl_boundary(self) -> None:
        self.context.remember_turn("вопрос", "ответ", "general")
        self.clock.now += 60
        self.assertIsNotNone(self.context.turn())

    def test_turn_expires_after_ttl(self) -> None:
        self.context.remember_turn("вопрос", "ответ", "general")
        self.clock.now += 60.001
        self.assertIsNone(self.context.turn())
        self.assertIsNone(self.context.turn())

    def test_zero_ttl_disables_turn(self) -> None:
        context = DialogContext(turn_ttl_sec=0, clock=self.clock)
        self.assertIsNone(context.remember_turn("вопрос", "ответ", "general"))
        self.assertIsNone(context.turn())
        self.assertEqual(context.history(), "")

    def test_history_formats_previous_turn(self) -> None:
        self.context.remember_turn("что такое фотосинтез", "Это процесс.", "general")
        history = self.context.history()
        self.assertIn("Предыдущий вопрос: что такое фотосинтез", history)
        self.assertIn("Предыдущий ответ: Это процесс.", history)
        self.assertNotIn("Тема разговора:", history)

    def test_followup_keeps_root_topic(self) -> None:
        self.context.remember_turn("что такое фотосинтез", "Это процесс.", "general")
        self.context.remember_turn("а почему это важно?", "Потому что.", "general", followup=True)
        self.context.remember_turn("а подробнее?", "Подробнее.", "general", followup=True)

        turn = self.context.turn()
        self.assertIsNotNone(turn)
        self.assertEqual(turn.root, "что такое фотосинтез")
        self.assertIn("Тема разговора: что такое фотосинтез", self.context.history())

    def test_followup_without_previous_turn_has_no_root(self) -> None:
        self.context.remember_turn("первый вопрос", "ответ", "general", followup=True)
        turn = self.context.turn()
        self.assertIsNotNone(turn)
        self.assertEqual(turn.root, "")

    def test_turn_answer_is_clipped(self) -> None:
        self.context.remember_turn("вопрос", "а" * 1300, "general")
        turn = self.context.turn()
        self.assertIsNotNone(turn)
        self.assertEqual(len(turn.answer), 1200)
        self.assertTrue(turn.answer.endswith("…"))

    def test_history_is_empty_after_expiry(self) -> None:
        self.context.remember_turn("вопрос", "ответ", "general")
        self.clock.now += 61
        self.assertEqual(self.context.history(), "")


class DialogNewsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.context = DialogContext(news_ttl_sec=900, clock=self.clock)

    def test_news_is_valid_at_ttl_boundary(self) -> None:
        self.context.remember_news([make_news()])
        self.clock.now += 900
        self.assertEqual(len(self.context.news()), 1)

    def test_news_expires_after_ttl(self) -> None:
        self.context.remember_news([make_news()])
        self.clock.now += 900.001
        self.assertEqual(self.context.news(), [])

    def test_empty_news_does_not_replace_previous(self) -> None:
        self.context.remember_news([make_news()])
        self.context.remember_news([])
        self.assertEqual([item.index for item in self.context.news()], [1])

    def test_news_returns_a_copy(self) -> None:
        self.context.remember_news([make_news()])
        items = self.context.news()
        items.clear()
        self.assertEqual(len(self.context.news()), 1)

    def test_new_digest_restarts_ttl(self) -> None:
        self.context.remember_news([make_news(1)])
        self.clock.now += 800
        self.context.remember_news([make_news(2)])
        self.clock.now += 800
        self.assertEqual([item.index for item in self.context.news()], [2])


class DialogResetTest(unittest.TestCase):
    def test_reset_clears_turn_news_and_city(self) -> None:
        context = DialogContext(clock=FakeClock())
        context.remember_turn("вопрос", "ответ", "weather")
        context.remember_news([make_news()])
        context.weather_city = "Москва"

        context.reset()

        self.assertIsNone(context.turn())
        self.assertEqual(context.news(), [])
        self.assertEqual(context.weather_city, "")
        self.assertEqual(context.history(), "")


if __name__ == "__main__":
    unittest.main()
