from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.dialog import DialogContext, DialogService, NewsItem, clip_text
from app.persona import PersonaMemory
from app.profile import Profile
from app.session import SessionMemory


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


class FakeLLM:
    def __init__(self, answer: str | Exception = "Ответ") -> None:
        self.answer = answer
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.calls.append((messages, kwargs))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class DialogServiceTest(unittest.TestCase):
    def test_entry_phrases_with_fillers(self) -> None:
        service = DialogService({"dialog": {}}, FakeLLM())
        for text in ("давай поговорим", "ну давай поговорим", "Давай поговорим!", "поговорим"):
            self.assertTrue(service.wants_entry(text), text)
        self.assertFalse(service.wants_entry("давай откроем ютуб"))

    def test_exit_phrases_with_fillers(self) -> None:
        service = DialogService({"dialog": {}}, FakeLLM())
        self.assertTrue(service.wants_exit("ну хватит"))
        self.assertTrue(service.wants_exit("закончим разговор"))
        self.assertFalse(service.wants_exit("хватит играть"))

    def test_reply_sends_one_message_and_system_prompt(self) -> None:
        llm = FakeLLM("  Ответ   модели  ")
        service = DialogService({"dialog": {"temperature": 0.4, "timeout_sec": 12}}, llm)
        answer = service.reply("как дела?")
        self.assertEqual(answer, "Ответ модели")
        messages, kwargs = llm.calls[0]
        self.assertEqual(messages, [{"role": "user", "content": "как дела?"}])
        self.assertIn("голосовой ассистент", kwargs["system"])
        self.assertEqual(kwargs["temperature"], 0.4)
        self.assertEqual(kwargs["timeout"], 12)

    def test_datetime_is_in_system_prompt(self) -> None:
        now = "Сегодня 13 ноября 2026 года, пятница. Сейчас 20:47."
        llm = FakeLLM()
        service = DialogService({"dialog": {}}, llm)
        with patch("app.dialog.datetime_context", return_value=now):
            service.reply("привет")
        self.assertIn(now, llm.calls[0][1]["system"])

    def test_profile_block_is_in_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.txt"
            path.write_text("Меня зовут Алексей.", encoding="utf-8")
            llm = FakeLLM()
            service = DialogService({"dialog": {}}, llm, Profile(path))
            service.reply("привет")
        self.assertIn("Меня зовут Алексей.", llm.calls[0][1]["system"])

    def test_empty_reply_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DialogService({"dialog": {}}, FakeLLM()).reply("  ")

    def test_empty_model_answer_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DialogService({"dialog": {}}, FakeLLM(" ")).reply("привет")

    def test_custom_phrases_from_config(self) -> None:
        service = DialogService(
            {"dialog": {"entry_phrases": ["начнем"], "exit_phrases": ["конец"]}},
            FakeLLM(),
        )
        self.assertTrue(service.wants_entry("начнем"))
        self.assertFalse(service.wants_entry("давай поговорим"))
        self.assertTrue(service.wants_exit("конец"))

    def test_limits_from_config(self) -> None:
        service = DialogService(
            {
                "dialog": {
                    "idle_timeout_sec": 5,
                    "max_utterance_sec": 7,
                    "no_speech_timeout_sec": 3,
                    "silence_ms": 900,
                    "max_errors": 3,
                }
            },
            FakeLLM(),
        )
        self.assertEqual(service.idle_timeout, 5)
        self.assertEqual(service.max_utterance, 7)
        self.assertEqual(service.no_speech_timeout, 3)
        self.assertEqual(service.silence_ms, 900)
        self.assertEqual(service.max_errors, 3)

    def test_history_is_sent_to_model(self) -> None:
        llm = FakeLLM("Первый ответ")
        service = DialogService({"dialog": {}}, llm)
        service.reply("привет")
        llm.answer = "Второй ответ"
        service.reply("как дела")
        messages, _ = llm.calls[1]
        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "привет"},
                {"role": "assistant", "content": "Первый ответ"},
                {"role": "user", "content": "как дела"},
            ],
        )

    def test_failed_turn_is_not_saved(self) -> None:
        llm = FakeLLM(RuntimeError("нет сети"))
        service = DialogService({"dialog": {}}, llm)
        with self.assertRaises(RuntimeError):
            service.reply("привет")
        self.assertEqual(service.sessions.history(), [])

    def test_empty_model_answer_is_not_saved(self) -> None:
        service = DialogService({"dialog": {}}, FakeLLM("   "))
        with self.assertRaises(ValueError):
            service.reply("привет")
        self.assertEqual(service.sessions.history(), [])

    def test_start_and_stop_close_session(self) -> None:
        saved: list = []
        service = DialogService({"dialog": {}}, FakeLLM("Ответ"))
        service.sessions.on_finish = saved.append
        service.start()
        service.reply("привет")
        service.stop()
        self.assertEqual(len(saved), 1)
        self.assertEqual(
            [message.role for message in saved[0].messages],
            ["user", "assistant"],
        )
        self.assertIsNone(service.sessions.current)

    def test_new_dialog_starts_with_clean_history(self) -> None:
        llm = FakeLLM("Ответ")
        service = DialogService({"dialog": {}}, llm)
        service.start()
        service.reply("старая тема")
        service.stop()
        service.start()
        service.reply("новая тема")
        self.assertEqual(
            llm.calls[-1][0],
            [{"role": "user", "content": "новая тема"}],
        )

    def test_history_limits_from_config(self) -> None:
        service = DialogService(
            {"dialog": {"history_messages": 1, "history_chars": 0}},
            FakeLLM(),
        )
        self.assertEqual(service.history_messages, 1)
        self.assertEqual(service.history_chars, 0)
        self.assertEqual(service.sessions.max_messages, 1)
        self.assertEqual(service.sessions.max_chars, 0)

    def test_history_defaults(self) -> None:
        service = DialogService({"dialog": {}}, FakeLLM())
        self.assertEqual(service.history_messages, 12)
        self.assertEqual(service.history_chars, 4000)

    def test_custom_session_memory_is_used(self) -> None:
        memory = SessionMemory()
        service = DialogService({"dialog": {}}, FakeLLM(), sessions=memory)
        self.assertIs(service.sessions, memory)

    def test_memory_block_is_added_to_system_prompt(self) -> None:
        class FakeMemory:
            def __init__(self) -> None:
                self.queries: list[str] = []

            def block(self, query: str) -> str:
                self.queries.append(query)
                return "Эпизодическая память: любит C++."

        memory = FakeMemory()
        llm = FakeLLM()
        service = DialogService({"dialog": {}}, llm, memory=memory)

        service.reply("что я изучал?")

        self.assertEqual(memory.queries, ["что я изучал?"])
        self.assertIn("Эпизодическая память: любит C++.", llm.calls[0][1]["system"])

    def test_memory_failure_does_not_break_reply(self) -> None:
        class BrokenMemory:
            def block(self, query: str) -> str:
                raise RuntimeError("Chroma умерла")

        llm = FakeLLM("Ответ")
        service = DialogService({"dialog": {}}, llm, memory=BrokenMemory())

        self.assertEqual(service.reply("привет"), "Ответ")
        self.assertNotIn("Эпизодическая", llm.calls[0][1]["system"])

    def test_without_memory_no_block(self) -> None:
        llm = FakeLLM()
        DialogService({"dialog": {}}, llm).reply("привет")
        self.assertNotIn("Эпизодическая", llm.calls[0][1]["system"])

    def test_persona_block_is_added_to_system_prompt(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "luna.txt"
        path.write_text("Люблю кофе.\n", encoding="utf-8")
        llm = FakeLLM()
        service = DialogService({"dialog": {}}, llm, persona=PersonaMemory(path))

        service.reply("привет")

        system = llm.calls[0][1]["system"]
        self.assertIn("Твои собственные факты", system)
        self.assertIn("Люблю кофе.", system)


class ProfileTest(unittest.TestCase):
    def make(self, text: str | None = None, limit: int = 4000) -> Profile:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "profile.txt"
        if text is not None:
            path.write_text(text, encoding="utf-8")
        return Profile(path, limit=limit)

    def test_missing_file_is_empty(self) -> None:
        profile = Profile(Path(tempfile.gettempdir()) / "нет-такого-профиля.txt")
        self.assertEqual(profile.text(), "")
        self.assertEqual(profile.block(), "")

    def test_without_path_is_empty(self) -> None:
        self.assertEqual(Profile(None).text(), "")

    def test_text_and_block(self) -> None:
        profile = self.make("Персона: спокойно.\n\nФакты:\n- любит кратко.\n")
        self.assertEqual(profile.text(), "Персона: спокойно.\n\nФакты:\n- любит кратко.")
        self.assertIn("Персона: спокойно.", profile.block())
        self.assertIn("Контекст от пользователя", profile.block())

    def test_bom_is_stripped(self) -> None:
        self.assertEqual(self.make("\ufeffПривет").text(), "Привет")

    def test_limit_cuts_by_line(self) -> None:
        profile = self.make("первая строка\nвторая строка\nтретья", limit=20)
        self.assertEqual(profile.text(), "первая строка\n…")

    def test_zero_limit_disables_cut(self) -> None:
        text = "а" * 100
        self.assertEqual(self.make(text, limit=0).text(), text)


if __name__ == "__main__":
    unittest.main()
