from __future__ import annotations

import threading
import unittest
from datetime import datetime

from app.rolling_summary import ROLLING_HEADER, RollingSummary
from app.session import DialogSession


class FakeLLM:
    def __init__(self, answer: str = "Кратко о разговоре", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[tuple[list[dict[str, str]], dict]] = []
        self.lock = threading.Lock()

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        with self.lock:
            self.calls.append((messages, kwargs))
        if self.error is not None:
            raise self.error
        return self.answer

    @property
    def texts(self) -> list[str]:
        with self.lock:
            return [messages[0]["content"] for messages, _ in self.calls]


class BlockingLLM:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.started.set()
        self.release.wait(5)
        return "Позднее содержание"


def make_session(*pairs: tuple[str, str]) -> DialogSession:
    session = DialogSession(started_at=datetime.now())
    for role, text in pairs:
        session.add(role, text)
    return session


class RollingSummaryTest(unittest.TestCase):
    def test_disabled_does_nothing(self) -> None:
        llm = FakeLLM()
        rolling = RollingSummary(llm, enabled=False, min_messages=1)
        rolling.observe(make_session(("user", "привет")), 1)
        rolling.close(timeout=2)

        self.assertEqual(llm.calls, [])
        self.assertEqual(rolling.block(), "")

    def test_small_batch_is_postponed(self) -> None:
        llm = FakeLLM()
        rolling = RollingSummary(llm, min_messages=2)
        session = make_session(("user", "раз"))
        rolling.observe(session, 1)
        rolling.close(timeout=2)

        self.assertEqual(llm.calls, [])
        self.assertEqual(rolling.block(), "")

    def test_batch_is_summarized_in_background(self) -> None:
        llm = FakeLLM("Говорили о кофе.")
        rolling = RollingSummary(llm, min_messages=2)
        session = make_session(("user", "люблю кофе"), ("assistant", "понятно"))
        rolling.observe(session, 2)
        rolling.close(timeout=5)

        self.assertEqual(len(llm.calls), 1)
        prompt = llm.texts[0]
        self.assertIn("Пользователь: люблю кофе", prompt)
        self.assertIn("Ассистент: понятно", prompt)
        self.assertIn(ROLLING_HEADER, rolling.block())
        self.assertIn("Говорили о кофе.", rolling.block())

    def test_next_batch_uses_previous_summary(self) -> None:
        llm = FakeLLM("Первое содержание")
        rolling = RollingSummary(llm, min_messages=1)
        rolling.observe(make_session(("user", "раз")), 1)
        rolling.close(timeout=5)

        rolling.observe(make_session(("user", "раз"), ("user", "два")), 2)
        rolling.close(timeout=5)

        self.assertIn("Прошлое содержание:\nПервое содержание", llm.texts[-1])
        self.assertIn("Пользователь: два", llm.texts[-1])

    def test_error_keeps_previous_summary(self) -> None:
        llm = FakeLLM("Старое содержание")
        rolling = RollingSummary(llm, min_messages=1)
        rolling.observe(make_session(("user", "раз")), 1)
        rolling.close(timeout=5)

        llm.error = RuntimeError("модель недоступна")
        with self.assertLogs("app.rolling_summary", level="ERROR"):
            rolling.observe(make_session(("user", "раз"), ("user", "два")), 2)
            rolling.close(timeout=5)

        self.assertIn("Старое содержание", rolling.block())

    def test_reset_discards_late_answer(self) -> None:
        llm = BlockingLLM()
        rolling = RollingSummary(llm, min_messages=1)
        rolling.observe(make_session(("user", "раз")), 1)
        self.assertTrue(llm.started.wait(2))

        rolling.reset()
        llm.release.set()
        rolling.close(timeout=5)

        self.assertEqual(rolling.block(), "")

    def test_long_summary_is_clipped_by_sentence(self) -> None:
        llm = FakeLLM("Первое предложение. Второе предложение.")
        rolling = RollingSummary(llm, min_messages=1, max_chars=20)
        rolling.observe(make_session(("user", "раз")), 1)
        rolling.close(timeout=5)

        self.assertIn("Первое предложение.", rolling.block())
        self.assertNotIn("Второе", rolling.block())

    def test_from_config_reads_section(self) -> None:
        rolling = RollingSummary.from_config(
            FakeLLM(),
            {"enabled": False, "min_messages": 7, "max_chars": 100, "timeout_sec": 5},
        )

        self.assertFalse(rolling.enabled)
        self.assertEqual(rolling.min_messages, 7)
        self.assertEqual(rolling.max_chars, 100)
        self.assertEqual(rolling.timeout, 5)

    def test_empty_answer_is_ignored(self) -> None:
        llm = FakeLLM("   ")
        rolling = RollingSummary(llm, min_messages=1)
        with self.assertLogs("app.rolling_summary", level="ERROR"):
            rolling.observe(make_session(("user", "раз")), 1)
            rolling.close(timeout=5)

        self.assertEqual(rolling.block(), "")


if __name__ == "__main__":
    unittest.main()
