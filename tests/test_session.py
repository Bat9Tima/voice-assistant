from __future__ import annotations

import unittest
from datetime import UTC, datetime

from app.session import DialogSession, Message, SessionMemory


def at(minute: int = 0) -> datetime:
    return datetime(2026, 11, 15, 12, minute, tzinfo=UTC)


class MessageTest(unittest.TestCase):
    def test_as_dict(self) -> None:
        message = Message("user", "привет", at(5))
        self.assertEqual(
            message.as_dict(),
            {
                "role": "user",
                "content": "привет",
                "at": "2026-11-15T12:05:00+00:00",
            },
        )


class DialogSessionTest(unittest.TestCase):
    def test_add_normalizes_whitespace(self) -> None:
        session = DialogSession(started_at=at())
        message = session.add("user", "  как   дела?  ")
        self.assertIsNotNone(message)
        self.assertEqual(message.content, "как дела?")

    def test_add_rejects_unknown_role_and_empty_text(self) -> None:
        session = DialogSession(started_at=at())
        self.assertIsNone(session.add("tool", "x"))
        self.assertIsNone(session.add("user", "   "))

    def test_finish_sets_ended_at_once(self) -> None:
        session = DialogSession(started_at=at())
        session.finish(at=at(10))
        session.finish(at=at(20))
        self.assertEqual(session.ended_at, at(10))
        self.assertTrue(session.finished)

    def test_as_dict(self) -> None:
        session = DialogSession(started_at=at(), id="abc")
        session.add("user", "привет", at=at(1))
        session.finish(at=at(2))
        data = session.as_dict()
        self.assertEqual(data["id"], "abc")
        self.assertEqual(data["started_at"], at().isoformat())
        self.assertEqual(data["ended_at"], at(2).isoformat())
        self.assertEqual(
            data["messages"],
            [{"role": "user", "content": "привет", "at": at(1).isoformat()}],
        )


class SessionMemoryTest(unittest.TestCase):
    def make(self, **kwargs) -> SessionMemory:
        self.saved: list[DialogSession] = []
        kwargs.setdefault("on_finish", self.saved.append)
        return SessionMemory(**kwargs)

    def test_start_opens_empty_session(self) -> None:
        memory = self.make()
        session = memory.start()
        self.assertIs(memory.current, session)
        self.assertFalse(session.finished)
        self.assertEqual(session.messages, [])

    def test_add_autostarts_session(self) -> None:
        memory = self.make()
        memory.add("user", "привет")
        self.assertIsNotNone(memory.current)
        self.assertEqual(len(memory.current.messages), 1)

    def test_history_returns_openai_messages(self) -> None:
        memory = self.make()
        memory.add("user", "привет")
        memory.add("assistant", "здравствуй")
        self.assertEqual(
            memory.history(),
            [
                {"role": "user", "content": "привет"},
                {"role": "assistant", "content": "здравствуй"},
            ],
        )

    def test_history_without_session_is_empty(self) -> None:
        self.assertEqual(self.make().history(), [])

    def test_history_limits_by_messages(self) -> None:
        memory = self.make(max_messages=2)
        for index in range(4):
            memory.add("user", f"m{index}")
        self.assertEqual(
            memory.history(),
            [
                {"role": "user", "content": "m2"},
                {"role": "user", "content": "m3"},
            ],
        )

    def test_history_limits_by_chars_and_keeps_newest(self) -> None:
        memory = self.make(max_chars=10)
        memory.add("user", "aaa")
        memory.add("assistant", "bbb")
        memory.add("user", "cccc")
        memory.add("assistant", "ddddd")
        self.assertEqual(
            memory.history(),
            [
                {"role": "user", "content": "cccc"},
                {"role": "assistant", "content": "ddddd"},
            ],
        )

    def test_long_latest_message_survives_small_budget(self) -> None:
        memory = self.make(max_chars=3)
        memory.add("user", "длинная реплика")
        self.assertEqual(
            memory.history(),
            [{"role": "user", "content": "длинная реплика"}],
        )

    def test_zero_limits_disable_truncation(self) -> None:
        memory = self.make(max_messages=0, max_chars=0)
        for index in range(20):
            memory.add("user", f"m{index}")
        self.assertEqual(len(memory.history()), 20)

    def test_finish_calls_hook_and_clears_current(self) -> None:
        memory = self.make()
        memory.add("user", "привет")
        session = memory.finish()
        self.assertIsNone(memory.current)
        self.assertIsNotNone(session)
        self.assertTrue(session.finished)
        self.assertEqual(self.saved, [session])

    def test_finish_twice_calls_hook_once(self) -> None:
        memory = self.make()
        memory.add("user", "привет")
        memory.finish()
        memory.finish()
        self.assertEqual(len(self.saved), 1)

    def test_empty_session_is_not_saved(self) -> None:
        memory = self.make()
        memory.start()
        memory.finish()
        self.assertEqual(self.saved, [])
        self.assertEqual(memory.finished, [])

    def test_start_closes_previous_session(self) -> None:
        memory = self.make()
        memory.add("user", "раз")
        first = memory.current
        memory.start()
        self.assertTrue(first.finished)
        self.assertEqual(self.saved, [first])
        self.assertIsNot(memory.current, first)

    def test_hook_error_does_not_break_finish(self) -> None:
        def boom(session: DialogSession) -> None:
            raise RuntimeError("нет диска")

        memory = SessionMemory(on_finish=boom)
        memory.add("user", "привет")
        with self.assertLogs("app.session", level="ERROR"):
            session = memory.finish()
        self.assertIsNotNone(session)
        self.assertIsNone(memory.current)

    def test_finished_list_is_capped(self) -> None:
        memory = self.make(keep_finished=2)
        for index in range(5):
            memory.add("user", f"m{index}")
            memory.finish()
        self.assertEqual(len(memory.finished), 2)

    def test_finished_returns_copy(self) -> None:
        memory = self.make()
        memory.add("user", "привет")
        memory.finish()
        memory.finished.clear()
        self.assertEqual(len(memory.finished), 1)

    def test_clear_forgets_everything(self) -> None:
        memory = self.make()
        memory.add("user", "привет")
        memory.clear()
        self.assertIsNone(memory.current)
        self.assertEqual(memory.finished, [])
        self.assertEqual(self.saved, [])

    def test_hooks_run_after_on_finish_in_order(self) -> None:
        calls: list[str] = []
        memory = SessionMemory(
            on_finish=lambda session: calls.append("save"),
            hooks=[lambda session: calls.append("summary")],
        )
        memory.add_hook(lambda session: calls.append("later"))
        memory.add("user", "привет")

        memory.finish()

        self.assertEqual(calls, ["save", "summary", "later"])

    def test_broken_hook_does_not_stop_others(self) -> None:
        calls: list[str] = []

        def boom(session: DialogSession) -> None:
            raise RuntimeError("нет сети")

        memory = SessionMemory(on_finish=boom).add_hook(lambda session: calls.append(session.id))
        memory.add("user", "привет")

        with self.assertLogs("app.session", level="ERROR"):
            session = memory.finish()

        self.assertEqual(calls, [session.id])

    def test_add_hook_returns_memory(self) -> None:
        memory = SessionMemory()
        self.assertIs(memory.add_hook(lambda session: None), memory)


if __name__ == "__main__":
    unittest.main()
