from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from app.session import DialogSession, SessionMemory
from app.session_store import SessionStore


def at(minute: int = 0) -> datetime:
    return datetime(2026, 11, 16, 12, minute, tzinfo=UTC)


def make_session(session_id: str = "s1", minute: int = 0) -> DialogSession:
    session = DialogSession(started_at=at(minute), id=session_id)
    session.add("user", "привет", at=at(minute + 1))
    session.add("assistant", "здравствуй", at=at(minute + 2))
    session.finish(at=at(minute + 3))
    return session


class SessionStoreTest(unittest.TestCase):
    def make(self) -> SessionStore:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        # Вложенной папки ещё нет: хранилище должно создать её само.
        return SessionStore(Path(directory.name) / "nested" / "dialog.db")

    def test_save_and_read_roundtrip(self) -> None:
        store = self.make()
        store.save(make_session())

        self.assertEqual(store.count(), 1)
        sessions = store.sessions()
        self.assertEqual(
            sessions,
            [
                {
                    "id": "s1",
                    "started_at": at().isoformat(),
                    "ended_at": at(3).isoformat(),
                    "messages_count": 2,
                    "status": "completed",
                }
            ],
        )
        self.assertEqual(
            store.messages("s1"),
            [
                {"role": "user", "content": "привет", "at": at(1).isoformat()},
                {"role": "assistant", "content": "здравствуй", "at": at(2).isoformat()},
            ],
        )

    def test_save_twice_does_not_duplicate(self) -> None:
        store = self.make()
        session = make_session()
        store.save(session)
        store.save(session)

        self.assertEqual(store.count(), 1)
        self.assertEqual(len(store.messages("s1")), 2)

    def test_empty_session_is_skipped(self) -> None:
        store = self.make()
        store.save(DialogSession(started_at=at(), id="empty"))

        self.assertEqual(store.count(), 0)
        self.assertEqual(store.messages("empty"), [])

    def test_sessions_are_newest_first(self) -> None:
        store = self.make()
        store.save(make_session("old", minute=0))
        store.save(make_session("new", minute=10))

        self.assertEqual([item["id"] for item in store.sessions()], ["new", "old"])

    def test_limit_returns_latest(self) -> None:
        store = self.make()
        for index in range(5):
            store.save(make_session(f"s{index}", minute=index))

        self.assertEqual([item["id"] for item in store.sessions(limit=2)], ["s4", "s3"])

    def test_delete_removes_session_and_messages(self) -> None:
        store = self.make()
        store.save(make_session())

        self.assertTrue(store.delete("s1"))
        self.assertFalse(store.delete("s1"))
        self.assertEqual(store.count(), 0)
        self.assertEqual(store.messages("s1"), [])

    def test_unknown_session_has_no_messages(self) -> None:
        self.assertEqual(self.make().messages("нет-такой"), [])

    def test_cyrillic_text_survives(self) -> None:
        store = self.make()
        session = DialogSession(started_at=at(), id="text")
        session.add("user", "Привет, как дела?\nРасскажи про Луну.", at=at(1))
        session.finish(at=at(2))
        store.save(session)

        # add() схлопывает пробелы, поэтому перенос строки становится пробелом.
        self.assertEqual(
            store.messages("text")[0]["content"],
            "Привет, как дела? Расскажи про Луну.",
        )

    def test_attach_writes_finished_session(self) -> None:
        store = self.make()
        memory = SessionMemory()
        store.attach(memory)

        memory.add("user", "привет")
        memory.add("assistant", "здравствуй")
        memory.finish()

        self.assertEqual(store.count(), 1)
        self.assertEqual(len(store.messages(memory.finished[0].id)), 2)

    def test_failed_save_does_not_break_session(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        # Путь указывает на папку: SQLite не сможет открыть её как файл.
        store = SessionStore(Path(directory.name))
        memory = SessionMemory()
        store.attach(memory)

        with self.assertLogs("app.session", level="ERROR"):
            memory.add("user", "привет")
            memory.add("assistant", "здравствуй")
            session = memory.finish()

        self.assertIsNotNone(session)
        self.assertIsNone(memory.current)

    # ---------------------------------------------------------- incremental
    def test_create_writes_active_session_once(self) -> None:
        store = self.make()
        session = DialogSession(started_at=at(), id="live")

        store.create(session)
        store.create(session)

        self.assertEqual(store.count(), 1)
        self.assertEqual(store.sessions()[0]["status"], "active")
        self.assertIsNone(store.sessions()[0]["ended_at"])
        self.assertEqual(store.messages("live"), [])

    def test_append_writes_message_immediately(self) -> None:
        store = self.make()
        session = DialogSession(started_at=at(), id="live")
        store.create(session)
        message = session.add("user", "привет", at=at(1))

        store.append(session, message, 0)

        self.assertEqual(
            store.messages("live"),
            [{"role": "user", "content": "привет", "at": at(1).isoformat()}],
        )
        self.assertEqual(store.sessions()[0]["messages_count"], 1)
        self.assertEqual(store.sessions()[0]["status"], "active")

    def test_append_ignores_duplicate_position(self) -> None:
        store = self.make()
        session = DialogSession(started_at=at(), id="live")
        store.create(session)
        message = session.add("user", "привет", at=at(1))
        store.append(session, message, 0)
        store.append(session, message, 0)

        self.assertEqual(len(store.messages("live")), 1)

    def test_finish_closes_session(self) -> None:
        store = self.make()
        session = make_session()
        store.create(session)
        for position, message in enumerate(session.messages):
            store.append(session, message, position)

        store.finish(session)

        saved = store.sessions()[0]
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["ended_at"], at(3).isoformat())
        self.assertEqual(saved["messages_count"], 2)
        self.assertEqual(store.active(), [])

    def test_finish_does_not_require_ended_at(self) -> None:
        store = self.make()
        session = DialogSession(started_at=at(), id="live")
        store.create(session)
        message = session.add("user", "привет", at=at(1))
        store.append(session, message, 0)

        store.finish(session)

        saved = store.sessions()[0]
        self.assertEqual(saved["status"], "completed")
        self.assertIsNotNone(saved["ended_at"])

    def test_crash_leaves_active_session_with_messages(self) -> None:
        store = self.make()
        session = DialogSession(started_at=at(), id="live")
        store.create(session)
        first = session.add("user", "привет", at=at(1))
        second = session.add("assistant", "здравствуй", at=at(2))
        store.append(session, first, 0)
        store.append(session, second, 1)

        # finish() не вызывали: программа закрылась посреди разговора.
        self.assertEqual([item["id"] for item in store.active()], ["live"])
        self.assertEqual(len(store.messages("live")), 2)

    def test_finish_of_empty_session_removes_it(self) -> None:
        store = self.make()
        session = DialogSession(started_at=at(), id="empty")
        store.create(session)

        store.finish(session)

        self.assertEqual(store.count(), 0)

    def test_finish_repairs_missing_messages(self) -> None:
        store = self.make()
        session = DialogSession(started_at=at(), id="live")
        store.create(session)
        first = session.add("user", "привет", at=at(1))
        store.append(session, first, 0)
        # Вторая реплика есть в памяти, но до диска не дошла.
        session.add("assistant", "здравствуй", at=at(2))

        store.finish(session)

        self.assertEqual(len(store.messages("live")), 2)
        self.assertEqual(store.sessions()[0]["messages_count"], 2)

    def test_attach_writes_while_talking(self) -> None:
        store = self.make()
        memory = SessionMemory()
        store.attach(memory)

        memory.start()
        memory.add("user", "привет")

        live = memory.current.id
        self.assertEqual([item["id"] for item in store.active()], [live])
        self.assertEqual(len(store.messages(live)), 1)

        memory.add("assistant", "здравствуй")
        memory.finish()

        self.assertEqual(store.sessions()[0]["status"], "completed")
        self.assertEqual(len(store.messages(live)), 2)


class SummaryStoreTest(unittest.TestCase):
    def make(self) -> SessionStore:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return SessionStore(Path(directory.name) / "dialog.db")

    def test_save_and_read_roundtrip(self) -> None:
        store = self.make()
        store.save(make_session())
        store.save_summary(
            "s1",
            "  Говорили про C++.\nОбсудили циклы.  ",
            model="test-model",
            prompt_version=2,
            created_at=at(10),
        )

        self.assertEqual(
            store.summary("s1"),
            {
                "session_id": "s1",
                "summary": "Говорили про C++.\nОбсудили циклы.",
                "created_at": at(10).isoformat(),
                "model": "test-model",
                "prompt_version": 2,
            },
        )

    def test_summary_is_replaced(self) -> None:
        store = self.make()
        store.save(make_session())
        store.save_summary("s1", "первое", created_at=at(10))
        store.save_summary("s1", "второе", created_at=at(11))

        self.assertEqual(store.summary("s1")["summary"], "второе")
        self.assertEqual(len(store.summaries()), 1)

    def test_empty_summary_is_rejected(self) -> None:
        store = self.make()
        store.save(make_session())
        with self.assertRaises(ValueError):
            store.save_summary("s1", "   ")

    def test_summary_requires_session(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.make().save_summary("нет-такой", "текст")

    def test_unknown_summary_is_none(self) -> None:
        self.assertIsNone(self.make().summary("нет-такой"))

    def test_summaries_are_newest_first(self) -> None:
        store = self.make()
        store.save(make_session("old", minute=0))
        store.save(make_session("new", minute=10))
        store.save_summary("old", "старое", created_at=at(20))
        store.save_summary("new", "новое", created_at=at(30))

        self.assertEqual(
            [item["session_id"] for item in store.summaries()],
            ["new", "old"],
        )
        self.assertEqual(
            [item["session_id"] for item in store.summaries(limit=1)],
            ["new"],
        )

    def test_delete_session_removes_summary(self) -> None:
        store = self.make()
        store.save(make_session())
        store.save_summary("s1", "резюме")

        self.assertTrue(store.delete("s1"))
        self.assertIsNone(store.summary("s1"))
        self.assertEqual(store.summaries(), [])


if __name__ == "__main__":
    unittest.main()
