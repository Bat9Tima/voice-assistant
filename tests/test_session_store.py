from __future__ import annotations

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

        memory.add("user", "привет")
        with self.assertLogs("app.session", level="ERROR"):
            session = memory.finish()

        self.assertIsNotNone(session)
        self.assertIsNone(memory.current)


if __name__ == "__main__":
    unittest.main()
