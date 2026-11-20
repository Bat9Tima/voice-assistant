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

    def test_recover_closes_crashed_session(self) -> None:
        store = self.make()
        session = DialogSession(started_at=at(), id="live")
        store.create(session)
        first = session.add("user", "привет", at=at(1))
        second = session.add("assistant", "здравствуй", at=at(2))
        store.append(session, first, 0)
        store.append(session, second, 1)

        recovered = store.recover()

        self.assertEqual(store.active(), [])
        saved = store.sessions()[0]
        self.assertEqual(saved["status"], "completed")
        # Время окончания — последняя реплика, а не момент следующего запуска.
        self.assertEqual(saved["ended_at"], at(2).isoformat())
        self.assertEqual([item.id for item in recovered], ["live"])
        self.assertEqual(recovered[0].started_at, at())
        self.assertEqual(recovered[0].ended_at, at(2))
        self.assertEqual(
            [(message.role, message.content) for message in recovered[0].messages],
            [("user", "привет"), ("assistant", "здравствуй")],
        )

    def test_recover_deletes_empty_active_session(self) -> None:
        store = self.make()
        store.create(DialogSession(started_at=at(), id="empty"))

        self.assertEqual(store.recover(), [])
        self.assertEqual(store.count(), 0)

    def test_recover_keeps_completed_sessions(self) -> None:
        store = self.make()
        store.save(make_session())

        self.assertEqual(store.recover(), [])
        self.assertEqual(store.active(), [])
        self.assertEqual(store.sessions()[0]["status"], "completed")

    def test_recover_is_idempotent(self) -> None:
        store = self.make()
        session = DialogSession(started_at=at(), id="live")
        store.create(session)
        store.append(session, session.add("user", "привет", at=at(1)), 0)

        self.assertEqual(len(store.recover()), 1)
        self.assertEqual(store.recover(), [])

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
                "data": {},
                "started_at": at(0).isoformat(),
            },
        )

    def test_structured_data_roundtrip(self) -> None:
        store = self.make()
        store.save(make_session())
        data = {
            "facts": [
                {"text": "Пользователь устал", "evidence": "я устал"},
            ],
            "topics": ["самочувствие"],
            "summary": "Короткий разговор.",
        }
        store.save_summary("s1", "Факты о пользователе:\n- Пользователь устал", data=data)

        self.assertEqual(store.summary("s1")["data"], data)
        self.assertEqual(store.summaries()[0]["data"], data)

    def test_invalid_data_json_is_read_as_empty(self) -> None:
        store = self.make()
        store.save(make_session())
        store.save_summary("s1", "Текст")
        conn = sqlite3.connect(store.path)
        try:
            conn.execute("UPDATE summaries SET data = 'не json' WHERE session_id = 's1'")
            conn.commit()
        finally:
            conn.close()

        self.assertEqual(store.summary("s1")["data"], {})

    def test_summaries_with_sessions_join_session_metadata(self) -> None:
        store = self.make()
        store.save(make_session("old", minute=0))
        store.save(make_session("new", minute=10))
        store.save_summary("old", "старое", created_at=at(20))
        store.save_summary("new", "новое", created_at=at(30))

        rows = store.summaries_with_sessions()

        self.assertEqual([row["session_id"] for row in rows], ["new", "old"])
        self.assertEqual(rows[0]["messages_count"], 2)
        self.assertEqual(rows[0]["status"], "completed")
        self.assertEqual(rows[0]["started_at"], at(10).isoformat())
        self.assertEqual(rows[0]["data"], {})

    def test_old_summaries_table_is_migrated(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "dialog.db"
        conn = sqlite3.connect(path)
        try:
            conn.executescript(
                """
                CREATE TABLE sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    messages_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'completed'
                );
                CREATE TABLE summaries (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version INTEGER NOT NULL DEFAULT 1
                );
                """
            )
            conn.execute(
                "INSERT INTO sessions (id, started_at, messages_count, status) "
                "VALUES ('s1', ?, 2, 'completed')",
                (at().isoformat(),),
            )
            conn.execute(
                "INSERT INTO summaries "
                "(session_id, summary, created_at, model, prompt_version) "
                "VALUES ('s1', 'старое', ?, 'test', 1)",
                (at(10).isoformat(),),
            )
            conn.commit()
        finally:
            conn.close()

        store = SessionStore(path)

        self.assertEqual(store.summary("s1")["summary"], "старое")
        self.assertEqual(store.summary("s1")["data"], {})

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
        self.assertEqual(
            [item["session_id"] for item in store.summaries(limit=None)],
            ["new", "old"],
        )

    def test_delete_session_removes_summary(self) -> None:
        store = self.make()
        store.save(make_session())
        store.save_summary("s1", "резюме")

        self.assertTrue(store.delete("s1"))
        self.assertIsNone(store.summary("s1"))
        self.assertEqual(store.summaries(), [])


class MemoryFactsStoreTest(unittest.TestCase):
    def make(self) -> SessionStore:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return SessionStore(Path(directory.name) / "dialog.db")

    def add(self, store: SessionStore, **fields) -> int:
        values = {
            "session_id": "s1",
            "kind": "fact",
            "fact_key": "preference",
            "text": "Пользователь любит кофе",
            "evidence": "обожаю кофе",
            "importance": "high",
            "confidence": 0.9,
            "created_at": at(1).isoformat(),
            "updated_at": at(1).isoformat(),
        }
        values.update(fields)
        return store.add_memory_fact(**values)

    def test_add_and_read_roundtrip(self) -> None:
        store = self.make()
        store.save(make_session())
        fact_id = self.add(store)

        rows = store.memory_facts()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], fact_id)
        self.assertEqual(rows[0]["status"], "active")
        self.assertEqual(rows[0]["confidence"], 0.9)
        self.assertEqual(store.memory_fact(fact_id)["text"], "Пользователь любит кофе")
        self.assertIsNone(store.memory_fact(fact_id + 100))
        self.assertEqual(store.memory_fact_count(), 1)

    def test_durability_defaults_and_roundtrips(self) -> None:
        store = self.make()
        store.save(make_session())
        default_id = self.add(store)
        temporary_id = self.add(
            store,
            fact_key="health",
            text="Пользователь болеет",
            durability="temporary",
        )

        self.assertEqual(store.memory_fact(default_id)["durability"], "permanent")
        self.assertEqual(store.memory_fact(temporary_id)["durability"], "temporary")

    def test_scope_and_event_date_roundtrip(self) -> None:
        store = self.make()
        store.save(make_session())
        default_id = self.add(store)
        event_id = self.add(
            store,
            fact_key="event",
            text="Пользователь вчера сдал экзамен",
            scope="event",
            event_at=at(2).isoformat(),
        )

        self.assertEqual(store.memory_fact(default_id)["scope"], "trait")
        self.assertEqual(store.memory_fact(default_id)["event_at"], "")
        self.assertEqual(store.memory_fact(event_id)["scope"], "event")
        self.assertEqual(store.memory_fact(event_id)["event_at"], at(2).isoformat())

    def test_status_filter_and_newest_first(self) -> None:
        store = self.make()
        store.save(make_session())
        first = self.add(store)
        second = self.add(
            store,
            kind="decision",
            fact_key="decision:курс",
            text="Пользователь решил бросить курс",
            created_at=at(2).isoformat(),
            updated_at=at(2).isoformat(),
        )
        store.update_memory_fact(first, status="superseded")

        active = store.memory_facts(status="active")
        all_rows = store.memory_facts(status="", limit=10)

        self.assertEqual([row["id"] for row in active], [second])
        self.assertEqual([row["id"] for row in all_rows], [second, first])
        self.assertEqual(store.memory_fact_count(), 1)
        self.assertEqual(store.memory_fact_count(status="superseded"), 1)

    def test_update_changes_allowed_fields_only(self) -> None:
        store = self.make()
        store.save(make_session())
        fact_id = self.add(store)

        self.assertTrue(
            store.update_memory_fact(
                fact_id,
                text="Пользователь обожает кофе",
                status="superseded",
                replaced_by=42,
                unknown="игнор",
            )
        )
        row = store.memory_facts()[0]
        self.assertEqual(row["text"], "Пользователь обожает кофе")
        self.assertEqual(row["status"], "superseded")
        self.assertEqual(row["replaced_by"], 42)
        self.assertFalse(store.update_memory_fact(fact_id))
        self.assertFalse(store.update_memory_fact(9999, text="нет"))

    def test_old_facts_table_gets_durability(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "dialog.db"
        conn = sqlite3.connect(path)
        try:
            conn.executescript(
                """
                CREATE TABLE sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    messages_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'completed'
                );
                CREATE TABLE memory_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    text TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    importance TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    replaced_by INTEGER,
                    conflict_with INTEGER,
                    resolved_by INTEGER
                );
                """
            )
            conn.execute(
                "INSERT INTO sessions (id, started_at, messages_count, status) "
                "VALUES ('s1', ?, 2, 'completed')",
                (at().isoformat(),),
            )
            conn.execute(
                "INSERT INTO memory_facts "
                "(session_id, kind, fact_key, text, evidence, importance, "
                "confidence, status, created_at, updated_at) "
                "VALUES ('s1', 'fact', 'preference', 'Пользователь любит кофе', "
                "'обожаю кофе', 'high', 0.9, 'active', ?, ?)",
                (at().isoformat(), at().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

        store = SessionStore(path)

        migrated = store.memory_facts()[0]
        self.assertEqual(migrated["durability"], "permanent")
        self.assertEqual(migrated["scope"], "trait")
        self.assertEqual(migrated["event_at"], "")

    def test_delete_by_session(self) -> None:
        store = self.make()
        store.save(make_session("s1"))
        store.save(make_session("s2", minute=10))
        self.add(store, session_id="s1")
        self.add(store, session_id="s2")

        self.assertEqual(store.delete_memory_facts("s1"), 1)
        self.assertEqual([row["session_id"] for row in store.memory_facts()], ["s2"])

    def test_delete_session_cascades_to_facts(self) -> None:
        store = self.make()
        store.save(make_session())
        self.add(store)

        self.assertTrue(store.delete("s1"))
        self.assertEqual(store.memory_facts(), [])

    def test_fact_requires_session(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.add(self.make(), session_id="нет-такой")

    def test_unknown_fact_fields_are_ignored(self) -> None:
        store = self.make()
        store.save(make_session())
        fact_id = self.add(store)

        self.assertFalse(store.update_memory_fact(fact_id, entirely_unknown="x"))

    def test_character_versions_roundtrip(self) -> None:
        store = self.make()
        first = store.add_character_version(
            created_at=at().isoformat(),
            source="stable",
            quote="не матерись",
            style={"profanity": "off"},
            moment={},
            moment_date="",
            summary="без мата",
        )
        second = store.add_character_version(
            created_at=at(5).isoformat(),
            source="moment",
            quote="сегодня без шуток",
            style={"profanity": "off"},
            moment={"humor": "off"},
            moment_date="2026-11-16",
            summary="без шуток",
        )

        self.assertEqual(first, 1)
        self.assertEqual(second, 2)
        latest = store.latest_character()
        self.assertEqual(latest["version"], 2)
        self.assertEqual(latest["style"], {"profanity": "off"})
        self.assertEqual(latest["moment"], {"humor": "off"})
        self.assertEqual(latest["moment_date"], "2026-11-16")
        history = store.character_history()
        self.assertEqual([row["version"] for row in history], [2, 1])
        self.assertEqual(history[1]["quote"], "не матерись")
        self.assertEqual(history[1]["style"], {"profanity": "off"})

    def test_character_history_limit(self) -> None:
        store = self.make()
        for index in range(3):
            store.add_character_version(
                created_at=at(index).isoformat(),
                source="stable",
                quote=f"просьба {index}",
                style={"length": "short"},
                moment={},
                moment_date="",
                summary=f"версия {index}",
            )

        self.assertEqual([row["version"] for row in store.character_history(2)], [3, 2])

    def test_empty_character(self) -> None:
        store = self.make()

        self.assertIsNone(store.latest_character())
        self.assertEqual(store.character_history(), [])

    def test_style_observations_roundtrip(self) -> None:
        store = self.make()
        first = store.add_style_observation(
            created_at=at().isoformat(),
            session_id="s1",
            words=5,
            chars=24,
            profanity_words=1,
            profanity_messages=1,
            rough_words=0,
            humor_messages=1,
            irony_messages=0,
            question=1,
        )
        second = store.add_style_observation(
            created_at=at(5).isoformat(),
            session_id="s1",
            words=2,
            chars=7,
        )

        rows = store.style_observations()

        self.assertEqual([row["id"] for row in rows], [second, first])
        self.assertEqual(rows[0]["words"], 2)
        self.assertEqual(rows[1]["profanity_words"], 1)
        self.assertEqual(rows[1]["humor_messages"], 1)
        self.assertEqual(rows[1]["question"], 1)
        self.assertEqual(store.style_observations(1)[0]["id"], second)


if __name__ == "__main__":
    unittest.main()
