from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, date, datetime
from pathlib import Path

from app.embeddings import EmbeddingError
from app.memory import (
    MEMORY_HEADER,
    ChromaVectorStore,
    MemoryHit,
    MemoryService,
    VectorHit,
)
from app.memory_manager import MemoryManager
from app.session import DialogSession
from app.session_store import SessionStore

try:
    import chromadb  # noqa: F401

    HAS_CHROMA = True
except Exception:  # noqa: BLE001
    HAS_CHROMA = False


def at(minute: int = 0) -> datetime:
    return datetime(2026, 11, 16, 12, minute, tzinfo=UTC)


class FakeEmbedder:
    """Векторы без сети: C++ и циклы близки, остальное — в стороне."""

    model = "fake-voyage"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        self.calls.append((list(texts), input_type))
        if self.error is not None:
            raise self.error
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "c++" in lowered or "цикл" in lowered or "изуча" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0])
        return vectors


class FakeVectors:
    """Хранилище векторов в памяти."""

    def __init__(self) -> None:
        self.items: dict[str, tuple[list[float], dict]] = {}
        self.hits: list[VectorHit] = []
        self.deleted: list[str] = []
        self.error: Exception | None = None
        self.where: dict | None = None

    def upsert(self, session_id: str, embedding: list[float], metadata: dict) -> None:
        if self.error is not None:
            raise self.error
        self.items[session_id] = (embedding, metadata)

    def search(
        self,
        embedding: list[float],
        limit: int,
        where: dict | None = None,
    ) -> list[VectorHit]:
        if self.error is not None:
            raise self.error
        self.where = where
        return self.hits[:limit]

    def indexed(self) -> dict[str, dict]:
        if self.error is not None:
            raise self.error
        return {session_id: metadata for session_id, (_, metadata) in self.items.items()}

    def delete_session(self, session_id: str) -> None:
        if self.error is not None:
            raise self.error
        self.deleted.append(session_id)
        for vector_id in list(self.items):
            _, metadata = self.items[vector_id]
            stored = str(metadata.get("session_id") or vector_id.split(":")[0])
            if stored == session_id:
                self.items.pop(vector_id, None)

    def delete_id(self, vector_id: str) -> None:
        if self.error is not None:
            raise self.error
        self.deleted.append(vector_id)
        self.items.pop(vector_id, None)

    def count(self) -> int:
        return len(self.items)


class FakeManager:
    """Менеджер фактов без базы: отдаёт заготовленный блок или ошибку."""

    def __init__(self, block: str = "", error: Exception | None = None) -> None:
        self.value = block
        self.error = error
        self.queries: list[str] = []
        self.related: list[dict] = []

    def block(self, query: str = "", *, related: list[dict] | None = None) -> str:
        self.queries.append(query)
        self.related = related or []
        if self.error is not None:
            raise self.error
        return self.value


class MemoryServiceTest(unittest.TestCase):
    def make(self, **kwargs) -> tuple[MemoryService, FakeEmbedder, FakeVectors, SessionStore]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        session = DialogSession(started_at=at(), id="s1")
        session.add("user", "привет")
        session.finish(at=at(9))
        store.save(session)
        store.save_summary(
            "s1",
            "Пользователь изучает C++ и циклы.",
            model="test",
            created_at=at(10),
        )
        embedder = FakeEmbedder()
        vectors = FakeVectors()
        # Старые тесты проверяют семантику и дайджест отдельно, поэтому
        # блок последних разговоров по умолчанию выключен.
        kwargs.setdefault("recent_limit", 0)
        memory = MemoryService(embedder, store, vectors, **kwargs)
        return memory, embedder, vectors, store

    def add_session(
        self,
        store: SessionStore,
        session_id: str,
        messages: int,
        text: str,
        *,
        minute: int = 1,
        started_at: datetime | None = None,
    ) -> None:
        session = DialogSession(started_at=started_at or at(minute), id=session_id)
        for index in range(messages):
            session.add(
                "user" if index % 2 == 0 else "assistant",
                f"реплика {index}",
                at=at(minute + 1) if started_at is None else started_at,
            )
        session.finish(at=at(minute + 2) if started_at is None else started_at)
        store.save(session)
        store.save_summary(
            session_id,
            text,
            created_at=at(minute + 3) if started_at is None else started_at,
        )

    # ------------------------------------------------------------------ write
    def test_remember_embeds_document(self) -> None:
        memory, embedder, vectors, _ = self.make()

        self.assertTrue(memory.remember("s1", "Пользователь изучает C++."))

        self.assertEqual(embedder.calls, [(["Пользователь изучает C++."], "document")])
        embedding, metadata = vectors.items["s1:0"]
        self.assertEqual(embedding, [1.0, 0.0, 0.0])
        self.assertEqual(metadata["embedding_model"], "fake-voyage")
        self.assertEqual(metadata["dim"], 3)

    def test_remember_skips_empty_summary(self) -> None:
        memory, embedder, vectors, _ = self.make()
        self.assertFalse(memory.remember("s1", "   "))
        self.assertEqual(embedder.calls, [])
        self.assertEqual(vectors.items, {})

    def test_embedding_error_does_not_raise(self) -> None:
        memory, embedder, vectors, _ = self.make()
        embedder.error = EmbeddingError("нет сети")

        with self.assertLogs("app.memory", level="WARNING"):
            self.assertFalse(memory.remember("s1", "текст"))

        self.assertEqual(vectors.items, {})

    def test_vector_store_error_does_not_raise(self) -> None:
        memory, _, vectors, _ = self.make()
        vectors.error = RuntimeError("диск")

        with self.assertLogs("app.memory", level="WARNING"):
            self.assertFalse(memory.remember("s1", "текст"))

    def test_forget_deletes_vector(self) -> None:
        memory, _, vectors, _ = self.make()
        memory.remember("s1", "текст")

        self.assertTrue(memory.forget("s1"))
        self.assertEqual(vectors.deleted, ["s1"])

    def test_remember_stores_model_and_version(self) -> None:
        memory, _, vectors, _ = self.make(index_version=2)

        memory.remember("s1", "текст")

        _, metadata = vectors.items["s1:0"]
        self.assertEqual(metadata["embedding_model"], "fake-voyage")
        self.assertEqual(metadata["index_version"], 2)

    def test_remember_indexes_each_topic(self) -> None:
        memory, embedder, vectors, _ = self.make()
        topics = [
            {"title": "учёба", "text": "Пользователь изучает C++."},
            {"title": "планы", "text": "Собирается на море."},
        ]

        self.assertTrue(memory.remember("s1", "Целиком.", topics=topics, created_at=at(10)))

        self.assertEqual(
            embedder.calls,
            [
                (
                    ["учёба. Пользователь изучает C++.", "планы. Собирается на море."],
                    "document",
                )
            ],
        )
        self.assertEqual(sorted(vectors.items), ["s1:0", "s1:1"])
        _, first = vectors.items["s1:0"]
        self.assertEqual(first["topic_no"], 0)
        self.assertEqual(first["topic_title"], "учёба")
        self.assertEqual(first["created_ts"], int(at(10).timestamp()))
        _, second = vectors.items["s1:1"]
        self.assertEqual(second["topic_no"], 1)
        self.assertEqual(second["topic_title"], "планы")

    def test_remember_without_topic_text_indexes_summary(self) -> None:
        memory, embedder, vectors, _ = self.make()

        memory.remember("s1", "Целиком.", topics=[{"title": "учёба"}], created_at=at(10))

        self.assertEqual(embedder.calls, [(["Целиком."], "document")])
        self.assertEqual(sorted(vectors.items), ["s1:0"])

    # ----------------------------------------------------------------- restore
    def test_restore_indexes_missing_summaries(self) -> None:
        memory, embedder, vectors, _ = self.make()

        self.assertEqual(memory.restore(), 1)
        self.assertIn("s1:0", vectors.items)
        self.assertEqual(embedder.calls, [(["Пользователь изучает C++ и циклы."], "document")])

        self.assertEqual(memory.restore(), 0)
        self.assertEqual(len(embedder.calls), 1)

    def test_restore_reindexes_other_model(self) -> None:
        memory, _, vectors, _ = self.make()
        vectors.items["s1:0"] = (
            [0.0, 1.0, 0.0],
            {"embedding_model": "old-model", "index_version": 1},
        )

        self.assertEqual(memory.restore(), 1)
        self.assertEqual(vectors.items["s1:0"][1]["embedding_model"], "fake-voyage")

    def test_restore_honors_index_version(self) -> None:
        memory, _, vectors, _ = self.make(index_version=2)
        vectors.items["s1:0"] = (
            [1.0, 0.0, 0.0],
            {"embedding_model": "fake-voyage", "index_version": 1},
        )

        self.assertEqual(memory.restore(), 1)
        self.assertEqual(vectors.items["s1:0"][1]["index_version"], 2)

    def test_restore_indexes_topics_from_v3(self) -> None:
        memory, embedder, vectors, store = self.make()
        store.save_summary(
            "s1",
            "Общее резюме.",
            created_at=at(10),
            data={
                "topics": [
                    {"title": "учёба", "text": "Пользователь изучает C++."},
                    {"title": "планы", "text": "Собирается на море."},
                ]
            },
        )

        self.assertEqual(memory.restore(), 2)
        self.assertEqual(
            embedder.calls,
            [
                (
                    ["учёба. Пользователь изучает C++.", "планы. Собирается на море."],
                    "document",
                )
            ],
        )
        self.assertEqual(sorted(vectors.items), ["s1:0", "s1:1"])
        self.assertEqual(vectors.items["s1:1"][1]["topic_title"], "планы")
        self.assertEqual(vectors.items["s1:1"][1]["created_ts"], int(at().timestamp()))

    def test_restore_removes_vectors_without_summary(self) -> None:
        memory, _, vectors, _ = self.make()
        vectors.items["gone"] = (
            [1.0, 0.0, 0.0],
            {"embedding_model": "fake-voyage", "index_version": 1},
        )

        self.assertEqual(memory.restore(), 1)
        self.assertIn("gone", vectors.deleted)

    def test_restore_survives_vector_store_error(self) -> None:
        memory, _, vectors, _ = self.make()
        vectors.error = RuntimeError("диск")

        with self.assertLogs("app.memory", level="WARNING"):
            self.assertEqual(memory.restore(), 0)

    # ------------------------------------------------------------------- read
    def test_recall_joins_summary_from_sqlite(self) -> None:
        memory, embedder, vectors, _ = self.make()
        vectors.hits = [VectorHit("s1", 0.8, {})]

        hits = memory.recall("что я изучал?")

        self.assertEqual(
            hits,
            [
                MemoryHit(
                    session_id="s1",
                    summary="Пользователь изучает C++ и циклы.",
                    score=0.8,
                    created_at=at().isoformat(),
                )
            ],
        )
        self.assertEqual(embedder.calls, [(["что я изучал?"], "query")])

    def test_recall_passes_model_and_version_filter(self) -> None:
        memory, _, vectors, _ = self.make(index_version=3)
        vectors.hits = [VectorHit("s1", 0.8, {})]

        memory.recall("вопрос")

        self.assertEqual(
            vectors.where,
            {"$and": [{"embedding_model": "fake-voyage"}, {"index_version": 3}]},
        )

    def test_recall_returns_topic_text_and_title(self) -> None:
        memory, _, vectors, store = self.make()
        store.save_summary(
            "s1",
            "Общее резюме.",
            data={
                "topics": [
                    {"title": "учёба", "text": "Пользователь изучает C++."},
                    {"title": "планы", "text": "Собирается на море."},
                ]
            },
        )
        vectors.hits = [VectorHit("s1:1", 0.8, {"session_id": "s1", "topic_no": 1})]

        hits = memory.recall("море")

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].summary, "Собирается на море.")
        self.assertEqual(hits[0].topic_title, "планы")
        self.assertEqual(hits[0].topic_no, 1)

    def test_recall_filters_day_in_chroma(self) -> None:
        memory, _, vectors, _ = self.make()
        vectors.hits = []

        memory.recall("вчера", day=date(2026, 11, 15))

        clauses = vectors.where["$and"]
        self.assertEqual(
            clauses[:2],
            [{"embedding_model": "fake-voyage"}, {"index_version": 2}],
        )
        self.assertIn("$gte", clauses[2]["created_ts"])
        self.assertIn("$lt", clauses[3]["created_ts"])
        self.assertLess(
            clauses[2]["created_ts"]["$gte"],
            clauses[3]["created_ts"]["$lt"],
        )

    def test_recall_skips_low_scores(self) -> None:
        memory, _, vectors, _ = self.make(min_score=0.5)
        vectors.hits = [VectorHit("s1", 0.39, {})]
        self.assertEqual(memory.recall("вопрос"), [])

    def test_recall_accepts_confident_best_below_threshold(self) -> None:
        memory, _, vectors, _ = self.make(min_score=0.35, min_margin=0.05)
        vectors.hits = [VectorHit("s1", 0.31, {}), VectorHit("s1", 0.22, {})]
        self.assertEqual(len(memory.recall("вопрос")), 1)

    def test_recall_rejects_flat_low_scores(self) -> None:
        memory, _, vectors, _ = self.make(min_score=0.35, min_margin=0.05)
        vectors.hits = [VectorHit("s1", 0.34, {}), VectorHit("s1", 0.32, {})]
        self.assertEqual(memory.recall("вопрос"), [])

    def test_recall_skips_unknown_sessions(self) -> None:
        memory, _, vectors, _ = self.make()
        vectors.hits = [VectorHit("нет-такой", 0.9, {})]
        self.assertEqual(memory.recall("вопрос"), [])

    def test_recall_prefers_fresh_close_summary(self) -> None:
        memory, _, vectors, store = self.make()
        old = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)
        self.add_session(store, "old", 2, "Старый разговор.", started_at=old)
        vectors.hits = [VectorHit("old", 0.6, {}), VectorHit("s1", 0.57, {})]

        hits = memory.recall("вопрос", now=at(30))

        self.assertEqual([hit.session_id for hit in hits], ["s1", "old"])

    def test_recall_keeps_clearly_better_old_summary(self) -> None:
        memory, _, vectors, store = self.make()
        old = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)
        self.add_session(store, "old", 2, "Старый разговор.", started_at=old)
        vectors.hits = [VectorHit("old", 0.9, {}), VectorHit("s1", 0.6, {})]

        hits = memory.recall("вопрос", now=at(30))

        self.assertEqual([hit.session_id for hit in hits], ["old", "s1"])

    def test_recall_respects_top_k(self) -> None:
        memory, _, vectors, _ = self.make(top_k=1)
        vectors.hits = [VectorHit("s1", 0.9, {}), VectorHit("s1", 0.8, {})]
        self.assertEqual(len(memory.recall("вопрос")), 1)

    def test_embedding_error_gives_empty_recall(self) -> None:
        memory, embedder, _, _ = self.make()
        embedder.error = EmbeddingError("нет сети")

        with self.assertLogs("app.memory", level="WARNING"):
            self.assertEqual(memory.recall("вопрос"), [])

    def test_vector_error_gives_empty_recall(self) -> None:
        memory, _, vectors, _ = self.make()
        vectors.error = RuntimeError("диск")

        with self.assertLogs("app.memory", level="WARNING"):
            self.assertEqual(memory.recall("вопрос"), [])

    def test_empty_query_gives_empty_recall(self) -> None:
        memory, embedder, _, _ = self.make()
        self.assertEqual(memory.recall("   "), [])
        self.assertEqual(embedder.calls, [])

    # ------------------------------------------------------------------ block
    def test_block_contains_header_summary_and_date(self) -> None:
        memory, _, vectors, _ = self.make()
        vectors.hits = [VectorHit("s1", 0.8, {})]

        block = memory.block("что я изучал?", now=at(20))

        self.assertTrue(block.startswith(MEMORY_HEADER))
        self.assertIn("Сегодня, 12:00", block)
        self.assertIn("Пользователь изучает C++ и циклы.", block)

    def test_block_is_empty_without_hits(self) -> None:
        memory, _, _, _ = self.make()
        self.assertEqual(memory.block("вопрос"), "")

    def test_block_respects_char_limit(self) -> None:
        memory, _, vectors, _ = self.make(max_chars=10)
        vectors.hits = [VectorHit("s1", 0.8, {})]
        self.assertEqual(memory.block("вопрос"), "")

    def test_block_puts_facts_before_memory_header(self) -> None:
        memory, _, vectors, _ = self.make()
        vectors.hits = [VectorHit("s1", 0.8, {})]
        memory.manager = FakeManager("Ключевые факты:\n- любит кофе")

        block = memory.block("что я люблю?")

        self.assertTrue(block.startswith("Ключевые факты:"))
        self.assertLess(block.index("Ключевые факты:"), block.index(MEMORY_HEADER))

    def test_block_returns_facts_without_summaries(self) -> None:
        memory, _, _, _ = self.make()
        memory.manager = FakeManager("Ключевые факты:\n- любит кофе")

        self.assertEqual(memory.block("вопрос"), "Ключевые факты:\n- любит кофе")

    def test_manager_error_does_not_break_memory(self) -> None:
        memory, _, vectors, _ = self.make()
        vectors.hits = [VectorHit("s1", 0.8, {})]
        memory.manager = FakeManager(error=RuntimeError("база занята"))

        with self.assertLogs("app.memory", level="WARNING"):
            block = memory.block("что я изучал?")

        self.assertTrue(block.startswith(MEMORY_HEADER))
        self.assertIn("Пользователь изучает C++ и циклы.", block)

    # ----------------------------------------------------------------- digest
    def test_digest_includes_today_long_sessions_only(self) -> None:
        memory, _, _, store = self.make(digest_min_messages=6)
        self.add_session(store, "long", 6, "Сегодняшний длинный разговор.")
        self.add_session(store, "short", 4, "Сегодняшний короткий разговор.", minute=5)
        yesterday = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
        self.add_session(
            store,
            "old",
            8,
            "Вчерашний разговор.",
            started_at=yesterday,
        )

        digest = memory.digest(now=at(20))

        self.assertEqual([hit.session_id for hit in digest], ["long"])
        self.assertEqual(digest[0].summary, "Сегодняшний длинный разговор.")

    def test_digest_respects_limit(self) -> None:
        memory, _, _, store = self.make(digest_min_messages=6, digest_limit=1)
        self.add_session(store, "first", 6, "Первый разговор.", minute=1)
        self.add_session(store, "second", 6, "Второй разговор.", minute=5)

        digest = memory.digest(now=at(20))

        self.assertEqual([hit.session_id for hit in digest], ["second"])

    def test_digest_can_be_disabled(self) -> None:
        memory, _, _, store = self.make(digest_min_messages=6, digest_limit=0)
        self.add_session(store, "long", 6, "Сегодняшний разговор.")

        self.assertEqual(memory.digest(now=at(20)), [])

    def test_digest_clips_long_summary(self) -> None:
        memory, _, _, store = self.make(
            digest_min_messages=6,
            digest_item_chars=20,
        )
        self.add_session(
            store,
            "long",
            6,
            "Первое предложение. Второе предложение.",
        )

        digest = memory.digest(now=at(20))

        self.assertTrue(digest[0].summary.endswith("…"))
        self.assertLessEqual(len(digest[0].summary), 21)

    def test_block_shows_timeline_without_duplicates(self) -> None:
        memory, _, _, store = self.make(recent_limit=2, digest_min_messages=6)
        self.add_session(store, "long", 6, "Сегодняшний разговор.")

        block = memory.block("что мы обсуждали?", now=at(20))

        lines = block.splitlines()
        self.assertTrue(lines[0].startswith("Эпизодическая память"))
        self.assertIn("- Сегодня, 12:01: Сегодняшний разговор.", lines)
        self.assertIn("- Сегодня, 12:00: Пользователь изучает C++ и циклы.", lines)
        self.assertLess(
            lines.index("- Сегодня, 12:01: Сегодняшний разговор."),
            lines.index("- Сегодня, 12:00: Пользователь изучает C++ и циклы."),
        )
        self.assertEqual(block.count("Сегодняшний разговор."), 1)

    def test_specific_query_skips_timeline(self) -> None:
        memory, _, vectors, store = self.make(recent_limit=1)
        self.add_session(store, "fresh", 4, "Свежий разговор.", minute=30)
        vectors.hits = [VectorHit("s1", 0.9, {})]

        lines = memory.block("что я изучал?", now=at(40)).splitlines()

        self.assertIn("- Сегодня, 12:00: Пользователь изучает C++ и циклы.", lines)
        self.assertNotIn("Свежий разговор.", lines)

    def test_block_puts_latest_summary_first(self) -> None:
        memory, _, _, store = self.make(recent_limit=2)
        self.add_session(store, "later", 4, "Поздний разговор.", minute=30)
        self.add_session(store, "earlier", 4, "Ранний разговор.", minute=10)

        lines = memory.block("что мы обсуждали?", now=at(40)).splitlines()

        self.assertLess(
            lines.index("- Сегодня, 12:30: Поздний разговор."),
            lines.index("- Сегодня, 12:10: Ранний разговор."),
        )
        self.assertNotIn("Пользователь изучает C++ и циклы.", lines)

    def test_recent_day_orders_last_topic_first(self) -> None:
        memory, _, _, store = self.make(recent_limit=1)
        self.add_session(store, "s2", 4, "Коротко.", started_at=at(20))
        store.save_summary(
            "s2",
            "Коротко.",
            created_at=at(20),
            data={
                "topics": [
                    {"title": "утро", "text": "Обсуждали утро."},
                    {"title": "вечер", "text": "Обсуждали вечер."},
                ]
            },
        )

        hits = memory.recent(at(30).date())

        self.assertEqual(
            [(hit.session_id, hit.topic_no) for hit in hits],
            [("s2", 1), ("s2", 0)],
        )
        self.assertEqual([hit.topic_title for hit in hits], ["вечер", "утро"])

        block = memory.block("что было сегодня?", now=at(30))
        self.assertLess(block.index("«вечер»"), block.index("«утро»"))

    def test_yesterday_query_orders_everything_by_time(self) -> None:
        memory, _, vectors, store = self.make(recent_limit=2)
        yesterday = datetime(2026, 11, 15, 19, 0, tzinfo=UTC)
        self.add_session(store, "old", 4, "Ранний разговор.", started_at=yesterday)
        self.add_session(
            store,
            "middle",
            4,
            "Средний разговор.",
            started_at=yesterday.replace(hour=20),
        )
        self.add_session(
            store,
            "late",
            4,
            "Поздний разговор.",
            started_at=yesterday.replace(hour=21),
        )
        vectors.hits = [VectorHit("old", 0.9, {})]

        lines = memory.block("что мы обсуждали вчера?", now=at(40)).splitlines()

        self.assertLess(
            lines.index("- Вчера, 21:00: Поздний разговор."),
            lines.index("- Вчера, 19:00: Ранний разговор."),
        )
        self.assertLess(
            lines.index("- Вчера, 20:00: Средний разговор."),
            lines.index("- Вчера, 19:00: Ранний разговор."),
        )

    def test_yesterday_query_shows_only_yesterday(self) -> None:
        memory, _, _, store = self.make(recent_limit=3)
        yesterday = datetime(2026, 11, 15, 19, 0, tzinfo=UTC)
        self.add_session(store, "old", 4, "Вчерашний разговор.", started_at=yesterday)
        self.add_session(store, "today", 4, "Сегодняшний разговор.", minute=30)

        block = memory.block("что мы обсуждали вчера?", now=at(40))

        self.assertIn("- Вчера, 19:00: Вчерашний разговор.", block)
        self.assertNotIn("Сегодняшний разговор.", block)

    def test_yesterday_query_drops_other_days(self) -> None:
        memory, _, vectors, _ = self.make(recent_limit=2)
        vectors.hits = [VectorHit("s1", 0.9, {})]

        self.assertEqual(memory.block("что было вчера?", now=at(20)), "")

    # ------------------------------------------------------------------ facts
    def add_fact(
        self,
        store: SessionStore,
        text: str,
        *,
        importance: str = "high",
        status: str = "active",
        moment: datetime | None = None,
    ) -> int:
        stamp = (moment or at(10)).isoformat()
        return store.add_memory_fact(
            session_id="s1",
            kind="fact",
            fact_key=text,
            text=text,
            evidence=text,
            importance=importance,
            confidence=0.9,
            status=status,
            created_at=stamp,
            updated_at=stamp,
        )

    def test_sync_facts_indexes_only_active_important(self) -> None:
        facts = FakeVectors()
        memory, _, _, store = self.make(facts_vectors=facts, manager=FakeManager())
        indexed = self.add_fact(store, "Пользователь изучает C++")
        self.add_fact(store, "Пользователь смотрел фильм", importance="low")
        old = self.add_fact(store, "Пользователь болеет", status="superseded")
        facts.items[f"fact:{old}"] = (
            [0.0, 1.0, 0.0],
            {"embedding_model": "fake-voyage", "index_version": 2},
        )

        self.assertEqual(memory.sync_facts(), 1)

        self.assertEqual(set(facts.items), {f"fact:{indexed}"})
        _, metadata = facts.items[f"fact:{indexed}"]
        self.assertEqual(metadata["fact_id"], indexed)
        self.assertEqual(metadata["fact_importance"], "high")
        self.assertEqual(memory.sync_facts(), 0)

    def test_recall_facts_filters_threshold_and_general_question(self) -> None:
        facts = FakeVectors()
        memory, _, _, store = self.make(
            facts_vectors=facts, manager=FakeManager(), facts_min_score=0.5
        )
        fact_id = self.add_fact(store, "Пользователь изучает C++")
        facts.hits = [VectorHit(f"fact:{fact_id}", 0.9, {"fact_id": fact_id})]

        found = memory.recall_facts("что я изучал?")

        self.assertEqual([row["id"] for row in found], [fact_id])
        self.assertEqual(found[0]["score"], 0.9)
        self.assertEqual(memory.recall_facts("что ты обо мне помнишь"), [])
        facts.hits = [VectorHit(f"fact:{fact_id}", 0.2, {"fact_id": fact_id})]
        self.assertEqual(memory.recall_facts("что я изучал?"), [])

    def test_facts_block_passes_semantic_hits_to_manager(self) -> None:
        facts = FakeVectors()
        manager = FakeManager(block="готовый блок")
        memory, _, _, store = self.make(facts_vectors=facts, manager=manager, facts_min_score=0.5)
        fact_id = self.add_fact(store, "Пользователь изучает C++")
        facts.hits = [VectorHit(f"fact:{fact_id}", 0.9, {"fact_id": fact_id})]

        self.assertEqual(memory.facts_block("что я изучал?"), "готовый блок")
        self.assertEqual([row["id"] for row in manager.related], [fact_id])

    # ----------------------------------------------------------------- config
    def test_from_config_disabled(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        self.assertIsNone(MemoryService.from_config({"memory": {"enabled": False}}, store))

    def test_from_config_without_key(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        with self.assertLogs("app.memory", level="WARNING"):
            self.assertIsNone(
                MemoryService.from_config(
                    {
                        "memory": {"enabled": True},
                        "cloud_llm": {"api_key": ""},
                        "tts": {"api_key": ""},
                    },
                    store,
                )
            )

    @unittest.skipUnless(HAS_CHROMA, "chromadb не установлен")
    def test_from_config_with_chroma(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        memory = MemoryService.from_config(
            {
                "memory": {
                    "enabled": True,
                    "path": str(Path(directory.name) / "chroma"),
                    "api_key": "key",
                }
            },
            store,
        )
        self.addCleanup(memory.close)
        self.assertIsNotNone(memory)
        self.assertEqual(memory.embedder.model, "mongodb/voyage-4")
        self.assertIsInstance(memory.vectors, ChromaVectorStore)
        self.assertEqual(memory.max_chars, 2400)
        self.assertEqual(memory.digest_min_messages, 6)
        self.assertEqual(memory.digest_limit, 3)
        self.assertEqual(memory.digest_item_chars, 320)
        self.assertEqual(memory.recent_limit, 2)
        self.assertEqual(memory.recent_item_chars, 320)


@unittest.skipUnless(HAS_CHROMA, "chromadb не установлен")
class ChromaMemoryTest(unittest.TestCase):
    def test_remember_and_recall_roundtrip(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        session = DialogSession(started_at=at(), id="s1")
        session.add("user", "привет")
        session.finish(at=at(9))
        store.save(session)
        store.save_summary("s1", "Пользователь изучает C++ и циклы.", created_at=at(10))
        memory = MemoryService(
            FakeEmbedder(),
            store,
            ChromaVectorStore(Path(directory.name) / "chroma"),
            min_score=0.5,
        )
        self.addCleanup(memory.close)

        self.assertTrue(memory.remember("s1", store.summary("s1")["summary"]))
        hits = memory.recall("что я изучал?")

        self.assertEqual([hit.session_id for hit in hits], ["s1"])
        self.assertGreater(hits[0].score, 0.9)
        self.assertIn("C++", memory.block("что я изучал?"))

        self.assertTrue(memory.forget("s1"))
        self.assertEqual(memory.recall("что я изучал?"), [])

    def test_stale_model_is_hidden_until_restored(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        session = DialogSession(started_at=at(), id="s1")
        session.add("user", "привет")
        session.finish(at=at(9))
        store.save(session)
        store.save_summary("s1", "Пользователь изучает C++ и циклы.", created_at=at(10))
        vectors = ChromaVectorStore(Path(directory.name) / "chroma")
        memory = MemoryService(FakeEmbedder(), store, vectors, min_score=0.5)
        self.addCleanup(memory.close)
        vectors.upsert(
            "s1",
            [1.0, 0.0, 0.0],
            {"embedding_model": "old-model", "index_version": 1},
        )

        self.assertEqual(memory.recall("что я изучал?"), [])
        self.assertEqual(memory.restore(), 1)
        self.assertEqual(
            [hit.session_id for hit in memory.recall("что я изучал?")],
            ["s1"],
        )

    def test_fact_search_roundtrip(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        manager = MemoryManager(store)
        path = Path(directory.name) / "chroma"
        memory = MemoryService(
            FakeEmbedder(),
            store,
            ChromaVectorStore(path),
            facts_vectors=ChromaVectorStore(path, "facts"),
            facts_min_score=0.5,
            min_score=0.5,
            manager=manager,
        )
        self.addCleanup(memory.close)
        session = DialogSession(started_at=at(), id="s1")
        session.add("user", "привет")
        session.finish(at=at(9))
        store.save(session)
        stamp = at(10).isoformat()
        store.add_memory_fact(
            session_id="s1",
            kind="fact",
            fact_key="учёба",
            text="Пользователь изучает C++",
            evidence="изучаю C++",
            importance="high",
            confidence=0.9,
            created_at=stamp,
            updated_at=stamp,
        )

        self.assertEqual(memory.sync_facts(), 1)

        hits = memory.recall_facts("что я изучал?")
        self.assertEqual([hit["text"] for hit in hits], ["Пользователь изучает C++"])
        self.assertIn("C++", memory.facts_block("что я изучал?"))


if __name__ == "__main__":
    unittest.main()
