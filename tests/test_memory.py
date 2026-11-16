from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from app.embeddings import EmbeddingError
from app.memory import (
    MEMORY_HEADER,
    ChromaVectorStore,
    MemoryHit,
    MemoryService,
    VectorHit,
)
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

    def delete(self, session_id: str) -> None:
        if self.error is not None:
            raise self.error
        self.deleted.append(session_id)
        self.items.pop(session_id, None)

    def count(self) -> int:
        return len(self.items)


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
        memory = MemoryService(embedder, store, vectors, **kwargs)
        return memory, embedder, vectors, store

    # ------------------------------------------------------------------ write
    def test_remember_embeds_document(self) -> None:
        memory, embedder, vectors, _ = self.make()

        self.assertTrue(memory.remember("s1", "Пользователь изучает C++."))

        self.assertEqual(embedder.calls, [(["Пользователь изучает C++."], "document")])
        embedding, metadata = vectors.items["s1"]
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

        _, metadata = vectors.items["s1"]
        self.assertEqual(metadata["embedding_model"], "fake-voyage")
        self.assertEqual(metadata["index_version"], 2)

    # ----------------------------------------------------------------- restore
    def test_restore_indexes_missing_summaries(self) -> None:
        memory, embedder, vectors, _ = self.make()

        self.assertEqual(memory.restore(), 1)
        self.assertIn("s1", vectors.items)
        self.assertEqual(embedder.calls, [(["Пользователь изучает C++ и циклы."], "document")])

        self.assertEqual(memory.restore(), 0)
        self.assertEqual(len(embedder.calls), 1)

    def test_restore_reindexes_other_model(self) -> None:
        memory, _, vectors, _ = self.make()
        vectors.items["s1"] = (
            [0.0, 1.0, 0.0],
            {"embedding_model": "old-model", "index_version": 1},
        )

        self.assertEqual(memory.restore(), 1)
        self.assertEqual(vectors.items["s1"][1]["embedding_model"], "fake-voyage")

    def test_restore_honors_index_version(self) -> None:
        memory, _, vectors, _ = self.make(index_version=2)
        vectors.items["s1"] = (
            [1.0, 0.0, 0.0],
            {"embedding_model": "fake-voyage", "index_version": 1},
        )

        self.assertEqual(memory.restore(), 1)
        self.assertEqual(vectors.items["s1"][1]["index_version"], 2)

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
                    created_at=at(10).isoformat(),
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

        block = memory.block("что я изучал?")

        self.assertTrue(block.startswith(MEMORY_HEADER))
        self.assertIn("16.11.2026 12:10", block)
        self.assertIn("Пользователь изучает C++ и циклы.", block)

    def test_block_is_empty_without_hits(self) -> None:
        memory, _, _, _ = self.make()
        self.assertEqual(memory.block("вопрос"), "")

    def test_block_respects_char_limit(self) -> None:
        memory, _, vectors, _ = self.make(max_chars=10)
        vectors.hits = [VectorHit("s1", 0.8, {})]
        self.assertEqual(memory.block("вопрос"), "")

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


if __name__ == "__main__":
    unittest.main()
