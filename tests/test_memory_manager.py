from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from app.memory_manager import (
    FACTS_HEADER,
    NEEDS_REVIEW,
    REVIEW_HEADER,
    MemoryManager,
    memory_key,
)
from app.session import DialogSession
from app.session_store import SessionStore
from app.summary_schema import StructuredSummary, SummaryItem


def at(minute: int = 0) -> datetime:
    return datetime(2026, 11, 17, 12, minute, tzinfo=UTC)


def make_summary(
    *,
    facts: list[tuple[str, str]] | None = None,
    decisions: list[tuple[str, str]] | None = None,
    unfinished: list[tuple[str, str]] | None = None,
) -> StructuredSummary:
    return StructuredSummary(
        facts=[SummaryItem(text=text, evidence=evidence) for text, evidence in facts or []],
        decisions=[
            SummaryItem(text=text, evidence=evidence)
            for text, evidence in decisions or []
        ],
        unfinished=[
            SummaryItem(text=text, evidence=evidence)
            for text, evidence in unfinished or []
        ],
    )


class MemoryManagerTest(unittest.TestCase):
    def make(self, **kwargs) -> tuple[MemoryManager, SessionStore]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        return MemoryManager(store, **kwargs), store

    def add_session(self, store: SessionStore, session_id: str, minute: int = 0) -> None:
        session = DialogSession(started_at=at(minute), id=session_id)
        session.add("user", "привет", at=at(minute))
        session.add("assistant", "здравствуй", at=at(minute + 1))
        session.finish(at=at(minute + 2))
        store.save(session)

    def test_ingest_stores_facts_and_hides_low(self) -> None:
        manager, store = self.make()
        self.add_session(store, "s1")
        data = make_summary(
            facts=[
                ("Пользователю очень плохо", "мне очень плохо"),
                ("Пользователь играет в Fortnite", "играю в фортнайт"),
            ]
        )

        result = manager.ingest("s1", data, created_at=at())

        self.assertEqual(result.added, 2)
        self.assertEqual(result.active, 2)
        self.assertEqual(store.memory_fact_count(), 2)
        self.assertEqual(
            {row["text"] for row in manager.facts()},
            {"Пользователю очень плохо"},
        )
        low = next(
            row
            for row in store.memory_facts()
            if row["text"] == "Пользователь играет в Fortnite"
        )
        self.assertEqual(low["importance"], "low")

    def test_meta_uncertainty_is_not_stored_as_fact(self) -> None:
        manager, store = self.make()
        self.add_session(store, "s1")
        manager.ingest(
            "s1",
            make_summary(
                facts=[
                    (
                        "Не ясно, есть ли у пользователя мысли о том, чтобы навредить себе",
                        "не знаю",
                    ),
                    (
                        "Не выяснено, есть ли рядом близкий человек",
                        "не знаю",
                    ),
                    ("Пользователь любит кофе", "обожаю кофе"),
                ]
            ),
            created_at=at(),
        )

        self.assertEqual(
            [row["text"] for row in store.memory_facts()],
            ["Пользователь любит кофе"],
        )

    def test_ingest_is_idempotent_for_session(self) -> None:
        manager, store = self.make()
        self.add_session(store, "s1")
        data = make_summary(facts=[("Пользователь любит кофе", "обожаю кофе")])

        manager.ingest("s1", data, created_at=at())
        manager.ingest("s1", data, created_at=at(5))

        self.assertEqual(store.memory_fact_count(), 1)
        self.assertEqual(len(manager.facts()), 1)

    def test_similar_fact_is_merged(self) -> None:
        manager, store = self.make()
        self.add_session(store, "s1")
        self.add_session(store, "s2", minute=5)
        manager.ingest(
            "s1",
            make_summary(facts=[("Пользователь любит кофе", "обожаю кофе")]),
            created_at=at(),
        )

        result = manager.ingest(
            "s2",
            make_summary(
                facts=[
                    (
                        "Пользователь очень любит кофе по утрам",
                        "я очень люблю кофе по утрам",
                    )
                ]
            ),
            created_at=at(5),
        )

        self.assertEqual(result.merged, 1)
        self.assertEqual(result.added, 0)
        facts = store.memory_facts()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["text"], "Пользователь очень любит кофе по утрам")

    def test_ambiguous_opposite_goes_to_review(self) -> None:
        manager, store = self.make()
        self.add_session(store, "s1")
        self.add_session(store, "s2", minute=5)
        manager.ingest(
            "s1",
            make_summary(facts=[("Пользователь любит кофе", "обожаю кофе")]),
            created_at=at(),
        )

        result = manager.ingest(
            "s2",
            make_summary(facts=[("Пользователь не любит кофе", "не люблю кофе")]),
            created_at=at(5),
        )

        self.assertEqual(result.review, 1)
        self.assertEqual(result.superseded, 0)
        self.assertEqual(result.added, 0)
        rows = store.memory_facts(limit=10)
        old = next(row for row in rows if row["text"] == "Пользователь любит кофе")
        new = next(row for row in rows if row["text"] == "Пользователь не любит кофе")
        self.assertEqual(old["status"], NEEDS_REVIEW)
        self.assertEqual(new["status"], NEEDS_REVIEW)
        self.assertEqual(old["conflict_with"], new["id"])
        self.assertEqual(new["conflict_with"], old["id"])
        self.assertEqual(manager.facts(), [])
        block = manager.block("что я люблю?")
        self.assertIn(REVIEW_HEADER, block)
        self.assertIn("Пользователь любит кофе", block)
        self.assertIn("Пользователь не любит кофе", block)

    def test_explicit_change_supersedes_old(self) -> None:
        manager, store = self.make()
        self.add_session(store, "s1")
        self.add_session(store, "s2", minute=5)
        manager.ingest(
            "s1",
            make_summary(facts=[("Пользователь любит кофе", "обожаю кофе")]),
            created_at=at(),
        )

        result = manager.ingest(
            "s2",
            make_summary(
                facts=[
                    (
                        "Пользователь больше не любит кофе",
                        "больше не люблю кофе",
                    )
                ]
            ),
            created_at=at(5),
        )

        self.assertEqual(result.superseded, 1)
        self.assertEqual(result.added, 1)
        self.assertEqual(result.review, 0)
        rows = store.memory_facts(limit=10)
        old = next(row for row in rows if row["text"] == "Пользователь любит кофе")
        new = next(
            row for row in rows if row["text"] == "Пользователь больше не любит кофе"
        )
        self.assertEqual(old["status"], "superseded")
        self.assertEqual(old["replaced_by"], new["id"])
        self.assertEqual(new["status"], "active")
        self.assertEqual([row["text"] for row in manager.facts()], [new["text"]])

    def test_recovery_updates_health_state(self) -> None:
        manager, store = self.make()
        self.add_session(store, "s1")
        self.add_session(store, "s2", minute=5)
        manager.ingest(
            "s1",
            make_summary(facts=[("Пользователь болеет", "я болею")]),
            created_at=at(),
        )

        result = manager.ingest(
            "s2",
            make_summary(
                facts=[("Пользователь уже выздоровел", "я уже выздоровел")]
            ),
            created_at=at(5),
        )

        self.assertEqual(result.superseded, 1)
        self.assertEqual(result.review, 0)
        self.assertEqual(
            [row["text"] for row in manager.facts()],
            ["Пользователь уже выздоровел"],
        )
        block = manager.block("как я?")
        self.assertNotIn(REVIEW_HEADER, block)

    def test_repeat_confirmation_dismisses_review(self) -> None:
        manager, store = self.make()
        self.add_session(store, "s1")
        self.add_session(store, "s2", minute=5)
        self.add_session(store, "s3", minute=10)
        manager.ingest(
            "s1",
            make_summary(facts=[("Пользователь любит кофе", "обожаю кофе")]),
            created_at=at(),
        )
        manager.ingest(
            "s2",
            make_summary(facts=[("Пользователь не любит кофе", "не люблю кофе")]),
            created_at=at(5),
        )

        result = manager.ingest(
            "s3",
            make_summary(facts=[("Пользователь не любит кофе", "не люблю кофе")]),
            created_at=at(10),
        )

        self.assertEqual(result.review, 0)
        review_rows = store.memory_facts(status=NEEDS_REVIEW, limit=10)
        self.assertEqual(review_rows, [])
        active = manager.facts()
        self.assertEqual([row["text"] for row in active], ["Пользователь не любит кофе"])
        self.assertNotIn(REVIEW_HEADER, manager.block("кофе"))

    def test_health_state_expires_without_marker(self) -> None:
        manager, store = self.make()
        self.add_session(store, "s1")
        self.add_session(store, "s2", minute=5)
        manager.ingest(
            "s1",
            make_summary(facts=[("Пользователь болеет", "я болею")]),
            created_at=at(),
        )

        result = manager.ingest(
            "s2",
            make_summary(facts=[("Пользователь здоров", "я здоров")]),
            created_at=datetime(2026, 11, 19, 12, 0, tzinfo=UTC),
        )

        self.assertEqual(result.superseded, 1)
        self.assertEqual(result.review, 0)
        rows = store.memory_facts(limit=10)
        old = next(row for row in rows if row["text"] == "Пользователь болеет")
        new = next(row for row in rows if row["text"] == "Пользователь здоров")
        self.assertEqual(old["durability"], "temporary")
        self.assertEqual(old["status"], "superseded")
        self.assertEqual(old["replaced_by"], new["id"])
        self.assertEqual(new["durability"], "temporary")
        self.assertEqual(new["status"], "active")
        self.assertEqual([row["text"] for row in manager.facts()], [new["text"]])

    def test_same_session_health_conflict_goes_to_review(self) -> None:
        manager, store = self.make()
        self.add_session(store, "s1")
        manager.ingest(
            "s1",
            make_summary(
                facts=[
                    ("Пользователь болеет", "я болею"),
                    ("Пользователь здоров", "я здоров"),
                ]
            ),
            created_at=at(),
        )

        rows = store.memory_facts(status=NEEDS_REVIEW, limit=10)
        self.assertEqual(len(rows), 2)
        self.assertEqual(manager.facts(), [])
        self.assertIn(REVIEW_HEADER, manager.block("как я?"))

    def test_preference_does_not_expire_with_time(self) -> None:
        manager, store = self.make()
        self.add_session(store, "s1")
        self.add_session(store, "s2", minute=5)
        manager.ingest(
            "s1",
            make_summary(facts=[("Пользователь любит кофе", "обожаю кофе")]),
            created_at=at(),
        )

        result = manager.ingest(
            "s2",
            make_summary(facts=[("Пользователь не любит кофе", "не люблю кофе")]),
            created_at=datetime(2026, 12, 17, 12, 0, tzinfo=UTC),
        )

        self.assertEqual(result.review, 1)
        self.assertEqual(result.superseded, 0)
        self.assertEqual(
            {row["durability"] for row in store.memory_facts(limit=10)},
            {"permanent"},
        )
        rows = store.memory_facts(status=NEEDS_REVIEW, limit=10)
        self.assertEqual(len(rows), 2)

    def test_decision_resolves_matching_unfinished_task(self) -> None:
        manager, store = self.make()
        self.add_session(store, "s1")
        self.add_session(store, "s2", minute=5)
        manager.ingest(
            "s1",
            make_summary(
                unfinished=[
                    (
                        "Пользователь планирует закончить проект Луна",
                        "надо закончить проект луна",
                    )
                ]
            ),
            created_at=at(),
        )

        result = manager.ingest(
            "s2",
            make_summary(
                decisions=[
                    (
                        "Пользователь решил закончить проект Луна",
                        "решил закончить проект луна",
                    )
                ]
            ),
            created_at=at(5),
        )

        self.assertEqual(result.resolved, 1)
        rows = store.memory_facts(limit=10)
        task = next(row for row in rows if row["kind"] == "unfinished")
        decision = next(row for row in rows if row["kind"] == "decision")
        self.assertEqual(task["status"], "resolved")
        self.assertEqual(task["resolved_by"], decision["id"])

    def test_reingest_keeps_task_resolved_by_existing_decision(self) -> None:
        manager, store = self.make()
        self.add_session(store, "s1")
        self.add_session(store, "s2", minute=5)
        task = make_summary(
            unfinished=[
                (
                    "Пользователь планирует закончить проект Луна",
                    "надо закончить проект луна",
                )
            ]
        )
        manager.ingest("s1", task, created_at=at())
        manager.ingest(
            "s2",
            make_summary(
                decisions=[
                    (
                        "Пользователь решил закончить проект Луна",
                        "решил закончить проект луна",
                    )
                ]
            ),
            created_at=at(5),
        )

        manager.ingest("s1", task, created_at=at(10))

        refreshed = next(
            row for row in store.memory_facts(limit=10) if row["kind"] == "unfinished"
        )
        self.assertEqual(refreshed["status"], "resolved")

    def test_block_has_header_and_skips_low(self) -> None:
        manager, store = self.make()
        self.add_session(store, "s1")
        manager.ingest(
            "s1",
            make_summary(
                facts=[
                    ("Пользователь любит кофе", "обожаю кофе"),
                    ("Пользователь смотрел фильм", "смотрел фильм"),
                ]
            ),
            created_at=at(),
        )

        block = manager.block("что я люблю?")

        self.assertTrue(block.startswith(FACTS_HEADER))
        self.assertIn("17.11.2026", block)
        self.assertIn("Пользователь любит кофе", block)
        self.assertNotIn("смотрел фильм", block)

    def test_block_is_empty_without_important_facts(self) -> None:
        manager, store = self.make()
        self.add_session(store, "s1")
        manager.ingest(
            "s1",
            make_summary(facts=[("Пользователь смотрел фильм", "смотрел фильм")]),
            created_at=at(),
        )

        self.assertEqual(manager.block("вопрос"), "")

    def test_facts_respect_limit(self) -> None:
        manager, store = self.make(facts_limit=1)
        self.add_session(store, "s1")
        manager.ingest(
            "s1",
            make_summary(
                facts=[
                    ("Пользователь любит кофе", "обожаю кофе"),
                    ("Пользователь любит чай", "обожаю чай"),
                ]
            ),
            created_at=at(),
        )

        self.assertEqual(len(manager.facts()), 1)

    def test_block_respects_char_limit(self) -> None:
        manager, store = self.make(facts_max_chars=len(FACTS_HEADER) + 5)
        self.add_session(store, "s1")
        manager.ingest(
            "s1",
            make_summary(facts=[("Пользователь любит кофе", "обожаю кофе")]),
            created_at=at(),
        )

        self.assertEqual(manager.block(), "")

    def test_from_config_reads_limits(self) -> None:
        manager, _ = self.make()
        configured = MemoryManager.from_config(
            manager.store, {"facts_limit": 2, "facts_max_chars": 100}
        )

        self.assertEqual(configured.facts_limit, 2)
        self.assertEqual(configured.facts_max_chars, 100)
        self.assertEqual(configured.merge_similarity, 0.4)

    def test_memory_key_groups_topics(self) -> None:
        self.assertEqual(memory_key("fact", "Пользователь любит кофе"), "preference")
        self.assertEqual(memory_key("fact", "Пользователь устал"), "health")
        self.assertTrue(
            memory_key("unfinished", "Закончить проект").startswith("task:")
        )


if __name__ == "__main__":
    unittest.main()
