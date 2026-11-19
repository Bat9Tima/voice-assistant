from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from app.cloud_llm import LLMError
from app.persona import PersonaMemory
from app.prompts import SELF_FACTS_SYSTEM, SUMMARY_SYSTEM
from app.session import DialogSession, SessionMemory
from app.session_store import SessionStore
from app.summarizer import Summarizer


def at(minute: int = 0) -> datetime:
    return datetime(2026, 11, 16, 12, minute, tzinfo=UTC)


def make_session(messages: list[tuple[str, str]] | None = None) -> DialogSession:
    session = DialogSession(started_at=at(), id="s1")
    for role, content in messages or [
        ("user", "привет"),
        ("assistant", "здравствуй"),
    ]:
        session.add(role, content, at=at(1))
    session.finish(at=at(9))
    return session


def make_crashed(store: SessionStore, session_id: str = "crash") -> DialogSession:
    """Сессия, оборванная сбоем: в базе active, finish не вызывался."""
    session = DialogSession(started_at=at(), id=session_id)
    store.create(session)
    for position, (role, text) in enumerate([("user", "привет"), ("assistant", "здравствуй")]):
        message = session.add(role, text, at=at(position + 1))
        store.append(session, message, position)
    return session


class FakeLLM:
    """Клиент модели без сети: записывает вызовы и отдаёт заготовку."""

    model = "fake-model"

    def __init__(
        self, answer: str = "Говорили о C++ и циклах.", error: Exception | None = None
    ) -> None:
        self.answer_text = answer
        self.error = error
        self.calls: list[dict[str, object]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.calls.append({"messages": messages, **kwargs})
        if self.error is not None:
            raise self.error
        return self.answer_text


class SummarizerTest(unittest.TestCase):
    def make(self, **kwargs) -> tuple[Summarizer, SessionStore, FakeLLM]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        llm = kwargs.pop("llm", None) or FakeLLM()
        kwargs.setdefault("background", False)
        return Summarizer(llm, store, **kwargs), store, llm

    # --------------------------------------------------------------- summary
    def test_summarize_saves_summary(self) -> None:
        summarizer, store, llm = self.make()
        session = make_session()
        store.save(session)

        text = summarizer.summarize(session)

        self.assertEqual(text, "Говорили о C++ и циклах.")
        self.assertEqual(
            store.summary("s1"),
            {
                "session_id": "s1",
                "summary": "Говорили о C++ и циклах.",
                "created_at": store.summary("s1")["created_at"],
                "model": "fake-model",
                "prompt_version": 3,
                "data": {},
                "started_at": at().isoformat(),
            },
        )
        call = llm.calls[0]
        self.assertEqual(call["system"], SUMMARY_SYSTEM)
        self.assertEqual(call["max_tokens"], 700)
        self.assertEqual(call["temperature"], 0.2)
        self.assertEqual(call["timeout"], 45.0)
        self.assertIsNone(call["model"])
        self.assertEqual(
            call["messages"],
            [
                {
                    "role": "user",
                    "content": "Пользователь: привет\nАссистент: здравствуй",
                }
            ],
        )

    def test_model_and_prompt_version_are_recorded(self) -> None:
        summarizer, store, _ = self.make(model="summary-model", prompt_version=3)
        session = make_session()
        store.save(session)

        summarizer.summarize(session)

        saved = store.summary("s1")
        self.assertEqual(saved["model"], "summary-model")
        self.assertEqual(saved["prompt_version"], 3)

    def test_structured_summary_is_validated_and_rendered(self) -> None:
        class StructuredLLM(FakeLLM):
            def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
                self.calls.append({"messages": messages, **kwargs})
                if kwargs.get("system") == SUMMARY_SYSTEM:
                    return (
                        '{"facts": ['
                        '{"text": "Пользователю очень плохо", "evidence": "мне очень плохо"},'
                        '{"text": "Пользователь любит кофе", "evidence": "обожаю кофе"}'
                        '], "topics": [{"title": "самочувствие", '
                        '"text": "Пользователю плохо, он любит кофе."}], '
                        '"summary": "Говорили о самочувствии."}'
                    )
                return "нет фактов"

        class FakeMemory:
            def __init__(self) -> None:
                self.saved: list[tuple[str, str]] = []
                self.topics: list[dict] = []
                self.created_at: datetime | None = None

            def remember(
                self,
                session_id: str,
                summary: str,
                *,
                topics: list[dict] | None = None,
                created_at: datetime | None = None,
            ) -> bool:
                self.saved.append((session_id, summary))
                self.topics = topics or []
                self.created_at = created_at
                return True

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        memory = FakeMemory()
        summarizer = Summarizer(StructuredLLM(), store, background=False, memory=memory)
        session = make_session(
            [
                ("user", "мне очень плохо"),
                ("assistant", "сочувствую"),
                ("user", "обожаю кофе"),
            ]
        )
        store.save(session)

        text = summarizer.summarize(session)

        self.assertEqual(
            text,
            "Факты о пользователе:\n"
            "- Пользователю очень плохо\n"
            "- Пользователь любит кофе\n"
            "Темы: самочувствие\n"
            "Говорили о самочувствии.",
        )
        saved = store.summary("s1")
        self.assertEqual(saved["summary"], text)
        self.assertEqual(saved["prompt_version"], 3)
        self.assertEqual(
            saved["data"]["facts"],
            [
                {"text": "Пользователю очень плохо", "evidence": "мне очень плохо"},
                {"text": "Пользователь любит кофе", "evidence": "обожаю кофе"},
            ],
        )
        self.assertEqual(
            saved["data"]["topics"],
            [{"title": "самочувствие", "text": "Пользователю плохо, он любит кофе."}],
        )
        self.assertEqual(memory.saved, [("s1", text)])
        self.assertEqual(
            memory.topics,
            [{"title": "самочувствие", "text": "Пользователю плохо, он любит кофе."}],
        )
        self.assertEqual(memory.created_at, at())

    def test_unconfirmed_fact_is_not_saved(self) -> None:
        answer = (
            '{"facts": [{"text": "Пользователь любит кофе", "evidence": "обожаю кофе"}],'
            ' "summary": "Короткий разговор."}'
        )
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        summarizer = Summarizer(FakeLLM(answer=answer), store, background=False)
        session = make_session()
        store.save(session)

        text = summarizer.summarize(session)

        self.assertEqual(text, "Короткий разговор.")
        self.assertEqual(store.summary("s1")["data"]["facts"], [])

    def test_short_session_is_skipped(self) -> None:
        summarizer, store, llm = self.make()
        session = make_session([("user", "привет")])
        store.save(session)

        summarizer.handle(session)

        self.assertEqual(llm.calls, [])
        self.assertIsNone(store.summary("s1"))

    def test_disabled_summarizer_does_nothing(self) -> None:
        summarizer, store, llm = self.make(enabled=False)
        session = make_session()
        store.save(session)

        summarizer.handle(session)

        self.assertEqual(llm.calls, [])
        self.assertIsNone(store.summary("s1"))

    def test_empty_answer_is_not_saved(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        session = make_session()
        store.save(session)
        summary = Summarizer(FakeLLM(answer="   "), store, background=False)

        with self.assertRaises(ValueError):
            summary.summarize(session)

        self.assertIsNone(store.summary("s1"))

    def test_llm_error_keeps_raw_session(self) -> None:
        summarizer, store, _ = self.make()
        summarizer.llm = FakeLLM(error=LLMError("нет сети"))
        session = make_session()
        store.save(session)

        with self.assertRaises(LLMError):
            summarizer.summarize(session)

        self.assertEqual(store.count(), 1)
        self.assertEqual(len(store.messages("s1")), 2)
        self.assertIsNone(store.summary("s1"))

    # -------------------------------------------------------------- transcript
    def test_long_transcript_keeps_head_and_tail(self) -> None:
        summarizer, _, _ = self.make(max_input_chars=45)
        session = make_session(
            [
                ("user", "старая длинная реплика про погоду и дела"),
                ("assistant", "старый ответ"),
                ("user", "новая реплика"),
            ]
        )

        text = summarizer.transcript(session)

        self.assertIn("\n…\n", text)
        self.assertTrue(text.startswith("Пользователь:"))
        self.assertIn("новая реплика", text)
        self.assertNotIn("погоду", text)
        self.assertLessEqual(len(text), 45)

    def test_short_transcript_is_not_cut(self) -> None:
        summarizer, _, _ = self.make(max_input_chars=200)
        session = make_session(
            [
                ("user", "привет"),
                ("assistant", "здравствуй"),
            ]
        )

        self.assertEqual(
            summarizer.transcript(session),
            "Пользователь: привет\nАссистент: здравствуй",
        )

    def test_transcript_without_limit_keeps_everything(self) -> None:
        summarizer, _, _ = self.make(max_input_chars=0)
        session = make_session()

        self.assertEqual(
            summarizer.transcript(session),
            "Пользователь: привет\nАссистент: здравствуй",
        )

    # ------------------------------------------------------------------ hooks
    def test_attach_saves_after_finish(self) -> None:
        summarizer, store, llm = self.make()
        memory = SessionMemory()
        store.attach(memory)
        summarizer.attach(memory)

        memory.add("user", "привет")
        memory.add("assistant", "здравствуй")
        memory.finish()

        session = memory.finished[0]
        self.assertEqual(store.count(), 1)
        self.assertEqual(store.summary(session.id)["summary"], "Говорили о C++ и циклах.")
        self.assertEqual(len(llm.calls), 1)

    def test_background_worker_writes_before_close(self) -> None:
        summarizer, store, _ = self.make(background=True)
        memory = SessionMemory()
        store.attach(memory)
        summarizer.attach(memory)

        memory.add("user", "привет")
        memory.add("assistant", "здравствуй")
        session = memory.finish()
        summarizer.close()

        self.assertEqual(store.summary(session.id)["summary"], "Говорили о C++ и циклах.")

    def test_failed_summary_does_not_break_exit(self) -> None:
        summarizer, store, _ = self.make()
        summarizer.llm = FakeLLM(error=LLMError("нет сети"))
        memory = SessionMemory()
        store.attach(memory)
        summarizer.attach(memory)

        memory.add("user", "привет")
        memory.add("assistant", "здравствуй")
        with self.assertLogs("app.session", level="ERROR"):
            session = memory.finish()

        self.assertIsNone(memory.current)
        self.assertEqual(store.messages(session.id)[0]["content"], "привет")
        self.assertIsNone(store.summary(session.id))

    # ----------------------------------------------------------------- config
    def test_from_config_defaults(self) -> None:
        summarizer = Summarizer.from_config(FakeLLM(), object())

        self.assertTrue(summarizer.enabled)
        self.assertTrue(summarizer.background)
        self.assertEqual(summarizer.max_input_chars, 6000)
        self.assertEqual(summarizer.max_tokens, 700)
        self.assertEqual(summarizer.timeout, 45.0)
        self.assertEqual(summarizer.min_messages, 2)
        self.assertEqual(summarizer.model, "")

    # ----------------------------------------------------------------- memory
    def test_summary_is_indexed_in_memory(self) -> None:
        class FakeMemory:
            def __init__(self) -> None:
                self.saved: list[tuple[str, str]] = []

            def remember(
                self,
                session_id: str,
                summary: str,
                *,
                topics: list[dict] | None = None,
                created_at: datetime | None = None,
            ) -> bool:
                self.saved.append((session_id, summary))
                return True

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        memory = FakeMemory()
        summarizer = Summarizer(FakeLLM(), store, background=False, memory=memory)
        session = make_session()
        store.save(session)

        summarizer.summarize(session)

        self.assertEqual(memory.saved, [("s1", "Говорили о C++ и циклах.")])

    def test_memory_failure_does_not_block_summary(self) -> None:
        class BrokenMemory:
            def remember(
                self,
                session_id: str,
                summary: str,
                *,
                topics: list[dict] | None = None,
                created_at: datetime | None = None,
            ) -> bool:
                raise RuntimeError("Chroma умерла")

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        summarizer = Summarizer(FakeLLM(), store, background=False, memory=BrokenMemory())
        session = make_session()
        store.save(session)

        with self.assertLogs("app.summarizer", level="ERROR"):
            text = summarizer.summarize(session)

        self.assertEqual(text, "Говорили о C++ и циклах.")
        self.assertEqual(store.summary("s1")["summary"], text)

    def test_self_facts_are_saved_to_persona(self) -> None:
        class PersonaLLM(FakeLLM):
            def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
                self.calls.append({"messages": messages, **kwargs})
                if kwargs.get("system") == SELF_FACTS_SYSTEM:
                    return "- Мне 23 года.\n- Живу в Москве."
                return "Говорили о C++ и циклах."

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        persona = PersonaMemory(Path(directory.name) / "luna.txt")
        llm = PersonaLLM()
        summarizer = Summarizer(llm, store, background=False, persona=persona)
        session = make_session()
        store.save(session)

        summarizer.summarize(session)

        self.assertEqual(persona.facts(), ["Мне 23 года.", "Живу в Москве."])
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[1]["system"], SELF_FACTS_SYSTEM)
        self.assertEqual(llm.calls[1]["temperature"], 0.0)

    def test_persona_failure_does_not_block_summary(self) -> None:
        class BrokenPersonaLLM(FakeLLM):
            def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
                if kwargs.get("system") == SELF_FACTS_SYSTEM:
                    raise RuntimeError("нет сети")
                return super().chat(messages, **kwargs)

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        persona = PersonaMemory(Path(directory.name) / "luna.txt")
        summarizer = Summarizer(
            BrokenPersonaLLM(),
            store,
            background=False,
            persona=persona,
        )
        session = make_session()
        store.save(session)

        with self.assertLogs("app.summarizer", level="ERROR"):
            text = summarizer.summarize(session)

        self.assertEqual(text, "Говорили о C++ и циклах.")
        self.assertEqual(store.summary("s1")["summary"], text)
        self.assertEqual(persona.facts(), [])

    def test_from_config_passes_memory(self) -> None:
        class FakeMemory:
            def remember(
                self,
                session_id: str,
                summary: str,
                *,
                topics: list[dict] | None = None,
                created_at: datetime | None = None,
            ) -> bool:
                return True

        memory = FakeMemory()
        summarizer = Summarizer.from_config(FakeLLM(), object(), memory=memory)
        self.assertIs(summarizer.memory, memory)

    def test_from_config_passes_persona(self) -> None:
        persona = PersonaMemory(None)
        summarizer = Summarizer.from_config(FakeLLM(), object(), persona=persona)
        self.assertIs(summarizer.persona, persona)

    def test_prose_summary_does_not_touch_manager(self) -> None:
        class FakeManager:
            def __init__(self) -> None:
                self.calls = []

            def ingest(self, session_id, summary) -> None:
                self.calls.append((session_id, summary))

        manager = FakeManager()
        summarizer, store, _ = self.make(manager=manager)
        session = make_session()
        store.save(session)

        text = summarizer.summarize(session)

        self.assertEqual(text, "Говорили о C++ и циклах.")
        self.assertEqual(manager.calls, [])

    def test_manager_receives_confirmed_items(self) -> None:
        class FakeManager:
            def __init__(self) -> None:
                self.calls = []

            def ingest(self, session_id, summary):
                self.calls.append((session_id, summary))
                return SimpleNamespace(added=1, merged=0, superseded=0, resolved=0, review=0)

        answer = (
            '{"facts": [{"text": "Пользователь любит кофе", "evidence": "обожаю кофе"}],'
            ' "decisions": [{"text": "Пользователь решил завести кота",'
            ' "evidence": "решил завести кота"}],'
            ' "summary": "Говорили о кофе."}'
        )
        manager = FakeManager()
        summarizer, store, _ = self.make(manager=manager, llm=FakeLLM(answer=answer))
        session = make_session(
            [
                ("user", "обожаю кофе"),
                ("assistant", "здорово"),
                ("user", "решил завести кота"),
            ]
        )
        store.save(session)

        with self.assertLogs("app.summarizer", level="INFO"):
            summarizer.summarize(session)

        session_id, structured = manager.calls[0]
        self.assertEqual(session_id, "s1")
        self.assertEqual([item.text for item in structured.facts], ["Пользователь любит кофе"])
        self.assertEqual(
            [item.text for item in structured.decisions],
            ["Пользователь решил завести кота"],
        )

    def test_manager_failure_does_not_block_summary(self) -> None:
        class BrokenManager:
            def ingest(self, session_id, summary) -> None:
                raise RuntimeError("база занята")

        answer = (
            '{"facts": [{"text": "Пользователь любит кофе", "evidence": "обожаю кофе"}],'
            ' "summary": "Говорили о кофе."}'
        )
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        summarizer = Summarizer(
            FakeLLM(answer=answer),
            store,
            background=False,
            manager=BrokenManager(),
        )
        session = make_session([("user", "обожаю кофе")])
        store.save(session)

        with self.assertLogs("app.summarizer", level="ERROR"):
            text = summarizer.summarize(session)

        self.assertEqual(
            text,
            "Факты о пользователе:\n- Пользователь любит кофе\nГоворили о кофе.",
        )
        self.assertEqual(store.summary("s1")["summary"], text)

    def test_from_config_passes_manager(self) -> None:
        manager = object()
        summarizer = Summarizer.from_config(FakeLLM(), object(), manager=manager)
        self.assertIs(summarizer.manager, manager)

    # ------------------------------------------------------------- recovery
    def test_recover_summarizes_crashed_session(self) -> None:
        summarizer, store, llm = self.make()
        make_crashed(store)

        self.assertEqual(summarizer.recover(), 1)

        self.assertEqual(store.sessions()[0]["status"], "completed")
        self.assertEqual(store.summary("crash")["summary"], "Говорили о C++ и циклах.")
        self.assertEqual(len(llm.calls), 1)

    def test_recover_skips_session_with_summary(self) -> None:
        summarizer, store, llm = self.make()
        make_crashed(store)
        store.save_summary("crash", "Резюме уже было.")

        self.assertEqual(summarizer.recover(), 1)

        self.assertEqual(store.summary("crash")["summary"], "Резюме уже было.")
        self.assertEqual(llm.calls, [])

    def test_recover_closes_session_when_disabled(self) -> None:
        summarizer, store, llm = self.make(enabled=False)
        make_crashed(store)

        self.assertEqual(summarizer.recover(), 1)

        self.assertEqual(store.active(), [])
        self.assertIsNone(store.summary("crash"))
        self.assertEqual(llm.calls, [])

    def test_recover_in_background_queues_summary(self) -> None:
        summarizer, store, _ = self.make(background=True)
        make_crashed(store)

        self.assertEqual(summarizer.recover(), 1)
        summarizer.close(timeout=5)

        self.assertIsNotNone(store.summary("crash"))

    def test_from_config_overrides(self) -> None:
        summarizer = Summarizer.from_config(
            FakeLLM(),
            object(),
            {
                "enabled": False,
                "model": "summary-model",
                "max_input_chars": 0,
                "max_tokens": 120,
                "temperature": 0.0,
                "timeout_sec": 5,
                "min_messages": 4,
                "prompt_version": 2,
                "background": False,
            },
        )

        self.assertFalse(summarizer.enabled)
        self.assertFalse(summarizer.background)
        self.assertEqual(summarizer.model, "summary-model")
        self.assertEqual(summarizer.max_input_chars, 0)
        self.assertEqual(summarizer.max_tokens, 120)
        self.assertEqual(summarizer.temperature, 0.0)
        self.assertEqual(summarizer.timeout, 5.0)
        self.assertEqual(summarizer.min_messages, 4)
        self.assertEqual(summarizer.prompt_version, 2)


if __name__ == "__main__":
    unittest.main()
