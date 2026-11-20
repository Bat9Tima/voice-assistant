from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from app.character import (
    AUTO_HEADER,
    CHARACTER_HEADER,
    HUMOR,
    IRONY,
    LENGTH,
    MOMENT_HEADER,
    PROFANITY,
    STABLE_HEADER,
    TONE,
    CharacterMemory,
    detect,
)
from app.dialog import DialogService
from app.session_store import SessionStore
from app.style_observer import StyleProfile


def at(day: int = 20) -> datetime:
    return datetime(2026, 11, day, 12, 0, tzinfo=UTC)


def profile(
    *,
    messages: int = 30,
    words: int | None = None,
    chars: int | None = None,
    profanity_messages: int = 0,
    profanity_words: int = 0,
    rough_words: int = 0,
    polite_messages: int = 0,
    humor_messages: int = 0,
    irony_messages: int = 0,
    question_messages: int = 0,
) -> StyleProfile:
    """Профиль с заданными средними; по умолчанию 5 слов на реплику."""
    total_words = messages * 5 if words is None else words
    return StyleProfile(
        messages=messages,
        words=total_words,
        chars=total_words * 4 if chars is None else chars,
        profanity_messages=profanity_messages,
        profanity_words=profanity_words,
        rough_words=rough_words,
        polite_messages=polite_messages,
        humor_messages=humor_messages,
        irony_messages=irony_messages,
        question_messages=question_messages,
    )


class FakeLLM:
    def __init__(self, answer: str = "Ответ") -> None:
        self.answer = answer
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.calls.append((messages, kwargs))
        return self.answer


class CharacterMemoryTest(unittest.TestCase):
    def make(self, **kwargs) -> tuple[CharacterMemory, SessionStore]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        return CharacterMemory(store, **kwargs), store

    def test_explicit_request_is_stable_version(self) -> None:
        memory, store = self.make()

        answer = memory.handle("не матерись", now=at())

        self.assertIn("без мата", answer)
        current = memory.current(now=at())
        self.assertEqual(current["style"], {PROFANITY: "off"})
        self.assertEqual(current["moment"], {})
        self.assertEqual(current["version"], 1)
        history = store.character_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["quote"], "не матерись")
        self.assertEqual(history[0]["source"], "stable")
        self.assertIn(STABLE_HEADER, memory.block(now=at()))
        self.assertIn("без мата", memory.block(now=at()))

    def test_situational_request_expires_next_day(self) -> None:
        memory, _ = self.make()

        answer = memory.handle("сегодня без шуток", now=at())

        self.assertIn("на сегодня", answer)
        current = memory.current(now=at())
        self.assertEqual(current["style"], {})
        self.assertEqual(current["moment"], {HUMOR: "off"})
        block = memory.block(now=at())
        self.assertIn(MOMENT_HEADER, block)
        self.assertIn("без шуток", block)
        self.assertEqual(memory.current(now=at(21))["moment"], {})
        self.assertNotIn("без шуток", memory.block(now=at(21)))

    def test_stable_request_clears_same_daily_rule(self) -> None:
        memory, _ = self.make()
        memory.handle("сегодня без мата", now=at())

        memory.handle("можно материться", now=at())

        current = memory.current(now=at())
        self.assertEqual(current["style"], {PROFANITY: "on"})
        self.assertEqual(current["moment"], {})
        self.assertIn("можно материться", memory.block(now=at()))
        self.assertNotIn("без мата", memory.block(now=at()))

    def test_without_list_is_one_command(self) -> None:
        memory, _ = self.make()
        memory.handle("без мата и шуток", now=at())

        self.assertEqual(
            memory.current(now=at())["style"],
            {PROFANITY: "off", HUMOR: "off"},
        )

    def test_repeated_without_is_one_command(self) -> None:
        for text in ("без мата и без шуток", "без мата без шуток"):
            with self.subTest(text=text):
                memory, _ = self.make()
                memory.handle(text, now=at())

                self.assertEqual(
                    memory.current(now=at())["style"],
                    {PROFANITY: "off", HUMOR: "off"},
                )

    def test_multiple_commands_are_applied(self) -> None:
        memory, _ = self.make()

        memory.handle("отвечай короче и мягче", now=at())

        self.assertEqual(
            memory.current(now=at())["style"],
            {LENGTH: "short", TONE: "soft"},
        )

    def test_detect_returns_explicit_signal(self) -> None:
        signals = detect("Не матерись, пожалуйста")

        self.assertEqual(
            [(signal.dimension, signal.value) for signal in signals],
            [(PROFANITY, "off")],
        )
        self.assertEqual(detect("короче, я устал"), [])

    def test_false_positives_are_not_commands(self) -> None:
        memory, store = self.make()
        texts = (
            "мне нравится, когда ты не материшься",
            "он сказал не матерись",
            "без шуток я устал",
            "без шуток и я устал",
            "короче я устал",
            "короче и я устал",
            "мне не нравится мат",
            "как ты со мной общаешься вчера",
        )
        for text in texts:
            self.assertIsNone(memory.handle(text, now=at()), text)
        self.assertEqual(store.character_history(), [])

    def test_report_shows_both_layers(self) -> None:
        memory, _ = self.make()
        memory.handle("не матерись", now=at())
        memory.handle("сегодня без шуток", now=at())

        answer = memory.describe(now=at())

        self.assertIn("без мата", answer)
        self.assertIn("без шуток", answer)

    def test_report_without_requests_says_usual(self) -> None:
        memory, _ = self.make()

        self.assertIn("как обычно", memory.describe(now=at()))

    def test_rollback_returns_previous_version(self) -> None:
        memory, _ = self.make()
        memory.handle("не матерись", now=at())
        memory.handle("короче", now=at())

        answer = memory.rollback("верни прежнюю манеру", now=at())

        self.assertIn("без мата", answer)
        current = memory.current(now=at())
        self.assertEqual(current["style"], {PROFANITY: "off"})
        self.assertEqual(current["version"], 3)
        self.assertEqual(memory.store.character_history()[0]["source"], "rollback")

    def test_rollback_to_core_removes_all(self) -> None:
        memory, _ = self.make()
        memory.handle("не матерись", now=at())

        answer = memory.rollback("верни прежнюю манеру", now=at())

        self.assertIn("обычной манере", answer)
        self.assertEqual(memory.current(now=at())["style"], {})
        self.assertEqual(memory.current(now=at())["version"], 2)

    def test_rollback_skips_expired_moment(self) -> None:
        memory, _ = self.make()
        memory.handle("сегодня без шуток", now=at())
        memory.handle("короче", now=at())

        answer = memory.rollback("верни прежнюю манеру", now=at(21))

        self.assertIn("обычной манере", answer)
        current = memory.current(now=at(21))
        self.assertEqual(current["style"], {})
        self.assertEqual(current["moment"], {})

    def test_rollback_after_reset_restores_previous_manner(self) -> None:
        memory, _ = self.make()
        memory.handle("не матерись", now=at())
        memory.handle("короче", now=at())
        memory.reset("верни обычную манеру", now=at())

        answer = memory.rollback("верни прежнюю манеру", now=at())

        self.assertIn("без мата", answer)
        self.assertEqual(
            memory.current(now=at())["style"],
            {PROFANITY: "off", LENGTH: "short"},
        )

    def test_rollback_without_history_is_safe(self) -> None:
        memory, store = self.make()

        answer = memory.handle("верни прежнюю манеру", now=at())

        self.assertIn("Прежней манеры нет", answer)
        self.assertEqual(store.character_history(), [])

    def test_reset_removes_all_versions(self) -> None:
        memory, _ = self.make()
        memory.handle("не матерись", now=at())
        memory.handle("короче", now=at())

        answer = memory.reset("верни обычную манеру", now=at())

        self.assertIn("обычной манере", answer)
        current = memory.current(now=at())
        self.assertEqual(current["style"], {})
        self.assertEqual(current["moment"], {})
        self.assertEqual(current["version"], 3)

    def test_repeat_does_not_add_version(self) -> None:
        memory, store = self.make()
        memory.handle("не матерись", now=at())

        answer = memory.handle("не матерись", now=at())

        self.assertIn("Уже так", answer)
        self.assertEqual(len(store.character_history()), 1)

    def test_moment_is_persisted_for_today(self) -> None:
        memory, store = self.make()
        memory.handle("сегодня без шуток", now=at())

        again = CharacterMemory(store)

        self.assertEqual(again.current(now=at())["moment"], {HUMOR: "off"})
        self.assertEqual(again.current(now=at(21))["moment"], {})

    def test_disabled_character_does_not_handle(self) -> None:
        memory, store = self.make(enabled=False)

        self.assertIsNone(memory.handle("не матерись", now=at()))
        self.assertEqual(memory.block(now=at()), "")
        self.assertEqual(store.character_history(), [])

    def test_same_moment_is_not_repeated_in_block(self) -> None:
        memory, _ = self.make()
        memory.handle("не шути", now=at())
        memory.handle("сегодня не шути", now=at())

        block = memory.block(now=at())

        self.assertEqual(block.count("без шуток"), 1)
        self.assertNotIn(MOMENT_HEADER, block)

    def test_tense_marker_is_stable(self) -> None:
        memory, _ = self.make()

        memory.handle("теперь без мата", now=at())

        current = memory.current(now=at())
        self.assertEqual(current["style"], {PROFANITY: "off"})
        self.assertEqual(current["moment"], {})

    def test_polite_words_do_not_hide_command(self) -> None:
        memory, _ = self.make()

        answer = memory.handle("ну пожалуйста, не матерись", now=at())

        self.assertIn("без мата", answer)

    def test_mirror_needs_minimum_messages(self) -> None:
        memory, store = self.make()

        auto = memory.mirror(profile(messages=19, profanity_messages=19), now=at())

        self.assertIsNone(auto)
        self.assertEqual(store.character_history(), [])

    def test_mirror_enables_profanity(self) -> None:
        memory, store = self.make()

        auto = memory.mirror(
            profile(messages=20, profanity_messages=5, profanity_words=5),
            now=at(),
        )

        self.assertEqual(auto, {PROFANITY: "on"})
        self.assertEqual(memory.current(now=at())["auto"], {PROFANITY: "on"})
        block = memory.block(now=at())
        self.assertIn(AUTO_HEADER, block)
        self.assertIn("мат уместен", block)
        history = store.character_history()
        self.assertEqual(history[0]["source"], "auto")
        self.assertEqual(history[0]["quote"], "")
        self.assertEqual(history[0]["auto"], {PROFANITY: "on"})

    def test_mirror_hysteresis_keeps_then_turns_off_profanity(self) -> None:
        memory, _ = self.make()
        memory.mirror(
            profile(messages=30, profanity_messages=5, profanity_words=5),
            now=at(),
        )

        # 2 из 30 — это 6.7%: выше порога удержания 5%.
        self.assertIsNone(
            memory.mirror(
                profile(messages=30, profanity_messages=2, profanity_words=2),
                now=at(),
            )
        )
        self.assertEqual(memory.current(now=at())["auto"], {PROFANITY: "on"})

        # 1 из 30 — 3.3%: пора выключать.
        auto = memory.mirror(
            profile(messages=30, profanity_messages=1, profanity_words=1),
            now=at(),
        )

        self.assertEqual(auto, {})
        self.assertEqual(memory.current(now=at())["auto"], {})

    def test_mirror_length_short_and_detailed(self) -> None:
        memory, _ = self.make()

        short = memory.mirror(profile(messages=30, words=90), now=at())
        detailed = memory.mirror(profile(messages=30, words=420), now=at())

        self.assertEqual(short, {LENGTH: "short"})
        self.assertEqual(detailed, {LENGTH: "detailed"})
        self.assertIn("отвечаю подробнее", memory.block(now=at()))

    def test_mirror_length_has_dead_zone(self) -> None:
        memory, _ = self.make()

        auto = memory.mirror(profile(messages=30, words=240), now=at())

        self.assertIsNone(auto)
        self.assertEqual(memory.current(now=at())["auto"], {})

    def test_mirror_enables_humor_and_irony(self) -> None:
        memory, _ = self.make()

        auto = memory.mirror(
            profile(messages=30, humor_messages=4, irony_messages=3),
            now=at(),
        )

        self.assertEqual(auto, {HUMOR: "on", IRONY: "on"})
        block = memory.block(now=at())
        self.assertIn("шутки уместны", block)
        self.assertIn("ирония уместна", block)

    def test_mirror_humor_and_irony_hysteresis(self) -> None:
        memory, _ = self.make()
        memory.mirror(
            profile(messages=30, humor_messages=4, irony_messages=3),
            now=at(),
        )

        # 2 шутки из 30 — 6.7%, 1 ирония — 3.3%: выше порогов удержания.
        self.assertIsNone(
            memory.mirror(
                profile(messages=30, humor_messages=2, irony_messages=1),
                now=at(),
            )
        )
        self.assertEqual(
            memory.current(now=at())["auto"],
            {HUMOR: "on", IRONY: "on"},
        )

        auto = memory.mirror(
            profile(messages=30, humor_messages=1, irony_messages=0),
            now=at(),
        )

        self.assertEqual(auto, {})

    def test_mirror_tone_direct_from_roughness(self) -> None:
        memory, _ = self.make()

        auto = memory.mirror(profile(messages=30, rough_words=5), now=at())

        self.assertEqual(auto, {TONE: "direct"})
        self.assertIn("говорю жёстче и прямее", memory.block(now=at()))

    def test_mirror_tone_direct_hysteresis(self) -> None:
        memory, _ = self.make()
        memory.mirror(profile(messages=30, rough_words=5), now=at())

        self.assertIsNone(memory.mirror(profile(messages=30, rough_words=2), now=at()))
        self.assertEqual(memory.current(now=at())["auto"], {TONE: "direct"})

        auto = memory.mirror(profile(messages=30, rough_words=1), now=at())

        self.assertEqual(auto, {})

    def test_mirror_tone_soft_from_politeness(self) -> None:
        memory, _ = self.make()

        auto = memory.mirror(profile(messages=30, polite_messages=6), now=at())

        self.assertEqual(auto, {TONE: "soft"})
        self.assertIn("говорю мягче", memory.block(now=at()))

    def test_mirror_tone_soft_requires_clean_speech(self) -> None:
        memory, _ = self.make()

        auto = memory.mirror(
            profile(
                messages=30,
                polite_messages=10,
                profanity_messages=1,
                profanity_words=1,
            ),
            now=at(),
        )

        self.assertIsNone(auto)
        self.assertEqual(memory.current(now=at())["auto"], {})

    def test_mirror_combines_dimensions(self) -> None:
        memory, _ = self.make()

        auto = memory.mirror(
            profile(
                messages=30,
                profanity_messages=5,
                profanity_words=5,
                rough_words=5,
                humor_messages=4,
                irony_messages=3,
            ),
            now=at(),
        )

        self.assertEqual(
            auto,
            {
                PROFANITY: "on",
                TONE: "direct",
                HUMOR: "on",
                IRONY: "on",
            },
        )

    def test_mirror_does_not_repeat_version(self) -> None:
        memory, store = self.make()
        data = profile(messages=30, profanity_messages=5, profanity_words=5)
        memory.mirror(data, now=at())

        again = memory.mirror(data, now=at())

        self.assertIsNone(again)
        self.assertEqual(len(store.character_history()), 1)

    def test_explicit_request_hides_automatic_layer(self) -> None:
        memory, _ = self.make()
        memory.handle("не матерись", now=at())
        memory.mirror(
            profile(messages=30, profanity_messages=10, profanity_words=10),
            now=at(),
        )

        block = memory.block(now=at())
        description = memory.describe(now=at())

        self.assertIn("без мата", block)
        self.assertNotIn("мат уместен", block)
        self.assertNotIn("Автоматически", description)
        self.assertEqual(memory.current(now=at())["auto"], {PROFANITY: "on"})

    def test_explicit_request_hides_humor_and_irony(self) -> None:
        memory, _ = self.make()
        memory.handle("без шуток и иронии", now=at())
        memory.mirror(
            profile(messages=30, humor_messages=10, irony_messages=10),
            now=at(),
        )

        block = memory.block(now=at())

        self.assertIn("без шуток", block)
        self.assertIn("без иронии", block)
        self.assertNotIn("шутки уместны", block)
        self.assertNotIn("ирония уместна", block)

    def test_explicit_request_hides_tone(self) -> None:
        memory, _ = self.make()
        memory.handle("будь мягче", now=at())
        memory.mirror(profile(messages=30, rough_words=10), now=at())

        block = memory.block(now=at())

        self.assertIn("мягче", block)
        self.assertNotIn("говорю жёстче и прямее", block)

    def test_describe_reports_automatic_layer(self) -> None:
        memory, _ = self.make()
        memory.mirror(
            profile(messages=30, profanity_messages=10, profanity_words=10),
            now=at(),
        )

        description = memory.describe(now=at())

        self.assertIn("Автоматически", description)
        self.assertIn("мат уместен", description)

    def test_rollback_restores_automatic_layer(self) -> None:
        memory, _ = self.make()
        memory.mirror(
            profile(messages=30, profanity_messages=10, profanity_words=10),
            now=at(),
        )
        memory.handle("не матерись", now=at())

        answer = memory.rollback("верни прежнюю манеру", now=at())

        current = memory.current(now=at())
        self.assertIn("мат уместен", answer)
        self.assertEqual(current["style"], {})
        self.assertEqual(current["auto"], {PROFANITY: "on"})
        self.assertIn("мат уместен", memory.block(now=at()))

    def test_reset_clears_auto_and_observations(self) -> None:
        memory, store = self.make()
        store.add_style_observation(
            created_at=at().isoformat(),
            words=5,
            chars=20,
            profanity_words=1,
            profanity_messages=1,
        )
        memory.mirror(
            profile(messages=30, profanity_messages=10, profanity_words=10),
            now=at(),
        )

        answer = memory.reset("верни обычную манеру", now=at())

        self.assertIn("обычной манере", answer)
        self.assertEqual(memory.current(now=at())["auto"], {})
        self.assertEqual(store.style_observations(), [])
        self.assertEqual(memory.block(now=at()), "")

    def test_reset_without_adaptations_clears_observations(self) -> None:
        memory, store = self.make()
        store.add_style_observation(created_at=at().isoformat(), words=3, chars=10)

        answer = memory.reset("верни обычную манеру", now=at())

        self.assertIn("как обычно", answer)
        self.assertEqual(store.style_observations(), [])

    def test_resets_observations_recognizes_reset_and_rollback(self) -> None:
        memory, _ = self.make()

        self.assertTrue(memory.resets_observations("верни обычную манеру"))
        self.assertTrue(memory.resets_observations("верни прежнюю манеру"))
        self.assertFalse(memory.resets_observations("не матерись"))

    def test_rollback_clears_observations(self) -> None:
        memory, store = self.make()
        memory.mirror(profile(messages=30, profanity_messages=5), now=at())
        memory.handle("не матерись", now=at())
        store.add_style_observation(created_at=at().isoformat(), words=5, chars=20)

        memory.rollback("верни прежнюю манеру", now=at())

        self.assertEqual(store.style_observations(), [])

    def test_disabled_character_does_not_mirror(self) -> None:
        memory, store = self.make(enabled=False)

        auto = memory.mirror(profile(messages=30, profanity_messages=10), now=at())

        self.assertIsNone(auto)
        self.assertEqual(store.character_history(), [])


class DialogCharacterTest(unittest.TestCase):
    def make(self) -> tuple[DialogService, FakeLLM, CharacterMemory]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        memory = CharacterMemory(store)
        llm = FakeLLM()
        service = DialogService({"dialog": {}}, llm, character=memory)
        return service, llm, memory

    def test_character_command_is_answered_without_model(self) -> None:
        service, llm, memory = self.make()

        answer = service.reply("не матерись")

        self.assertIn("без мата", answer)
        self.assertEqual(llm.calls, [])
        self.assertEqual(len(service.sessions.current.messages), 2)
        self.assertIn("без мата", memory.block())

    def test_character_block_is_in_next_system_prompt(self) -> None:
        service, llm, _ = self.make()
        service.reply("не матерись")

        service.reply("привет")

        system = llm.calls[0][1]["system"]
        self.assertIn(CHARACTER_HEADER, system)
        self.assertIn("без мата", system)
        self.assertIn("реальная девушка", system)

    def test_character_block_follows_memory(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        memory = CharacterMemory(store)

        class Facts:
            def block(self, text: str) -> str:
                return "Ключевые факты: уместен мат"

        llm = FakeLLM()
        service = DialogService({"dialog": {}}, llm, memory=Facts(), character=memory)
        service.reply("не матерись")

        service.reply("привет")

        system = llm.calls[0][1]["system"]
        self.assertLess(system.index("Ключевые факты"), system.index(CHARACTER_HEADER))

    def test_empty_character_adds_no_block(self) -> None:
        service, llm, _ = self.make()

        service.reply("привет")

        system = llm.calls[0][1]["system"]
        self.assertNotIn(CHARACTER_HEADER, system)

    def test_broken_character_falls_back_to_model(self) -> None:
        class Broken:
            def handle(self, text: str) -> str:
                raise RuntimeError("boom")

            def block(self) -> str:
                return ""

        llm = FakeLLM("Ответ модели")
        service = DialogService({"dialog": {}}, llm, character=Broken())

        with self.assertLogs("app.dialog", level="WARNING"):
            answer = service.reply("привет")

        self.assertEqual(answer, "Ответ модели")
        self.assertEqual(len(llm.calls), 1)

    def test_broken_character_block_falls_back_to_model(self) -> None:
        class Broken:
            def handle(self, text: str) -> str | None:
                return None

            def block(self) -> str:
                raise RuntimeError("boom")

        llm = FakeLLM("Ответ модели")
        service = DialogService({"dialog": {}}, llm, character=Broken())

        with self.assertLogs("app.dialog", level="WARNING"):
            answer = service.reply("привет")

        self.assertEqual(answer, "Ответ модели")
        self.assertNotIn(CHARACTER_HEADER, llm.calls[0][1]["system"])

    def test_report_is_intercepted_without_model(self) -> None:
        service, llm, _ = self.make()
        service.reply("не матерись")

        answer = service.reply("как ты со мной общаешься")

        self.assertIn("без мата", answer)
        self.assertEqual(len(llm.calls), 0)

    def test_situational_command_uses_moment_header(self) -> None:
        service, _, memory = self.make()
        with patch("app.character._now", return_value=at()):
            answer = service.reply("сегодня без шуток")

            self.assertIn("на сегодня", answer)
            self.assertEqual(memory.current()["moment"], {HUMOR: "off"})
            self.assertIn(MOMENT_HEADER, memory.block())

    def test_day_boundary_keeps_stable_but_drops_moment(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(Path(directory.name) / "dialog.db")
        memory = CharacterMemory(store)
        memory.handle("не матерись", now=at())
        memory.handle("сегодня без шуток", now=at())

        current = memory.current(now=at() + timedelta(days=1))

        self.assertEqual(current["style"], {PROFANITY: "off"})
        self.assertEqual(current["moment"], {})
        self.assertEqual(len(store.character_history()), 2)
