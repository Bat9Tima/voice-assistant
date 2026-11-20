from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.character import CharacterMemory
from app.dialog import DialogService
from app.session_store import SessionStore
from app.style_observer import (
    MIN_MESSAGES,
    StyleObserver,
    StyleProfile,
    analyze,
)


def make_store(test: unittest.TestCase) -> SessionStore:
    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    return SessionStore(Path(directory.name) / "dialog.db")


class FakeLLM:
    def __init__(self, answer: str = "Ответ") -> None:
        self.answer = answer
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.calls.append((messages, kwargs))
        return self.answer


class AnalyzeTest(unittest.TestCase):
    def test_profanity_is_counted(self) -> None:
        sample = analyze("Ну ты и сука, блядь")

        self.assertEqual(sample.words, 5)
        self.assertEqual(sample.profanity_words, 2)
        self.assertEqual(sample.profanity_messages, 1)

    def test_plain_words_are_not_profanity(self) -> None:
        for text in (
            "хлеб и сукно",
            "бляха на ремне",
            "стеб и погреб",
            "хутор у реки",
            "мудрый совет",
            "ветка и сук",
            "сучья у дома",
            "похудание",
            "выхухоль",
        ):
            with self.subTest(text=text):
                self.assertEqual(analyze(text).profanity_words, 0)

    def test_prefixed_profanity_is_counted(self) -> None:
        for text in ("охуел", "нихуя", "похуист", "сучка", "пидрила"):
            with self.subTest(text=text):
                self.assertEqual(analyze(text).profanity_words, 1)

    def test_rough_words_are_separate_from_profanity(self) -> None:
        sample = analyze("это говно, ты идиот")

        self.assertEqual(sample.profanity_words, 0)
        self.assertEqual(sample.rough_words, 2)
        self.assertEqual(analyze("мразь и тварь").rough_words, 2)

    def test_laughter_is_humor_but_plain_words_are_not(self) -> None:
        self.assertEqual(analyze("ахаха ну ты даешь").humor_messages, 1)
        self.assertEqual(analyze("халат и хочу есть").humor_messages, 0)

    def test_humor_and_irony_markers(self) -> None:
        self.assertEqual(analyze("это шутка").humor_messages, 1)
        self.assertEqual(analyze("чистый сарказм").irony_messages, 1)
        self.assertEqual(analyze("ирония уместна").irony_messages, 1)

    def test_polite_markers_are_counted(self) -> None:
        self.assertEqual(analyze("пожалуйста, сделай это").polite_messages, 1)
        self.assertEqual(analyze("спасибо, ты лучшая").polite_messages, 1)
        self.assertEqual(analyze("сделай это").polite_messages, 0)

    def test_negated_humor_is_not_humor(self) -> None:
        for text in (
            "без шуток, я устал",
            "не шучу",
            "это не шутка",
            "не до шуток",
            "мне не смешно",
        ):
            with self.subTest(text=text):
                self.assertEqual(analyze(text).humor_messages, 0)

    def test_negated_irony_is_not_irony(self) -> None:
        for text in (
            "без иронии",
            "это не ирония",
            "без сарказма",
            "не иронизируй",
            "мне не иронично",
        ):
            with self.subTest(text=text):
                self.assertEqual(analyze(text).irony_messages, 0)

    def test_question_and_length(self) -> None:
        sample = analyze("как дела?")

        self.assertEqual(sample.words, 2)
        self.assertEqual(sample.chars, 9)
        self.assertEqual(sample.question, 1)

    def test_empty_text_gives_zeros(self) -> None:
        self.assertEqual(analyze("   "), analyze(""))


class StyleProfileTest(unittest.TestCase):
    def profile(self, *texts: str) -> StyleProfile:
        return StyleProfile.from_samples([analyze(text) for text in texts])

    def test_rates(self) -> None:
        profile = self.profile("сука", "как дела?", "да")

        self.assertEqual(profile.messages, 3)
        self.assertEqual(profile.words, 4)
        self.assertAlmostEqual(profile.profanity_message_rate, 1 / 3)
        self.assertAlmostEqual(profile.question_rate, 1 / 3)
        self.assertAlmostEqual(profile.avg_words, 4 / 3)
        self.assertAlmostEqual(profile.confidence, 3 / 50)

    def test_suggestions_need_enough_data(self) -> None:
        profile = self.profile(*(["сука"] * (MIN_MESSAGES - 1)))

        self.assertEqual(profile.suggestions(), {})

    def test_profanity_suggestion(self) -> None:
        profile = self.profile(*(["сука"] * 5 + ["привет"] * 15))

        self.assertEqual(profile.suggestions().get("profanity"), "on")

    def test_clean_history_suggests_no_profanity(self) -> None:
        profile = self.profile(*(["привет"] * 50))

        self.assertEqual(profile.suggestions().get("profanity"), "off")

    def test_short_and_detailed_length(self) -> None:
        short = self.profile(*(["да"] * 20))
        detailed = self.profile(
            *(
                [
                    "сегодня я долго думал о том, как лучше построить этот разговор "
                    "и что стоит сказать в конце"
                ]
                * 20
            )
        )

        self.assertEqual(short.suggestions().get("length"), "short")
        self.assertEqual(detailed.suggestions().get("length"), "detailed")

    def test_humor_and_irony_suggestions(self) -> None:
        profile = self.profile(*(["ахаха ирония"] * 4 + ["привет"] * 16))

        self.assertEqual(profile.suggestions().get("humor"), "on")
        self.assertEqual(profile.suggestions().get("irony"), "on")

    def test_rough_tone_suggestion(self) -> None:
        profile = self.profile(*(["говно и идиот"] * 5 + ["привет"] * 15))

        self.assertEqual(profile.suggestions().get("tone"), "direct")

    def test_soft_tone_suggestion(self) -> None:
        profile = self.profile(*(["пожалуйста, сделай это спокойно"] * 6 + ["привет"] * 14))

        self.assertEqual(profile.suggestions().get("tone"), "soft")

    def test_summary_mentions_metrics(self) -> None:
        summary = self.profile(*(["сука"] * 20)).summary()

        self.assertIn("Стиль:", summary)
        self.assertIn("мат 100%", summary)
        self.assertIn("profanity=on", summary)

    def test_empty_summary(self) -> None:
        self.assertIn("данных пока нет", StyleProfile().summary())


class StyleObserverTest(unittest.TestCase):
    def test_from_config(self) -> None:
        observer = StyleObserver.from_config(
            None,
            {
                "enabled": False,
                "window_messages": 5,
                "decision_messages": 3,
                "log_every": 2,
            },
        )

        self.assertFalse(observer.enabled)
        self.assertEqual(observer.window_messages, 5)
        self.assertEqual(observer.decision_messages, 3)
        self.assertEqual(observer.log_every, 2)

    def test_from_config_defaults(self) -> None:
        observer = StyleObserver.from_config(None)

        self.assertTrue(observer.enabled)
        self.assertEqual(observer.window_messages, 200)
        self.assertEqual(observer.decision_messages, 30)
        self.assertEqual(observer.log_every, 10)

    def test_recent_profile_uses_decision_window(self) -> None:
        observer = StyleObserver(None, window_messages=10, decision_messages=3)
        for text in ("сука", "привет", "привет", "привет"):
            observer.observe(text)

        self.assertEqual(observer.profile().messages, 4)
        self.assertEqual(observer.profile().profanity_messages, 1)
        self.assertEqual(observer.recent_profile().messages, 3)
        self.assertEqual(observer.recent_profile().profanity_messages, 0)

    def test_load_fills_recent_window(self) -> None:
        store = make_store(self)
        first = StyleObserver(store)
        for text in ("сука", "привет", "привет"):
            first.observe(text)

        second = StyleObserver(store, window_messages=10, decision_messages=2)

        self.assertEqual(second.profile().messages, 3)
        self.assertEqual(second.recent_profile().messages, 2)
        self.assertEqual(second.recent_profile().profanity_messages, 0)

    def test_reset_clears_history(self) -> None:
        store = make_store(self)
        observer = StyleObserver(store)
        observer.observe("сука")

        observer.reset()

        self.assertEqual(observer.profile().messages, 0)
        self.assertEqual(observer.recent_profile().messages, 0)
        self.assertEqual(store.style_observations(), [])

    def test_observe_accumulates_and_saves(self) -> None:
        store = make_store(self)
        observer = StyleObserver(store)

        sample = observer.observe("сука, как дела?")

        self.assertIsNotNone(sample)
        self.assertEqual(observer.profile().messages, 1)
        self.assertEqual(observer.profile().profanity_messages, 1)
        rows = store.style_observations()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["profanity_messages"], 1)
        self.assertNotIn("text", rows[0])

    def test_persistence_roundtrip(self) -> None:
        store = make_store(self)
        first = StyleObserver(store)
        first.observe("сука")
        first.observe("привет")

        second = StyleObserver(store)

        self.assertEqual(second.profile().messages, 2)
        self.assertEqual(second.profile().profanity_messages, 1)

    def test_window_limits_loaded_history(self) -> None:
        store = make_store(self)
        first = StyleObserver(store)
        for text in ("один", "два", "сука"):
            first.observe(text)

        second = StyleObserver(store, window_messages=2)

        self.assertEqual(second.profile().messages, 2)
        self.assertEqual(second.profile().profanity_messages, 1)

    def test_disabled_observer_does_nothing(self) -> None:
        store = make_store(self)
        observer = StyleObserver(store, enabled=False)

        self.assertIsNone(observer.observe("сука"))
        self.assertEqual(observer.profile().messages, 0)
        self.assertEqual(store.style_observations(), [])

    def test_empty_text_is_skipped(self) -> None:
        store = make_store(self)
        observer = StyleObserver(store)

        self.assertIsNone(observer.observe("   "))
        self.assertEqual(observer.profile().messages, 0)
        self.assertEqual(store.style_observations(), [])

    def test_broken_store_does_not_break_observation(self) -> None:
        class BrokenStore:
            def style_observations(self, limit: int = 200) -> list[dict]:
                raise RuntimeError("boom")

            def add_style_observation(self, **kwargs) -> int:
                raise RuntimeError("boom")

        with self.assertLogs("app.style_observer", level="WARNING"):
            observer = StyleObserver(BrokenStore())

        with self.assertLogs("app.style_observer", level="WARNING"):
            observer.observe("сука")

        self.assertEqual(observer.profile().messages, 1)
        self.assertEqual(observer.profile().profanity_messages, 1)

    def test_logs_every_n_messages(self) -> None:
        observer = StyleObserver(None, log_every=2)

        with self.assertLogs("app.style_observer", level="INFO") as logs:
            observer.observe("привет")
            observer.observe("привет")

        self.assertEqual(len(logs.records), 1)
        self.assertIn("Стиль:", logs.records[0].getMessage())

    def test_flush_logs_partial_window(self) -> None:
        observer = StyleObserver(None, log_every=10)
        observer.observe("привет")

        with self.assertLogs("app.style_observer", level="INFO") as logs:
            observer.flush()
            observer.flush()

        self.assertEqual(len(logs.records), 1)

    def test_profile_is_a_copy(self) -> None:
        observer = StyleObserver(None)
        observer.observe("привет")

        profile = observer.profile()
        profile.messages = 99

        self.assertEqual(observer.profile().messages, 1)


class DialogStyleTest(unittest.TestCase):
    def make(
        self,
        *,
        character: bool = False,
        style: StyleObserver | None = None,
    ) -> tuple[DialogService, FakeLLM, StyleObserver]:
        store = make_store(self)
        observer = style if style is not None else StyleObserver(store)
        llm = FakeLLM()
        memory = CharacterMemory(store) if character else None
        service = DialogService(
            {"dialog": {}},
            llm,
            character=memory,
            style=observer,
        )
        return service, llm, observer

    def test_normal_reply_is_observed(self) -> None:
        service, llm, observer = self.make()

        service.reply("сука, как дела?")

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(observer.profile().messages, 1)
        self.assertEqual(observer.profile().profanity_messages, 1)

    def test_character_command_is_not_observed(self) -> None:
        service, llm, observer = self.make(character=True)

        service.reply("не матерись")

        self.assertEqual(llm.calls, [])
        self.assertEqual(observer.profile().messages, 0)

    def test_shadow_does_not_change_prompt(self) -> None:
        service, llm, observer = self.make(character=True)
        service.reply("не матерись")

        service.reply("сука, привет")

        system = llm.calls[0][1]["system"]
        self.assertIn("Манера общения", system)
        self.assertEqual(observer.profile().messages, 1)

    def make_mirror(
        self,
    ) -> tuple[DialogService, FakeLLM, StyleObserver, CharacterMemory, SessionStore]:
        store = make_store(self)
        character = CharacterMemory(store)
        observer = StyleObserver(store)
        llm = FakeLLM()
        # Резюме в фоне тоже зовёт llm.chat и сбивает проверку последнего
        # вызова. Для автоподстройки оно не нужно.
        service = DialogService(
            {"dialog": {"context": {"rolling_summary": {"enabled": False}}}},
            llm,
            character=character,
            style=observer,
        )
        return service, llm, observer, character, store

    def test_auto_mirroring_appears_in_prompt(self) -> None:
        service, llm, _, character, _ = self.make_mirror()
        for _ in range(20):
            service.reply("сука опять это говно не работает")

        system = llm.calls[-1][1]["system"]

        self.assertIn("Автоматически, по речи пользователя", system)
        self.assertIn("мат уместен", system)
        self.assertEqual(character.current()["auto"].get("profanity"), "on")

    def test_explicit_command_overrides_auto_mirroring(self) -> None:
        service, llm, _, character, _ = self.make_mirror()
        for _ in range(20):
            service.reply("сука опять это говно не работает")
        service.reply("не матерись")

        service.reply("сука опять это говно не работает")

        system = llm.calls[-1][1]["system"]
        self.assertIn("без мата", system)
        self.assertNotIn("мат уместен", system)
        self.assertEqual(character.current()["auto"].get("profanity"), "on")

    def test_reset_clears_auto_mirroring(self) -> None:
        service, llm, _, character, store = self.make_mirror()
        for _ in range(20):
            service.reply("сука опять это говно не работает")

        service.reply("верни обычную манеру")

        self.assertEqual(store.style_observations(), [])
        self.assertEqual(character.current()["auto"], {})
        service.reply("привет")
        system = llm.calls[-1][1]["system"]
        self.assertNotIn("Автоматически, по речи пользователя", system)

    def test_auto_humor_appears_in_prompt(self) -> None:
        service, llm, _, character, _ = self.make_mirror()
        for _ in range(20):
            service.reply("ахаха, ну ты выдал, это смешно")

        system = llm.calls[-1][1]["system"]

        self.assertIn("шутки уместны", system)
        self.assertEqual(character.current()["auto"].get("humor"), "on")

    def test_auto_irony_and_direct_tone_appear_in_prompt(self) -> None:
        service, llm, _, character, _ = self.make_mirror()
        for _ in range(20):
            service.reply("ну конечно, чистая ирония, и ты ведёшь себя как идиот")

        system = llm.calls[-1][1]["system"]

        self.assertIn("ирония уместна", system)
        self.assertIn("говорю жёстче и прямее", system)
        self.assertEqual(character.current()["auto"].get("tone"), "direct")

    def test_auto_soft_tone_appears_in_prompt(self) -> None:
        service, llm, _, character, _ = self.make_mirror()
        for _ in range(20):
            service.reply("пожалуйста, сделай это спокойно и без спешки")

        system = llm.calls[-1][1]["system"]

        self.assertIn("говорю мягче", system)
        self.assertEqual(character.current()["auto"].get("tone"), "soft")

    def test_rollback_resets_style_window(self) -> None:
        service, _, observer, character, store = self.make_mirror()
        for _ in range(20):
            service.reply("сука опять это говно не работает")
        for _ in range(30):
            service.reply("сегодня хорошая погода и мы гуляем в парке")
        self.assertEqual(character.current()["auto"], {})

        service.reply("верни прежнюю манеру")

        self.assertEqual(character.current()["auto"].get("profanity"), "on")
        self.assertEqual(store.style_observations(), [])
        self.assertEqual(observer.recent_profile().messages, 0)

        service.reply("привет")
        self.assertEqual(character.current()["auto"].get("profanity"), "on")

    def test_mirror_failure_does_not_break_reply(self) -> None:
        class BrokenCharacter:
            def handle(self, text: str) -> None:
                return None

            def block(self) -> str:
                return ""

            def mirror(self, profile) -> None:
                raise RuntimeError("boom")

        service = DialogService(
            {"dialog": {}},
            FakeLLM("Ответ модели"),
            character=BrokenCharacter(),
            style=StyleObserver(None),
        )

        with self.assertLogs("app.dialog", level="WARNING"):
            answer = service.reply("привет")

        self.assertEqual(answer, "Ответ модели")

    def test_observer_failure_does_not_break_reply(self) -> None:
        class Broken:
            def observe(self, text: str, **kwargs) -> None:
                raise RuntimeError("boom")

            def flush(self) -> None:
                raise RuntimeError("boom")

        service = DialogService({"dialog": {}}, FakeLLM("Ответ модели"), style=Broken())

        with self.assertLogs("app.dialog", level="WARNING"):
            answer = service.reply("привет")

        self.assertEqual(answer, "Ответ модели")

    def test_stop_flushes_observer(self) -> None:
        service, _, observer = self.make()
        observer.log_every = 100
        service.reply("привет")

        with self.assertLogs("app.style_observer", level="INFO") as logs:
            service.stop()

        self.assertEqual(len(logs.records), 1)

    def test_disabled_observer_is_not_called(self) -> None:
        store = make_store(self)
        observer = StyleObserver(store, enabled=False)
        service, llm, _ = self.make(style=observer)

        service.reply("сука")

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(observer.profile().messages, 0)
        self.assertEqual(store.style_observations(), [])


if __name__ == "__main__":
    unittest.main()
