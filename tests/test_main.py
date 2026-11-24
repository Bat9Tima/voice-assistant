from __future__ import annotations

import threading
import time
import unittest
from collections import deque
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from PIL import Image

from app.actions import CommandResult
from app.addressee import (
    ADDRESSED,
    BACKGROUND,
    CLOSING,
    AddresseeDecision,
)
from app.addressee import (
    MODE_ACTIVE as ADDRESSEE_ACTIVE,
)
from app.addressee import (
    MODE_SHADOW as ADDRESSEE_SHADOW,
)
from app.frame_processor import ChangeResult
from app.image_search import (
    DEFAULT_TRIGGERS as IMAGE_TRIGGERS,
)
from app.image_search import (
    ImageMatch,
    ImageSearchError,
    ImageSearchResult,
)
from app.image_search import (
    match_request as match_image_request,
)
from app.intent_classifier import (
    ANSWER,
    DIALOG,
    INFO,
    MODE_ACTIVE,
    MODE_SHADOW,
    SOURCE_FALLBACK,
    SOURCE_MODEL,
    UNKNOWN,
    DialogState,
    IntentDecision,
)
from app.local_planner import LocalPlan, LocalPlannerError
from app.main import Assistant, _hotkey_key, _make_hotkey_listener
from app.qa import QAAnswer, QAService, Route
from app.screen_capture import Frame
from app.screen_watch import WatchEvent
from app.search import SearchError, SearchResult
from app.vision import DEFAULT_TRIGGERS, VisionError, match_request
from app.vision_comments import CommentStatus
from app.vision_events import (
    DEFAULT_FRAME_GAP_SEC,
    DEFAULT_IDLE_AFTER_SEC,
    DEFAULT_MAX_EVENTS,
    DEFAULT_MAX_FRAMES,
    EventTimeline,
)
from app.vision_watch import WatchStatus
from app.visual_intent import OTHER, SCREEN
from app.visual_router import VisualRouter
from app.wake import BargeInMatch
from app.weather import DailyForecast, GeoPlace, WeatherError, WeatherReport


class FakeRouter:
    def __init__(self, result: CommandResult | None = None) -> None:
        self.result = result
        self.seen: list[str] = []
        self.plan_result = CommandResult(True, True, "Выполнено")
        self.plans: list[tuple[str, dict]] = []

    def handle(self, text: str) -> CommandResult | None:
        self.seen.append(text)
        return self.result

    def execute_plan(self, action: str, args: dict) -> CommandResult:
        self.plans.append((action, args))
        return self.plan_result


class FakeLocalPlanner:
    def __init__(self, plan: LocalPlan | None = None, error: Exception | None = None) -> None:
        self.plan_result = plan
        self.error = error
        self.seen: list[str] = []

    def plan(self, text: str) -> LocalPlan | None:
        self.seen.append(text)
        if self.error is not None:
            raise self.error
        return self.plan_result


class FakeVision:
    def __init__(self, answer: str = "На экране открыт браузер") -> None:
        self.enabled = True
        self.answer = answer
        self.error: Exception | None = None
        self.seen: list[str] = []
        self.contexts: list[str] = []
        self.snapshots = 0
        self.captures = 0
        self.describe_calls = 0
        self.frame_answers = 0
        self.remembered = 0
        self.frames: list[Any] = []
        self.frame_batches: list[list[Any]] = []
        self.pending = False

    def match(self, text: str) -> tuple[bool, str]:
        return match_request(text, DEFAULT_TRIGGERS)

    def wants_snapshot(self, text: str) -> bool:
        return text == "посмотри на экран"

    def available(self) -> bool:
        return True

    def capture(self) -> bytes:
        self.captures += 1
        if self.error is not None:
            raise self.error
        self.pending = True
        return b"jpeg"

    def pending_image(self) -> bytes | None:
        return b"old-jpeg" if self.pending else None

    def snapshot(self) -> bytes:
        self.snapshots += 1
        return self.capture()

    def remember_frame(self, frame: Any) -> bytes:
        self.remembered += 1
        self.frames.append(frame)
        if self.error is not None:
            raise self.error
        self.pending = True
        return b"jpeg"

    def has_pending(self) -> bool:
        return self.pending

    def match_followup(self, text: str) -> bool:
        return "кнопка" in text or "написано" in text

    def describe(self, request: str = "", *, context: str = "") -> str:
        self.describe_calls += 1
        self.seen.append(request)
        self.contexts.append(context)
        if self.error is not None:
            raise self.error
        self.pending = True
        return self.answer

    def describe_frame(self, frame: Any, request: str = "", *, context: str = "") -> str:
        self.frame_answers += 1
        self.frames.append(frame)
        self.seen.append(request)
        self.contexts.append(context)
        if self.error is not None:
            raise self.error
        self.pending = True
        return self.answer

    def describe_frames(
        self,
        frames: list[Any] | tuple[Any, ...],
        request: str = "",
        *,
        context: str = "",
    ) -> str:
        self.frame_batches.append(list(frames))
        self.seen.append(request)
        self.contexts.append(context)
        if self.error is not None:
            raise self.error
        self.pending = True
        return self.answer

    def describe_pending(self, request: str, *, context: str = "") -> str:
        self.seen.append(request)
        self.contexts.append(context)
        if self.error is not None:
            raise self.error
        return self.answer


class FakeImageSearch:
    def __init__(
        self,
        result: ImageSearchResult | None = None,
        available: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.enabled = True
        self.is_available = available
        self.result = result or ImageSearchResult(
            images=(
                ImageMatch(
                    page_title="Илон Маск",
                    host="ru.wikipedia.org",
                    passage="Предприниматель и изобретатель.",
                ),
            )
        )
        self.error = error
        self.images: list[bytes] = []
        self.max_results = 5
        self.max_passage_chars = 300
        self.timeout = 1.0

    def available(self) -> bool:
        return self.is_available

    def match(self, text: str) -> tuple[bool, str]:
        return match_image_request(text, IMAGE_TRIGGERS)

    def search(self, image: bytes, **kwargs: Any) -> ImageSearchResult:
        self.images.append(image)
        if self.error is not None:
            raise self.error
        return self.result


class FakeTTS:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.phrases: list[str] = []
        self.said: list[str] = []
        self.speaking = False
        self.current_text = ""
        self.stops = 0

    def phrase(self, category: str, phrase_id: str | None = None) -> None:
        self.phrases.append(category)

    def say(self, text: str) -> None:
        self.said.append(text)

    def stop(self) -> None:
        self.stops += 1
        self.speaking = False

    def shutdown(self) -> None:
        pass


class FakeTray:
    def __init__(self, status: dict[str, Any]) -> None:
        self._status = status
        self.states: list[str] = []

    def status(self) -> dict[str, Any]:
        return self._status

    def set_state(self, state: str) -> None:
        self.states.append(state)


class FakeVisionWatch:
    def __init__(self) -> None:
        self.active = False
        self.paused = False
        self.start_calls = 0
        self.stop_calls = 0
        self.started_at: float | None = None
        self.changes = 0
        self.window_title: str | None = None
        self.error: str | None = None
        self.interval_sec = 2.0
        self.auto_off_sec = 600.0
        self.last_frame: Any = None
        self.timeline = EventTimeline()

    def start(self) -> bool:
        self.start_calls += 1
        if self.active:
            return False
        self.timeline.reset()
        self.active = True
        self.started_at = time.monotonic()
        return True

    def stop(self) -> bool:
        self.stop_calls += 1
        if not self.active:
            return False
        self.active = False
        self.timeline.reset()
        return True

    def pause(self) -> bool:
        if not self.active:
            return False
        self.paused = True
        self.timeline.reset()
        return True

    def resume(self) -> bool:
        if not self.active:
            return False
        self.paused = False
        return True

    def status(self) -> WatchStatus:
        return WatchStatus(
            active=self.active,
            paused=self.paused,
            started_at=self.started_at,
            last_poll_at=self.started_at,
            last_change_at=self.started_at,
            polls=2 if self.active else 0,
            changes=self.changes,
            window_title=self.window_title,
            error=self.error,
            interval_sec=self.interval_sec,
            auto_off_sec=self.auto_off_sec,
        )


class FakeVisionComments:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.active = False
        self.paused = False
        self.start_calls = 0
        self.stop_calls = 0
        self.pause_calls = 0
        self.resume_calls = 0
        self.enqueued: list[tuple[Any, Any]] = []
        self.use_frame = True
        self.comments = 0
        self.attempts = 0
        self.described = "Комментарии включены."
        self.min_interval_sec = 90.0

    def start(self) -> bool:
        self.start_calls += 1
        if not self.available or self.active:
            return False
        self.active = True
        self.paused = False
        return True

    def stop(self) -> bool:
        self.stop_calls += 1
        if not self.active:
            return False
        self.active = False
        self.paused = False
        return True

    def pause(self) -> bool:
        self.pause_calls += 1
        if not self.active or self.paused:
            return False
        self.paused = True
        return True

    def resume(self) -> bool:
        self.resume_calls += 1
        if not self.active or not self.paused:
            return False
        self.paused = False
        return True

    def enqueue(self, event: Any, frame: Any) -> bool:
        self.enqueued.append((event, frame))
        return self.active and not self.paused

    def status(self) -> CommentStatus:
        return CommentStatus(
            available=self.available,
            active=self.active,
            paused=self.paused,
            use_frame=self.use_frame,
            min_interval_sec=self.min_interval_sec,
            max_per_hour=10,
            attempts=self.attempts,
            comments=self.comments,
            last_attempt_at=None,
            last_comment_at=None,
            next_allowed_at=None,
        )

    def describe(self, *, now: float | None = None) -> str:
        return self.described


class FakeQA:
    def __init__(self, answer: QAAnswer | None = None, kind: str = "general") -> None:
        self.result = answer or QAAnswer(
            True, "вопрос", "Короткий ответ", "Полный ответ", "general"
        )
        self.kind = kind
        self.routed: list[str] = []
        self.answered: list[str] = []
        self.shown: list[QAAnswer] = []

    def route(self, text: str, *, in_dialog: bool = False) -> Route:
        self.routed.append(text)
        return Route(self.kind)

    def answer(self, text: str, *, route: Route | None = None) -> QAAnswer:
        self.answered.append(text)
        return self.result

    def show(self, answer: QAAnswer) -> bool:
        self.shown.append(answer)
        return True


class FakeIntent:
    def __init__(
        self,
        mode: str = MODE_SHADOW,
        decision: IntentDecision | None = None,
        *,
        visual: IntentDecision | None = None,
    ) -> None:
        self.mode = mode
        self.decision = decision or IntentDecision(
            DIALOG,
            confidence=0.9,
            raw=DIALOG,
            source=SOURCE_MODEL,
        )
        self.visual = visual or IntentDecision(
            UNKNOWN,
            source=SOURCE_FALLBACK,
        )
        self.state = DialogState()
        self.observed: list[tuple[str, bool]] = []
        self.classified: list[tuple[str, bool]] = []
        self.visual_classified: list[tuple[str, bool]] = []

    def observe(self, text: str, *, in_dialog: bool) -> None:
        self.observed.append((text, in_dialog))

    def classify(self, text: str, *, in_dialog: bool) -> IntentDecision:
        self.classified.append((text, in_dialog))
        return self.decision

    def classify_visual(self, text: str, *, in_dialog: bool) -> IntentDecision:
        self.visual_classified.append((text, in_dialog))
        return self.visual

    def blocks_lookup(self, decision: IntentDecision) -> bool:
        return decision.source == SOURCE_MODEL and decision.intent in {
            ANSWER,
            DIALOG,
            "clarify",
            UNKNOWN,
        }

    def blocks_search(self, decision: IntentDecision) -> bool:
        return self.blocks_lookup(decision)


class FakeAddressee:
    def __init__(
        self,
        kind: str = ADDRESSED,
        mode: str = ADDRESSEE_ACTIVE,
        *,
        weak: bool = False,
    ) -> None:
        self.kind = kind
        self.mode = mode
        self.enabled = True
        self.weak = weak
        self.seen: list[tuple[str, bool, str]] = []

    def classify(
        self,
        text: str,
        *,
        first_turn: bool = False,
        last_question: str = "",
    ) -> AddresseeDecision:
        self.seen.append((text, first_turn, last_question))
        return AddresseeDecision(
            self.kind,
            confidence=0.9,
            raw=self.kind,
            source=SOURCE_MODEL,
            weak=self.weak,
        )

    def observe(
        self,
        text: str,
        *,
        first_turn: bool = False,
        last_question: str = "",
    ) -> None:
        self.classify(text, first_turn=first_turn, last_question=last_question)


class TrayStatusTest(unittest.TestCase):
    @staticmethod
    def make(tray: Any) -> Assistant:
        assistant = Assistant.__new__(Assistant)
        assistant.tray = tray
        assistant.cfg = {"wake_word": "хей луна", "hotkey": "ctrl+alt+space"}
        assistant.stt = SimpleNamespace(loaded=False)
        assistant.wake = None
        assistant._state_lock = threading.Lock()
        assistant._state = "starting"
        assistant._paused = False
        return assistant

    def test_status_exposes_tray_state(self) -> None:
        self.assertIsNone(self.make(None).status()["tray"])

        tray_status = {
            "available": True,
            "running": True,
            "error": None,
            "restarts": 2,
        }
        status = self.make(FakeTray(tray_status)).status()
        self.assertEqual(set(status["tray"]), {"available", "running", "error", "restarts"})
        self.assertEqual(status["tray"], tray_status)

    def test_status_without_observation_is_none(self) -> None:
        self.assertIsNone(self.make(None).status()["observation"])

    def test_status_exposes_observation(self) -> None:
        assistant = self.make(None)
        assistant.vision_watch = FakeVisionWatch()
        assistant.vision_watch.active = True
        assistant.vision_watch.changes = 2

        observation = assistant.status()["observation"]

        self.assertEqual(
            observation,
            {"active": True, "paused": False, "changes": 2},
        )

    def test_observation_makes_tray_watching(self) -> None:
        assistant = self.make(FakeTray({}))
        assistant.vision_watch = FakeVisionWatch()
        assistant.vision_watch.active = True

        assistant.set_state("idle")
        self.assertEqual(assistant.state, "idle")
        self.assertEqual(assistant.tray.states, ["watching"])

        assistant.vision_watch.active = False
        assistant.set_state("idle")
        self.assertEqual(assistant.tray.states, ["watching", "idle"])

    def test_comments_make_tray_commenting(self) -> None:
        assistant = self.make(FakeTray({}))
        assistant.vision_watch = FakeVisionWatch()
        assistant.vision_watch.active = True
        assistant.vision_comments = FakeVisionComments()
        assistant.vision_comments.active = True

        assistant.set_state("idle")
        self.assertEqual(assistant.tray.states, ["commenting"])

        assistant.vision_comments.active = False
        assistant.set_state("idle")
        self.assertEqual(assistant.tray.states, ["commenting", "watching"])

    def test_status_exposes_comments(self) -> None:
        assistant = self.make(None)
        assistant.vision_comments = FakeVisionComments()
        assistant.vision_comments.active = True
        assistant.vision_comments.comments = 2
        assistant.vision_comments.attempts = 3

        comments = assistant.status()["comments"]

        assert comments is not None
        self.assertTrue(comments["active"])
        self.assertEqual(comments["comments"], 2)
        self.assertEqual(comments["attempts"], 3)


class PauseHotkeyTest(unittest.TestCase):
    @staticmethod
    def make(**cfg: Any) -> Assistant:
        assistant = Assistant.__new__(Assistant)
        assistant.cfg = {
            "hotkey": "ctrl+alt+space",
            "pause_hotkey": "ctrl+alt+m",
            **cfg,
        }
        assistant._paused = False
        assistant.mic = None
        assistant.notifications = []
        assistant.states = []
        assistant.tts = FakeTTS()
        assistant.notify = lambda message, title="Ассистент": assistant.notifications.append(
            message
        )
        assistant.set_state = assistant.states.append
        return assistant

    def test_bindings_include_pause_hotkey(self) -> None:
        assistant = self.make()
        bindings = assistant._hotkey_bindings()
        self.assertEqual(
            set(bindings),
            {"<ctrl>+<alt>+<space>", "<ctrl>+<alt>+<77>"},
        )
        self.assertEqual(bindings["<ctrl>+<alt>+<77>"], assistant.toggle_pause)

    def test_empty_pause_hotkey_disables_binding(self) -> None:
        assistant = self.make(pause_hotkey="")
        self.assertEqual(
            set(assistant._hotkey_bindings()),
            {"<ctrl>+<alt>+<space>"},
        )

    def test_duplicate_pause_hotkey_keeps_listen(self) -> None:
        assistant = self.make(pause_hotkey="ctrl+alt+space")
        bindings = assistant._hotkey_bindings()
        self.assertEqual(set(bindings), {"<ctrl>+<alt>+<space>"})
        self.assertEqual(bindings["<ctrl>+<alt>+<space>"], assistant.trigger_listen)

    def test_toggle_pause_pauses_observation(self) -> None:
        assistant = self.make()
        assistant.vision_watch = FakeVisionWatch()
        assistant.vision_watch.active = True

        assistant.toggle_pause()
        self.assertTrue(assistant.vision_watch.paused)

        assistant.toggle_pause()
        self.assertFalse(assistant.vision_watch.paused)
        self.assertTrue(assistant.vision_watch.active)

    def test_toggle_pause_switches_state_and_speaks(self) -> None:
        assistant = self.make()
        assistant.toggle_pause()
        self.assertTrue(assistant._paused)
        self.assertEqual(assistant.states, ["paused"])
        self.assertEqual(assistant.notifications, ["Пауза"])
        self.assertEqual(assistant.tts.phrases, ["mic"])

        assistant.toggle_pause()
        self.assertFalse(assistant._paused)
        self.assertEqual(assistant.states, ["paused", "idle"])
        self.assertEqual(assistant.notifications, ["Пауза", "Снова слушаю"])


class HotkeyLayoutTest(unittest.TestCase):
    def test_russian_letter_uses_virtual_key(self) -> None:
        from pynput import keyboard

        key = keyboard.KeyCode(char="ь", vk=77)
        self.assertEqual(_hotkey_key(key), keyboard.KeyCode.from_vk(77))

    def test_modifiers_are_canonical(self) -> None:
        from pynput import keyboard

        self.assertEqual(_hotkey_key(keyboard.Key.ctrl_l), keyboard.Key.ctrl)
        self.assertEqual(_hotkey_key(keyboard.Key.alt_r), keyboard.Key.alt)

    def test_pause_hotkey_fires_in_russian_layout(self) -> None:
        from pynput import keyboard

        assistant = PauseHotkeyTest.make()
        listener = _make_hotkey_listener(assistant._hotkey_bindings())
        listener._on_press(keyboard.Key.ctrl_l, False)
        listener._on_press(keyboard.Key.alt_l, False)
        listener._on_press(keyboard.KeyCode(char="ь", vk=77), False)
        self.assertTrue(assistant._paused)


class CommandRoutingTest(unittest.TestCase):
    def make(self, result: CommandResult | None = None) -> Assistant:
        assistant = Assistant.__new__(Assistant)
        assistant.router = FakeRouter(result)
        assistant.planner = FakeLocalPlanner()
        assistant.qa = None
        assistant.vision = FakeVision()
        assistant.image_search = FakeImageSearch()
        assistant.llm = FakeLLM()
        assistant.tts = FakeTTS()
        assistant.notifications = []
        assistant.states = []
        assistant.notify = lambda message, title="Ассистент": assistant.notifications.append(
            message
        )
        assistant.set_state = lambda state: assistant.states.append(state)
        return assistant

    def test_unknown_command_falls_back(self) -> None:
        assistant = self.make()
        with patch("app.main.beep") as beep:
            message = assistant.run_text("сделай что-нибудь странное")
        self.assertEqual(message, "Не понял команду")
        self.assertEqual(assistant.router.seen, ["сделай что нибудь странное"])
        beep.assert_called_once_with(320, 120)

    def test_router_command_with_word_agent_is_not_special(self) -> None:
        assistant = self.make(CommandResult(True, True, "Открываю YouTube"))
        message = assistant.run_text("включи фильм агент 007")
        self.assertEqual(message, "Открываю YouTube")
        self.assertEqual(assistant.router.seen, ["включи фильм агент 007"])

    def test_vision_command_bypasses_router(self) -> None:
        assistant = self.make()
        assistant.tts = FakeTTS(enabled=True)
        message = assistant.run_text("посмотри на экран, что за ошибка")
        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(assistant.vision.seen, ["что за ошибка"])
        self.assertEqual(assistant.router.seen, [])
        self.assertEqual(assistant.planner.seen, [])
        self.assertEqual(assistant.tts.said, ["На экране открыт браузер"])

    def test_vision_error_is_reported(self) -> None:
        assistant = self.make()
        assistant.vision.error = RuntimeError("нет доступа")
        with patch("app.main.beep") as beep:
            message = assistant.run_text("посмотри на экран")
        self.assertEqual(message, "Не удалось посмотреть на экран")
        beep.assert_called_once_with(320, 120)
        self.assertEqual(assistant.tts.phrases, ["error"])

    def test_vision_disabled_is_reported(self) -> None:
        assistant = self.make()
        assistant.vision.enabled = False
        with patch("app.main.beep") as beep:
            message = assistant.run_text("что на экране")
        self.assertEqual(message, "Зрение выключено")
        beep.assert_called_once_with(320, 120)

    def test_vision_snapshot_then_followup(self) -> None:
        assistant = self.make()
        assistant.tts = FakeTTS(enabled=True)
        message = assistant.run_text("посмотри на экран")
        self.assertEqual(message, "Снимок экрана готов. Спрашивай, что на нём.")
        self.assertEqual(assistant.vision.snapshots, 1)
        self.assertEqual(assistant.vision.seen, [])
        self.assertEqual(assistant.router.seen, [])
        self.assertEqual(assistant.planner.seen, [])
        self.assertEqual(assistant.tts.phrases, ["ok"])

        message = assistant.run_text("где здесь кнопка")
        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(assistant.vision.seen, ["где здесь кнопка"])
        self.assertEqual(assistant.router.seen, [])
        self.assertEqual(assistant.planner.seen, [])
        self.assertEqual(assistant.tts.said, ["На экране открыт браузер"])

    def test_vision_followup_without_snapshot_is_not_vision(self) -> None:
        assistant = self.make()
        with patch("app.main.beep"):
            message = assistant.run_text("где здесь кнопка")
        self.assertEqual(message, "Не понял команду")
        self.assertEqual(assistant.vision.seen, [])
        self.assertEqual(assistant.router.seen, ["где здесь кнопка"])

    def test_image_search_bypasses_router_and_vision(self) -> None:
        assistant = self.make()
        assistant.tts = FakeTTS(enabled=True)
        assistant.llm = FakeLLM("Это Илон Маск.")
        message = assistant.run_text("поищи по фото кто это")
        self.assertEqual(message, "Это Илон Маск.")
        self.assertEqual(assistant.router.seen, [])
        self.assertEqual(assistant.planner.seen, [])
        self.assertEqual(assistant.vision.seen, [])
        self.assertEqual(assistant.image_search.images, [b"jpeg"])
        self.assertEqual(assistant.llm.calls[0][0], "кто это")
        self.assertIn("Илон Маск", assistant.llm.calls[0][1])
        self.assertEqual(assistant.llm.kwargs[0].get("style"), "image_search")
        self.assertEqual(assistant.tts.said, ["Это Илон Маск."])

    def test_image_search_ignores_pending_frame(self) -> None:
        assistant = self.make()
        assistant.vision.pending = True
        assistant.run_text("поищи по фото")
        self.assertEqual(assistant.vision.captures, 1)
        self.assertEqual(assistant.image_search.images, [b"jpeg"])

    def test_image_search_captures_without_pending(self) -> None:
        assistant = self.make()
        assistant.vision.pending = False
        assistant.run_text("поищи по фото")
        self.assertEqual(assistant.vision.captures, 1)

    def test_image_search_empty_results(self) -> None:
        assistant = self.make()
        assistant.image_search.result = ImageSearchResult()
        message = assistant.run_text("поищи по фото")
        self.assertEqual(message, "Ничего похожего не нашёл")
        self.assertEqual(assistant.llm.calls, [])

    def test_image_search_error_is_reported(self) -> None:
        assistant = self.make()
        assistant.image_search.error = ImageSearchError("Яндекс не принял ключ поиска")
        with patch("app.main.beep") as beep:
            message = assistant.run_text("поищи по фото")
        self.assertEqual(message, "Яндекс не принял ключ поиска")
        beep.assert_called_once_with(320, 120)
        self.assertEqual(assistant.tts.phrases, ["error"])

    def test_image_search_not_configured(self) -> None:
        assistant = self.make()
        assistant.image_search.is_available = False
        with patch("app.main.beep"):
            message = assistant.run_text("поищи по фото")
        self.assertIn("не настроен", message)
        self.assertEqual(assistant.image_search.images, [])

    def test_image_search_llm_failure_uses_fallback(self) -> None:
        assistant = self.make()
        assistant.llm = FakeLLM(error=RuntimeError("нет сети"))
        message = assistant.run_text("поищи по фото")
        self.assertIn("Нашёл похожие изображения", message)

    def test_kto_eto_does_not_call_image_search(self) -> None:
        assistant = self.make()
        with patch("app.main.beep"):
            assistant.run_text("кто это")
        self.assertEqual(assistant.image_search.images, [])

    def test_canonicalizes_text_before_routing(self) -> None:
        assistant = self.make(CommandResult(True, True, "Открываю Google"))
        message = assistant.run_text("открой гугель")
        self.assertEqual(message, "Открываю Google")
        self.assertEqual(assistant.router.seen, ["открой гугл"])

    def test_free_form_command_uses_local_planner(self) -> None:
        assistant = self.make()
        assistant.planner = FakeLocalPlanner(LocalPlan("youtube", {"query": "lo-fi", "index": 1}))
        assistant.router.plan_result = CommandResult(True, True, "Включаю на YouTube: lo-fi")
        message = assistant.run_text("вруби лофай на ютубе")
        self.assertEqual(message, "Включаю на YouTube: lo-fi")
        self.assertEqual(assistant.planner.seen, ["вруби лоуфай на ютубе"])
        self.assertEqual(
            assistant.router.plans,
            [("youtube", {"query": "lo-fi", "index": 1})],
        )

    def test_local_planner_none_falls_back(self) -> None:
        assistant = self.make()
        with patch("app.main.beep") as beep:
            message = assistant.run_text("сделай что-нибудь странное")
        self.assertEqual(message, "Не понял команду")
        self.assertEqual(assistant.planner.seen, ["сделай что нибудь странное"])
        beep.assert_called_once_with(320, 120)

    def test_router_command_skips_local_planner(self) -> None:
        assistant = self.make(CommandResult(True, True, "Открываю YouTube"))
        assistant.run_text("открой ютуб")
        self.assertEqual(assistant.planner.seen, [])

    def test_local_planner_error_falls_back(self) -> None:
        assistant = self.make()
        assistant.planner = FakeLocalPlanner(error=LocalPlannerError("нет сервера"))
        with patch("app.main.beep") as beep:
            message = assistant.run_text("вруби музыку")
        self.assertEqual(message, "Не понял команду")
        beep.assert_called_once_with(320, 120)

    def test_question_uses_qa_before_local_planner(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA()
        message = assistant.run_text("что такое фотосинтез")
        self.assertEqual(message, "Короткий ответ")
        self.assertEqual(assistant.planner.seen, [])
        self.assertEqual(assistant.qa.routed, ["что такое фотосинтез"])
        self.assertEqual(assistant.qa.answered, ["что такое фотосинтез"])
        self.assertEqual(len(assistant.qa.shown), 1)

    def test_qa_error_beeps_without_tab(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA(QAAnswer(False, "вопрос", "Нужен ключ", "Нужен ключ", "error"))
        with patch("app.main.beep") as beep:
            message = assistant.run_text("какие новости")
        self.assertEqual(message, "Нужен ключ")
        beep.assert_called_once_with(320, 120)
        self.assertEqual(assistant.qa.shown, [])
        self.assertEqual(assistant.tts.phrases, ["error"])
        self.assertEqual(assistant.tts.said, [])

    def test_no_model_reason_is_spoken(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA(
            QAAnswer(
                False,
                "вопрос",
                "Модель недоступна.",
                "Модель недоступна.",
                "no_model",
            )
        )
        with patch("app.main.beep") as beep:
            message = assistant.run_text("что такое фотосинтез")
        self.assertEqual(message, "Модель недоступна.")
        beep.assert_called_once_with(320, 120)
        self.assertEqual(assistant.tts.said, ["Модель недоступна."])
        self.assertEqual(assistant.tts.phrases, [])
        self.assertEqual(assistant.qa.shown, [])

    def test_router_command_does_not_use_qa(self) -> None:
        assistant = self.make(CommandResult(True, True, "Открываю YouTube"))
        assistant.qa = FakeQA()
        message = assistant.run_text("открой ютуб")
        self.assertEqual(message, "Открываю YouTube")
        self.assertEqual(assistant.qa.answered, [])
        self.assertEqual(assistant.qa.shown, [])

    def test_datetime_answer_skips_ack_phrase(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA(
            QAAnswer(
                True,
                "который час",
                "Сейчас 20 часов 47 минут.",
                "Сейчас 20 часов 47 минут.",
                "time",
            ),
            kind="datetime",
        )
        assistant.tts = FakeTTS(enabled=True)
        message = assistant.run_text("который час")
        self.assertEqual(message, "Сейчас 20 часов 47 минут.")
        self.assertEqual(assistant.tts.phrases, [])
        self.assertEqual(assistant.tts.said, ["Сейчас 20 часов 47 минут."])


class FakeLLM:
    def __init__(
        self,
        text: str = "Ответ модели.",
        available: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.is_available = available
        self.error = error
        self.calls: list[tuple[str, str]] = []
        self.kwargs: list[dict[str, Any]] = []

    def available(self) -> bool:
        return self.is_available

    def answer(self, question: str, context: str = "", **kwargs: Any) -> str:
        self.calls.append((question, context))
        self.kwargs.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return self.text


class FakeSearch:
    def __init__(
        self,
        results: list[SearchResult] | None = None,
        available: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.results = results or []
        self.is_available = available
        self.error = error
        self.calls: list[tuple[str, bool]] = []
        self.article_calls: list[str] = []
        self.extract_calls: list[str] = []

    def available(self) -> bool:
        return self.is_available

    def search(self, query: str, *, news: bool = False) -> list[SearchResult]:
        self.calls.append((query, news))
        if self.error is not None:
            raise self.error
        return self.results

    def search_article(self, query: str, *, limit: int = 3) -> list[SearchResult]:
        self.article_calls.append(query)
        if self.error is not None:
            raise self.error
        return self.results[:limit]

    def extract(self, url: str) -> str:
        self.extract_calls.append(url)
        if self.error is not None:
            raise self.error
        return ""


class FakeWeather:
    def __init__(
        self,
        report: WeatherReport | None = None,
        error: Exception | None = None,
    ) -> None:
        self.report = report or make_e2e_report()
        self.error = error
        self.cities: list[str | None] = []

    def forecast(self, city: str | None = None) -> WeatherReport:
        self.cities.append(city)
        if self.error is not None:
            raise self.error
        return self.report


class FakeBrowser:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, action: str, **args: Any) -> dict[str, Any]:
        self.calls.append((action, args))
        return {"ok": True}


def make_e2e_report(place: str = "Москва") -> WeatherReport:
    return WeatherReport(
        place=GeoPlace(place, "Россия", 55.75, 37.61),
        temperature=2.0,
        feels_like=-1.0,
        code=3,
        wind=0.8,
        humidity=89,
        daily=(
            DailyForecast("2026-11-13", 0.0, 2.0, 77),
            DailyForecast("2026-11-14", 1.0, 4.0, 63),
            DailyForecast("2026-11-15", 2.0, 4.0, 61),
        ),
    )


class CommandPipelineTest(unittest.TestCase):
    """Assistant._process → QAService → внешние сервисы, все внешние вызовы на фейках."""

    def make(
        self,
        *,
        llm: FakeLLM | None = None,
        search: FakeSearch | None = None,
        weather: FakeWeather | None = None,
        browser: FakeBrowser | None = None,
        router: FakeRouter | None = None,
    ) -> Assistant:
        assistant = Assistant.__new__(Assistant)
        assistant.router = router or FakeRouter()
        assistant.planner = FakeLocalPlanner()
        assistant.tts = FakeTTS(enabled=True)
        assistant.notifications = []
        assistant.states = []

        def notify(message: str, title: str = "Ассистент") -> None:
            assistant.notifications.append(message)

        assistant.notify = notify
        assistant.set_state = assistant.states.append

        qa = QAService(
            {
                "qa": {
                    "enabled": True,
                    "weather": {"city": "Москва"},
                    "search": {"api_key": "test-key"},
                }
            },
            llm=llm,
            browser=browser or FakeBrowser(),
        )
        qa.search = search or FakeSearch()
        qa.weather = weather or FakeWeather()
        assistant.qa = qa
        return assistant

    def test_weather_answer_renders_tab_and_speaks(self) -> None:
        weather = FakeWeather(make_e2e_report())
        browser = FakeBrowser()
        assistant = self.make(weather=weather, browser=browser)

        with patch("app.main.beep") as beep:
            message = assistant.run_text("какая погода в Москве")

        self.assertIn("Москва", message)
        self.assertEqual(weather.cities, ["москве"])
        self.assertEqual(assistant.tts.phrases, ["weather"])
        self.assertEqual(assistant.tts.said, [message])
        self.assertEqual(assistant.states, ["thinking", "idle"])
        beep.assert_not_called()
        self.assertEqual(len(browser.calls), 1)
        action, args = browser.calls[0]
        self.assertEqual(action, "render")
        self.assertIn("Завтра", args["html"])

    def test_weather_followup_reuses_city(self) -> None:
        weather = FakeWeather(make_e2e_report())
        assistant = self.make(weather=weather)

        first = assistant.run_text("какая погода в Москве")
        second = assistant.run_text("а завтра?")

        self.assertIn("Москва", first)
        self.assertIn("завтра", second.lower())
        self.assertEqual(weather.cities, ["москве", "Москва"])
        self.assertEqual(assistant.tts.phrases, ["weather", "weather"])
        self.assertEqual(len(assistant.tts.said), 2)

    def test_general_answer_uses_llm_and_opens_tab(self) -> None:
        llm = FakeLLM("Фотосинтез — это процесс.")
        search = FakeSearch(available=False)
        browser = FakeBrowser()
        assistant = self.make(llm=llm, search=search, browser=browser)

        message = assistant.run_text("что такое фотосинтез")

        self.assertEqual(message, "Фотосинтез — это процесс.")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(search.calls, [])
        self.assertEqual(assistant.tts.phrases, ["wait"])
        self.assertEqual(assistant.tts.said, [message])
        self.assertEqual(len(browser.calls), 1)

    def test_general_followup_passes_history(self) -> None:
        llm = FakeLLM("Фотосинтез — это процесс.")
        assistant = self.make(llm=llm, search=FakeSearch(available=False))

        assistant.run_text("что такое фотосинтез")
        assistant.run_text("а почему это важно?")

        self.assertEqual(len(llm.calls), 2)
        self.assertIn("что такое фотосинтез", llm.kwargs[1]["history"])
        self.assertEqual(assistant.tts.phrases, ["wait", "wait"])

    def test_news_answer_opens_tab_and_remembers_items(self) -> None:
        search = FakeSearch(
            [
                SearchResult(
                    "Выборы",
                    "https://example.com/news",
                    "Свежая новость про выборы и курсы.",
                )
            ]
        )
        llm = FakeLLM("Главная новость про выборы [1].")
        browser = FakeBrowser()
        assistant = self.make(llm=llm, search=search, browser=browser)

        message = assistant.run_text("какие новости")

        self.assertTrue(message)
        self.assertTrue(search.calls[0][1])
        self.assertEqual(assistant.tts.phrases, ["news"])
        self.assertEqual(len(browser.calls), 1)
        self.assertTrue(assistant.qa.context.news())

    def test_empty_search_is_not_an_error(self) -> None:
        browser = FakeBrowser()
        assistant = self.make(search=FakeSearch(results=[]), browser=browser)

        with patch("app.main.beep") as beep:
            message = assistant.run_text("курс доллара")

        self.assertEqual(message, "Ничего не нашлось.")
        self.assertEqual(assistant.tts.phrases, ["wait"])
        self.assertEqual(assistant.tts.said, [message])
        self.assertEqual(browser.calls, [])
        beep.assert_not_called()

    def test_search_failure_speaks_generic_error(self) -> None:
        browser = FakeBrowser()
        assistant = self.make(search=FakeSearch(error=SearchError("boom")), browser=browser)

        with patch("app.main.beep") as beep:
            message = assistant.run_text("курс доллара")

        self.assertEqual(message, "Сервис не отвечает.")
        self.assertEqual(assistant.tts.phrases, ["wait", "error"])
        self.assertEqual(assistant.tts.said, [])
        self.assertEqual(browser.calls, [])
        beep.assert_called_once_with(320, 120)

    def test_no_model_reason_is_spoken(self) -> None:
        browser = FakeBrowser()
        assistant = self.make(
            llm=FakeLLM(available=False),
            search=FakeSearch(available=False),
            browser=browser,
        )

        with patch("app.main.beep") as beep:
            message = assistant.run_text("что такое фотосинтез")

        self.assertEqual(message, "Модель недоступна.")
        self.assertEqual(assistant.tts.phrases, ["wait"])
        self.assertEqual(assistant.tts.said, [message])
        self.assertEqual(browser.calls, [])
        beep.assert_called_once_with(320, 120)

    def test_missing_search_key_is_an_error(self) -> None:
        browser = FakeBrowser()
        assistant = self.make(search=FakeSearch(available=False), browser=browser)

        with patch("app.main.beep") as beep:
            message = assistant.run_text("какие новости")

        self.assertEqual(message, "Нужен ключ Tavily")
        self.assertEqual(assistant.tts.phrases, ["news", "error"])
        self.assertEqual(assistant.tts.said, [])
        self.assertEqual(browser.calls, [])
        beep.assert_called_once_with(320, 120)

    def test_weather_failure_speaks_generic_error(self) -> None:
        browser = FakeBrowser()
        assistant = self.make(weather=FakeWeather(error=WeatherError("down")), browser=browser)

        with patch("app.main.beep") as beep:
            message = assistant.run_text("какая погода")

        self.assertEqual(message, "Не смог узнать погоду.")
        self.assertEqual(assistant.tts.phrases, ["weather", "error"])
        self.assertEqual(assistant.tts.said, [])
        self.assertEqual(browser.calls, [])
        beep.assert_called_once_with(320, 120)

    def test_datetime_answer_is_local_and_silent_ack(self) -> None:
        llm = FakeLLM()
        search = FakeSearch()
        weather = FakeWeather()
        browser = FakeBrowser()
        assistant = self.make(llm=llm, search=search, weather=weather, browser=browser)

        message = assistant.run_text("какое сегодня число")

        self.assertTrue(message.startswith("Сегодня "))
        self.assertEqual(assistant.tts.phrases, [])
        self.assertEqual(assistant.tts.said, [message])
        self.assertEqual(llm.calls, [])
        self.assertEqual(search.calls, [])
        self.assertEqual(weather.cities, [])
        self.assertEqual(browser.calls, [])
        self.assertIsNone(assistant.qa.context.turn())

    def test_browser_command_bypasses_qa(self) -> None:
        router = FakeRouter(CommandResult(True, True, "Открываю YouTube"))
        llm = FakeLLM()
        search = FakeSearch()
        weather = FakeWeather()
        browser = FakeBrowser()
        assistant = self.make(
            llm=llm,
            search=search,
            weather=weather,
            browser=browser,
            router=router,
        )

        with patch("app.main.beep") as beep:
            message = assistant.run_text("открой ютуб")

        self.assertEqual(message, "Открываю YouTube")
        self.assertEqual(router.seen, ["открой ютуб"])
        self.assertEqual(assistant.tts.phrases, ["ack"])
        self.assertEqual(llm.calls, [])
        self.assertEqual(search.calls, [])
        self.assertEqual(weather.cities, [])
        self.assertEqual(browser.calls, [])
        beep.assert_not_called()

    def test_unexpected_error_is_caught(self) -> None:
        assistant = self.make(llm=FakeLLM(error=RuntimeError("boom")))

        with patch("app.main.beep") as beep, patch("app.main.log.exception") as log_error:
            message = assistant.run_text("что такое фотосинтез")

        self.assertEqual(message, "Ошибка: boom")
        self.assertEqual(assistant.tts.phrases, ["wait", "error"])
        self.assertEqual(assistant.tts.said, [])
        self.assertEqual(assistant.states, ["thinking", "idle"])
        beep.assert_called_once_with(320, 120)
        log_error.assert_called_once()

    def test_unknown_command(self) -> None:
        assistant = self.make()

        with patch("app.main.beep") as beep:
            message = assistant.run_text("сделай что-нибудь странное")

        self.assertEqual(message, "Не понял команду")
        self.assertEqual(assistant.tts.phrases, ["ack", "unknown"])
        beep.assert_called_once_with(320, 120)

    def test_shadow_observes_dialog_search_and_keeps_it(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA(kind="search")
        assistant.intent = FakeIntent(MODE_SHADOW)
        assistant._dialog_active = True

        message = assistant.run_text("курс доллара")

        self.assertEqual(message, "Короткий ответ")
        self.assertEqual(assistant.qa.answered, ["курс доллара"])
        self.assertEqual(assistant.intent.observed, [("курс доллара", True)])
        self.assertEqual(assistant.intent.classified, [])

    def test_active_blocks_false_search_in_dialog(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA(kind="search")
        assistant.intent = FakeIntent(
            MODE_ACTIVE,
            IntentDecision(DIALOG, confidence=0.9, raw=DIALOG, source=SOURCE_MODEL),
        )
        assistant._dialog_active = True

        message = assistant.run_text("курс доллара")

        self.assertEqual(message, "Не понял команду")
        self.assertEqual(assistant.qa.answered, [])
        self.assertEqual(assistant.intent.classified, [("курс доллара", True)])
        self.assertEqual(assistant.intent.observed, [])

    def test_active_blocks_false_news_in_dialog(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA(kind="news")
        assistant.intent = FakeIntent(
            MODE_ACTIVE,
            IntentDecision(DIALOG, confidence=0.9, raw=DIALOG, source=SOURCE_MODEL),
        )
        assistant._dialog_active = True

        message = assistant.run_text("новости")

        self.assertEqual(message, "Не понял команду")
        self.assertEqual(assistant.qa.answered, [])
        self.assertEqual(assistant.intent.classified, [("новости", True)])

    def test_active_keeps_real_news(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA(kind="news")
        assistant.intent = FakeIntent(
            MODE_ACTIVE,
            IntentDecision(INFO, confidence=0.99, raw=INFO, source=SOURCE_MODEL),
        )
        assistant._dialog_active = True

        message = assistant.run_text("новости")

        self.assertEqual(message, "Короткий ответ")
        self.assertEqual(assistant.qa.answered, ["новости"])
        self.assertEqual(assistant.intent.classified, [("новости", True)])

    def test_active_does_not_gate_datetime(self) -> None:
        assistant = self.make()
        assistant.intent = FakeIntent(
            MODE_ACTIVE,
            IntentDecision(DIALOG, confidence=0.9, raw=DIALOG, source=SOURCE_MODEL),
        )
        assistant._dialog_active = True

        message = assistant.run_text("который час")

        self.assertTrue(message.startswith("Сейчас "))
        self.assertEqual(assistant.intent.classified, [])
        self.assertEqual(assistant.intent.observed, [("который час", True)])

    def test_active_keeps_real_info_search(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA(kind="search")
        assistant.intent = FakeIntent(
            MODE_ACTIVE,
            IntentDecision(INFO, confidence=0.99, raw=INFO, source=SOURCE_MODEL),
        )
        assistant._dialog_active = True

        message = assistant.run_text("курс доллара")

        self.assertEqual(message, "Короткий ответ")
        self.assertEqual(assistant.qa.answered, ["курс доллара"])
        self.assertEqual(assistant.intent.classified, [("курс доллара", True)])

    def test_active_keeps_search_on_classifier_fallback(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA(kind="search")
        assistant.intent = FakeIntent(
            MODE_ACTIVE,
            IntentDecision(
                UNKNOWN,
                confidence=0.0,
                raw="",
                source=SOURCE_FALLBACK,
            ),
        )
        assistant._dialog_active = True

        message = assistant.run_text("курс доллара")

        self.assertEqual(message, "Короткий ответ")
        self.assertEqual(assistant.qa.answered, ["курс доллара"])

    def test_unknown_outside_dialog_is_observed(self) -> None:
        assistant = self.make()
        assistant.intent = FakeIntent(MODE_SHADOW)

        with patch("app.main.beep"):
            message = assistant.run_text("сделай что-нибудь странное")

        self.assertEqual(message, "Не понял команду")
        self.assertEqual(
            assistant.intent.observed,
            [("сделай что нибудь странное", False)],
        )

    def test_new_command_interrupts_previous_speech(self) -> None:
        assistant = self.make()
        assistant.tts.speaking = True

        with patch("app.main.beep"):
            assistant.run_text("сделай что-нибудь странное")

        self.assertEqual(assistant.tts.stops, 1)


class FakeMic:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.flushed = 0

    def read(self, timeout: float = 0.5) -> bytes | None:
        return self.chunks.pop(0) if self.chunks else None

    def flush(self) -> None:
        self.flushed += 1


class FakeBarge:
    def __init__(self, match: Any) -> None:
        self.match = match
        self.resets = 0

    def feed(self, pcm: bytes) -> Any:
        return self.match

    def reset(self) -> None:
        self.resets += 1


class FakeWake:
    def __init__(self) -> None:
        self.resets = 0
        self.triggered = False
        self.last_match: str | None = None
        self.fed = 0
        self.match = None

    def feed(self, pcm: bytes) -> bool:
        self.fed += 1
        return self.triggered

    def reset(self) -> None:
        self.resets += 1


class BargeInHandlingTest(unittest.TestCase):
    def make(self, match: Any, current_text: str = "") -> Assistant:
        assistant = Assistant.__new__(Assistant)
        assistant.mic = FakeMic([b"pcm"])
        assistant.barge = FakeBarge(match)
        assistant.wake = FakeWake()
        assistant.tts = FakeTTS(enabled=True)
        assistant.tts.current_text = current_text
        assistant._preroll = deque([b"old", b"old2", b"new"], maxlen=12)
        assistant._last_wake = 0.0
        assistant.states = []
        assistant.set_state = assistant.states.append
        assistant.listen_calls = []
        assistant._listen_and_execute = lambda pre_roll=None, chime=True: (
            assistant.listen_calls.append((pre_roll, chime))
        )
        return assistant

    def test_stop_word_interrupts_speech(self) -> None:
        assistant = self.make(BargeInMatch("stop", "стоп"))

        assistant._handle_speech()

        self.assertEqual(assistant.tts.stops, 1)
        self.assertEqual(assistant.barge.resets, 1)
        self.assertEqual(assistant.wake.resets, 1)
        self.assertEqual(list(assistant._preroll), [])
        self.assertEqual(assistant.states, ["idle"])
        self.assertEqual(assistant.listen_calls, [])

    def test_own_stop_word_is_ignored(self) -> None:
        assistant = self.make(
            BargeInMatch("stop", "стоп"),
            current_text="Можно сказать стоп, чтобы остановить.",
        )

        assistant._handle_speech()

        self.assertEqual(assistant.tts.stops, 0)
        self.assertEqual(assistant.barge.resets, 1)
        self.assertEqual(assistant.listen_calls, [])

    def test_wake_word_interrupts_and_listens_without_chime(self) -> None:
        assistant = self.make(BargeInMatch("wake", "хей луна"))

        assistant._handle_speech()

        self.assertEqual(assistant.tts.stops, 1)
        self.assertEqual(assistant.barge.resets, 1)
        self.assertEqual(assistant.wake.resets, 1)
        self.assertEqual(assistant.listen_calls, [([b"old", b"old2", b"new", b"pcm"], False)])
        self.assertEqual(list(assistant._preroll), [])

    def test_no_match_keeps_speaking(self) -> None:
        assistant = self.make(None)

        assistant._handle_speech()

        self.assertEqual(assistant.tts.stops, 0)
        self.assertEqual(assistant.listen_calls, [])
        self.assertIn(b"pcm", assistant._preroll)

    def test_detector_error_does_not_break_loop(self) -> None:
        assistant = self.make(None)

        def boom(pcm: bytes) -> Any:
            raise RuntimeError("vosk упал")

        assistant.barge.feed = boom

        with self.assertLogs("app.main", level="ERROR"):
            assistant._handle_speech()

        self.assertEqual(assistant.tts.stops, 0)
        self.assertEqual(assistant.barge.resets, 1)
        self.assertEqual(assistant.wake.resets, 1)
        self.assertEqual(assistant.listen_calls, [])

    def test_wake_detector_catches_phrase_free_vosk_missed(self) -> None:
        assistant = self.make(None)
        assistant.wake.triggered = True
        assistant.wake.last_match = "хей луна"

        assistant._handle_speech()

        self.assertEqual(assistant.tts.stops, 1)
        self.assertEqual(assistant.wake.fed, 1)
        self.assertEqual(assistant.listen_calls, [([b"old", b"old2", b"new", b"pcm"], False)])
        self.assertEqual(list(assistant._preroll), [])

    def test_own_wake_phrase_is_ignored(self) -> None:
        assistant = self.make(
            None,
            current_text="Скажите «хей луна» или нажмите ctrl+alt+space.",
        )
        assistant.wake.triggered = True
        assistant.wake.last_match = "хей луна"

        assistant._handle_speech()

        self.assertEqual(assistant.tts.stops, 0)
        self.assertEqual(assistant.listen_calls, [])

    def test_barge_match_does_not_feed_wake_detector(self) -> None:
        assistant = self.make(BargeInMatch("stop", "стоп"))

        assistant._handle_speech()

        self.assertEqual(assistant.wake.fed, 0)


class SessionRecoveryTest(unittest.TestCase):
    def test_recover_sessions_calls_summarizer(self) -> None:
        calls: list[int] = []

        class FakeSummarizer:
            def recover(self) -> int:
                calls.append(1)
                return 1

        Assistant._recover_sessions(FakeSummarizer())

        self.assertEqual(calls, [1])

    def test_recover_sessions_logs_error(self) -> None:
        class BrokenSummarizer:
            def recover(self) -> int:
                raise RuntimeError("база недоступна")

        with self.assertLogs("app.main", level="ERROR"):
            Assistant._recover_sessions(BrokenSummarizer())


class FakeDialog:
    def __init__(self, answer: str = "ответ диалога", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.replied: list[str] = []
        self.hints: list[str] = []
        self.owned: list[str] = []
        self.exchanges: list[tuple[str, str]] = []
        self.started = 0
        self.stopped = 0
        self.max_errors = 2
        self.idle_timeout = 60
        self.max_utterance = 60
        self.no_speech_timeout = 10
        self.silence_ms = 1200

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def wants_entry(self, text: str) -> bool:
        return text in {"давай поговорим", "поговорим"}

    def wants_exit(self, text: str) -> bool:
        return text in {"хватит", "закончим"}

    def owns(self, text: str) -> bool:
        self.owned.append(text)
        return text in {"что ты помнишь", "будь грубее"}

    def remember_exchange(self, question: str, answer: str) -> None:
        self.exchanges.append((question, answer))

    def recent_context(self) -> str:
        return "\n".join(
            f"Пользователь: {question}\nЛуна: {answer}" for question, answer in self.exchanges
        )

    def reply(self, text: str, on_chunk=None, hint: str = "") -> str:  # noqa: ANN001
        self.replied.append(text)
        self.hints.append(hint)
        if self.error is not None:
            raise self.error
        return self.answer


class StreamingFakeStream:
    def __init__(self) -> None:
        self.fed: list[str] = []
        self.closed = 0

    def feed(self, piece: str) -> None:
        self.fed.append(piece)

    def close(self) -> None:
        self.closed += 1


class StreamingFakeTTS(FakeTTS):
    def __init__(self) -> None:
        super().__init__(enabled=True)
        self.streams: list[StreamingFakeStream] = []

    def stream(self) -> StreamingFakeStream:
        stream = StreamingFakeStream()
        self.streams.append(stream)
        return stream


class StreamingFakeDialog(FakeDialog):
    """Диалог, который отдаёт ответ кусками через on_chunk."""

    streaming = True

    def __init__(
        self,
        answer: str = "ответ диалога",
        pieces: list[str] | None = None,
        error_after: Exception | None = None,
    ) -> None:
        super().__init__(answer)
        self.pieces = pieces if pieces is not None else [answer]
        self.error_after = error_after

    def reply(self, text: str, on_chunk=None, hint: str = "") -> str:  # noqa: ANN001
        self.replied.append(text)
        self.hints.append(hint)
        for piece in self.pieces:
            if on_chunk is not None:
                on_chunk(piece)
        if self.error_after is not None:
            raise self.error_after
        return self.answer


class DialogModeTest(unittest.TestCase):
    def make(self, result: CommandResult | None = None) -> Assistant:
        assistant = Assistant.__new__(Assistant)
        assistant.router = FakeRouter(result)
        assistant.planner = FakeLocalPlanner()
        assistant.qa = None
        assistant.tts = FakeTTS()
        assistant.notifications = []
        assistant.states = []
        assistant.notify = lambda message, title="Ассистент": assistant.notifications.append(
            message
        )
        assistant.set_state = lambda state: assistant.states.append(state)
        assistant.dialog = FakeDialog()
        assistant.visual_router = VisualRouter({"vision": {}})
        assistant.cfg = {"dialog": {"auto_start": "off"}}
        assistant._dialog_active = False
        assistant._dialog_last = 0.0
        assistant._dialog_errors = 0
        assistant._vision_context_at = 0.0
        return assistant

    def test_entry_phrase_enables_dialog(self) -> None:
        assistant = self.make()
        with patch("app.main.beep"):
            message = assistant.run_text("давай поговорим")
        self.assertTrue(assistant._dialog_active)
        self.assertEqual(message, "Слушаю. О чём поговорим?")
        self.assertEqual(assistant.router.seen, [])
        self.assertIn(message, assistant.tts.said)
        self.assertEqual(assistant.dialog.started, 1)

    def test_auto_start_always_enters_dialog_on_wake_speech(self) -> None:
        assistant = self.make()
        assistant.cfg["dialog"]["auto_start"] = "always"

        message = assistant._process("как дела", voiced=True, wake=True)

        self.assertTrue(assistant._dialog_active)
        self.assertEqual(assistant.dialog.started, 1)
        self.assertEqual(assistant.dialog.replied, ["как дела"])
        self.assertEqual(message, "ответ диалога")
        self.assertNotIn("Слушаю. О чём поговорим?", assistant.tts.said)

    def test_auto_start_always_keeps_command_and_dialog(self) -> None:
        assistant = self.make(CommandResult(True, True, "Открываю YouTube"))
        assistant.cfg["dialog"]["auto_start"] = "always"

        message = assistant._process("открой ютуб", voiced=True, wake=True)

        self.assertTrue(assistant._dialog_active)
        self.assertEqual(assistant.router.seen, ["открой ютуб"])
        self.assertEqual(message, "Открываю YouTube")
        self.assertEqual(assistant.dialog.replied, [])

    def test_auto_start_conversation_does_not_enter_on_command(self) -> None:
        assistant = self.make(CommandResult(True, True, "Открываю YouTube"))
        assistant.cfg["dialog"]["auto_start"] = "conversation"

        message = assistant._process("открой ютуб", voiced=True, wake=True)

        self.assertFalse(assistant._dialog_active)
        self.assertEqual(message, "Открываю YouTube")
        self.assertEqual(assistant.dialog.started, 0)

    def test_auto_start_off_does_not_enter_dialog(self) -> None:
        assistant = self.make()
        assistant.cfg["dialog"]["auto_start"] = "off"

        assistant._process("как дела", voiced=True, wake=True)

        self.assertFalse(assistant._dialog_active)
        self.assertEqual(assistant.dialog.started, 0)

    def test_entry_phrase_still_greets_with_auto_start(self) -> None:
        assistant = self.make()
        assistant.cfg["dialog"]["auto_start"] = "always"
        with patch("app.main.beep"):
            message = assistant._process("давай поговорим", voiced=True, wake=True)

        self.assertTrue(assistant._dialog_active)
        self.assertEqual(message, "Слушаю. О чём поговорим?")
        self.assertIn(message, assistant.tts.said)

    def test_start_dialog_without_greeting(self) -> None:
        assistant = self.make()

        message = assistant._start_dialog(greet=False)

        self.assertEqual(message, "")
        self.assertTrue(assistant._dialog_active)
        self.assertEqual(assistant.dialog.started, 1)
        self.assertEqual(assistant.tts.said, [])

    def test_exit_phrase_disables_dialog(self) -> None:
        assistant = self.make()
        assistant._dialog_active = True
        message = assistant.run_text("хватит")
        self.assertFalse(assistant._dialog_active)
        self.assertEqual(message, "Хорошо, закончили.")
        self.assertEqual(assistant.router.seen, [])
        self.assertEqual(assistant.dialog.stopped, 1)

    def test_free_speech_goes_to_dialog_not_qa(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA(kind="general")
        assistant._dialog_active = True
        message = assistant.run_text("как дела")
        self.assertEqual(message, "ответ диалога")
        self.assertEqual(assistant.dialog.replied, ["как дела"])
        self.assertEqual(assistant.qa.answered, [])
        self.assertEqual(assistant.planner.seen, [])

    def test_command_inside_dialog_stays_a_command(self) -> None:
        assistant = self.make(CommandResult(True, True, "Открываю YouTube"))
        assistant._dialog_active = True
        message = assistant.run_text("открой ютуб")
        self.assertEqual(message, "Открываю YouTube")
        self.assertEqual(assistant.dialog.replied, [])
        self.assertTrue(assistant._dialog_active)

    def test_dialog_followup_uses_pending_snapshot_without_pointer_words(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant.vision.pending = True
        assistant._dialog_active = True

        message = assistant.run_text("реши номер 325")

        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(assistant.vision.seen, ["реши номер 325"])
        self.assertEqual(assistant.dialog.replied, [])
        self.assertEqual(assistant.router.seen, ["реши номер 325"])

    def test_dialog_question_after_vision_keeps_history(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant._dialog_active = True
        assistant.run_text("посмотри на экран")
        assistant.run_text("что ты номер 353 видишь")
        assistant.vision.seen.clear()
        assistant.dialog.replied.clear()
        assistant.dialog.hints.clear()

        message = assistant.run_text("найди область определения функции")

        self.assertEqual(message, "ответ диалога")
        self.assertEqual(assistant.vision.seen, [])
        self.assertEqual(assistant.dialog.replied, ["найди область определения функции"])
        self.assertIn("описание", assistant.dialog.hints[-1])

    def test_zero_context_ttl_disables_description_fallback(self) -> None:
        assistant = self.make()
        assistant.visual_router = VisualRouter({"vision": {"context_ttl_sec": 0}})
        assistant._vision_context_at = time.monotonic()

        self.assertFalse(assistant._vision_context_recent())

    def test_new_snapshot_clears_old_visual_context(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant._dialog_active = True
        assistant.run_text("посмотри на экран, что за ошибка")
        self.assertGreater(assistant._vision_context_at, 0.0)

        assistant.run_text("посмотри на экран")

        self.assertEqual(assistant._vision_context_at, 0.0)

    def test_vision_followup_receives_dialog_context(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant._dialog_active = True
        assistant.dialog.remember_exchange("обсуждали номер 353", "Номер 353: эллипс")

        assistant.run_text("посмотри на экран")
        assistant.run_text("реши его")

        self.assertEqual(assistant.vision.seen, ["реши его"])
        self.assertIn("обсуждали номер 353", assistant.vision.contexts[-1])

    def test_expired_frame_answers_from_description(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant._dialog_active = True
        assistant.run_text("посмотри на экран, что за ошибка")
        assistant.vision.pending = False
        assistant.vision.error = VisionError("нет свежего снимка экрана")
        assistant.dialog.replied.clear()
        assistant.dialog.hints.clear()

        message = assistant._answer_pending_vision(assistant.vision, "где здесь кнопка")

        self.assertEqual(message, "ответ диалога")
        self.assertEqual(assistant.dialog.replied, ["где здесь кнопка"])
        self.assertIn("описание", assistant.dialog.hints[-1])

    def test_dialog_followup_does_not_steal_command(self) -> None:
        assistant = self.make(CommandResult(True, True, "Открываю YouTube"))
        assistant.vision = FakeVision()
        assistant.vision.pending = True
        assistant._dialog_active = True

        message = assistant.run_text("открой ютуб")

        self.assertEqual(message, "Открываю YouTube")
        self.assertEqual(assistant.vision.seen, [])
        self.assertEqual(assistant.dialog.replied, [])
        self.assertEqual(assistant.router.seen, ["открой ютуб"])

    def test_dialog_followup_skips_qa_topic(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant.vision.pending = True
        assistant.qa = FakeQA(kind="weather")
        assistant._dialog_active = True

        message = assistant.run_text("какая погода")

        self.assertEqual(message, "Короткий ответ")
        self.assertEqual(assistant.vision.seen, [])
        self.assertEqual(assistant.qa.answered, ["какая погода"])

    def test_dialog_followup_skips_dialog_owned_phrase(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant.vision.pending = True
        assistant._dialog_active = True

        message = assistant.run_text("что ты помнишь")

        self.assertEqual(message, "ответ диалога")
        self.assertEqual(assistant.vision.seen, [])
        self.assertEqual(assistant.dialog.replied, ["что ты помнишь"])
        self.assertEqual(assistant.dialog.owned, ["что ты помнишь"])

    def test_vision_answer_is_saved_to_dialog_history(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant.vision.pending = True
        assistant._dialog_active = True

        assistant.run_text("реши номер 325")

        self.assertEqual(
            assistant.dialog.exchanges,
            [("реши номер 325", "На экране открыт браузер")],
        )

    def test_one_shot_vision_saves_full_question_to_dialog_history(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant._dialog_active = True

        assistant.run_text("посмотри на экран, что за ошибка")

        self.assertEqual(
            assistant.dialog.exchanges,
            [("посмотри на экран что за ошибка", "На экране открыт браузер")],
        )

    def test_one_shot_vision_is_saved_without_dialog(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant._dialog_active = False

        assistant.run_text("посмотри на экран, что за ошибка")

        self.assertEqual(
            assistant.dialog.exchanges,
            [("посмотри на экран что за ошибка", "На экране открыт браузер")],
        )

    def test_visual_context_answers_without_new_snapshot(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant.run_text("посмотри на экран, что за ошибка")
        assistant.dialog.replied.clear()
        assistant.dialog.hints.clear()

        # Кадр ещё в памяти, но описание уже есть: вопрос идёт по тексту.
        message = assistant.run_text("что написано на экране")

        self.assertEqual(message, "ответ диалога")
        self.assertEqual(assistant.dialog.replied, ["что написано на экране"])
        self.assertIn("описание", assistant.dialog.hints[-1])
        self.assertEqual(assistant.vision.seen, ["что за ошибка"])

    def test_visual_context_in_dialog_uses_hint(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant._dialog_active = True
        assistant.run_text("посмотри на экран, что за ошибка")
        assistant.dialog.replied.clear()
        assistant.dialog.hints.clear()

        message = assistant.run_text("а что в левом верхнем углу")

        self.assertEqual(message, "ответ диалога")
        self.assertEqual(assistant.dialog.replied, ["а что в левом верхнем углу"])
        self.assertIn("описание", assistant.dialog.hints[-1])
        self.assertEqual(assistant.vision.seen, ["что за ошибка"])

    def test_visual_question_without_context_asks_for_screen(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()

        message = assistant.run_text("что на фото")

        self.assertIn("Покажи экран", message)
        self.assertEqual(assistant.dialog.replied, [])

    def test_specific_qa_topic_wins_over_deictic_visual(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant.run_text("посмотри на экран, что за ошибка")
        assistant.qa = FakeQA(kind="weather")

        message = assistant.run_text("что там с погодой")

        self.assertEqual(message, "Короткий ответ")
        self.assertEqual(assistant.qa.answered, ["что там с погодой"])
        self.assertEqual(assistant.dialog.replied, [])

    def test_general_route_does_not_steal_visual_question(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant.run_text("посмотри на экран, что за ошибка")
        assistant.qa = FakeQA(kind="general")
        assistant.dialog.replied.clear()
        assistant.dialog.hints.clear()

        message = assistant.run_text("где здесь кнопка")

        self.assertEqual(message, "ответ диалога")
        self.assertEqual(assistant.qa.answered, [])
        self.assertEqual(assistant.vision.seen, ["что за ошибка"])

    def test_general_question_after_vision_stays_text(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant.run_text("посмотри на экран, что за ошибка")
        assistant.vision.pending = False

        with patch("app.main.beep"):
            message = assistant.run_text("какая погода")

        self.assertEqual(message, "Не понял команду")
        self.assertEqual(assistant.dialog.replied, [])

    def test_explicit_vision_command_still_takes_new_snapshot(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant.run_text("посмотри на экран, что за ошибка")
        assistant.vision.pending = False
        assistant.vision.seen.clear()

        message = assistant.run_text("посмотри на экран, что написано")

        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(assistant.vision.seen, ["что написано"])

    def test_stopping_dialog_clears_visual_context(self) -> None:
        assistant = self.make()
        assistant.vision = FakeVision()
        assistant._dialog_active = True
        assistant.run_text("посмотри на экран, что за ошибка")

        assistant._stop_dialog("phrase")

        self.assertEqual(assistant._vision_context_at, 0.0)

    def test_background_voice_utterance_is_ignored(self) -> None:
        assistant = self.make()
        assistant.addressee = FakeAddressee(BACKGROUND)
        assistant._dialog_active = True
        assistant._dialog_last = time.monotonic() - 5
        assistant._dialog_ignored = 0
        assistant._dialog_turns = 0
        before = assistant._dialog_last

        message = assistant._process("реклама про скидки", voiced=True)

        self.assertEqual(message, "")
        self.assertEqual(assistant.dialog.replied, [])
        self.assertEqual(assistant.router.seen, [])
        self.assertEqual(assistant._dialog_ignored, 1)
        self.assertGreater(assistant._dialog_last, before)

    def test_weak_addressed_after_background_is_skipped(self) -> None:
        assistant = self.make()
        assistant.addressee = FakeAddressee(ADDRESSED, weak=True)
        assistant._dialog_active = True
        assistant._dialog_last = time.monotonic() - 5
        assistant._dialog_ignored = 2
        assistant._dialog_turns = 2
        before = assistant._dialog_last

        message = assistant._process("ну да, наверное", voiced=True)

        self.assertEqual(message, "")
        self.assertEqual(assistant.dialog.replied, [])
        self.assertEqual(assistant._dialog_ignored, 3)
        self.assertGreater(assistant._dialog_last, before)

    def test_weak_addressed_without_background_answers(self) -> None:
        assistant = self.make()
        assistant.addressee = FakeAddressee(ADDRESSED, weak=True)
        assistant._dialog_active = True
        assistant._dialog_last = time.monotonic() - 5
        assistant._dialog_ignored = 0
        assistant._dialog_turns = 2

        message = assistant._process("ну да, наверное", voiced=True)

        self.assertEqual(message, "ответ диалога")
        self.assertEqual(assistant.dialog.replied, ["ну да наверное"])
        self.assertEqual(assistant._dialog_ignored, 0)

    def test_confident_addressed_resets_background_streak(self) -> None:
        assistant = self.make()
        assistant.addressee = FakeAddressee(ADDRESSED)
        assistant._dialog_active = True
        assistant._dialog_last = time.monotonic() - 5
        assistant._dialog_ignored = 2
        assistant._dialog_turns = 2
        before = assistant._dialog_last

        assistant._process("расскажи, как прошёл день", voiced=True)

        self.assertEqual(assistant._dialog_ignored, 0)
        self.assertGreater(assistant._dialog_last, before)

    def test_shadow_mode_still_answers_background(self) -> None:
        assistant = self.make()
        assistant.addressee = FakeAddressee(BACKGROUND, mode=ADDRESSEE_SHADOW)
        assistant._dialog_active = True
        assistant._dialog_last = time.monotonic()
        assistant._dialog_turns = 0

        message = assistant._process("реклама про скидки", voiced=True)

        self.assertEqual(message, "ответ диалога")
        self.assertEqual(assistant.dialog.replied, ["реклама про скидки"])

    def test_closing_voice_utterance_stops_dialog(self) -> None:
        assistant = self.make()
        assistant.addressee = FakeAddressee(CLOSING)
        assistant._dialog_active = True
        assistant._dialog_last = time.monotonic()
        assistant._dialog_turns = 0

        message = assistant._process("ну все спасибо", voiced=True)

        self.assertFalse(assistant._dialog_active)
        self.assertEqual(message, "Хорошо, закончили.")
        self.assertEqual(assistant.dialog.stopped, 1)
        self.assertEqual(assistant.dialog.replied, [])

    def test_background_does_not_close_dialog(self) -> None:
        assistant = self.make()
        assistant.addressee = FakeAddressee(ADDRESSED)
        assistant.mic = None
        assistant._dialog_active = True
        assistant._dialog_last = time.monotonic() - 1
        assistant._dialog_ignored = 5
        assistant._dialog_turns = 5

        assistant._dialog_step()

        self.assertTrue(assistant._dialog_active)
        self.assertEqual(assistant.dialog.stopped, 0)

    def test_idle_silence_closes_dialog(self) -> None:
        assistant = self.make()
        assistant.addressee = FakeAddressee(ADDRESSED)
        assistant.mic = None
        assistant._dialog_active = True
        assistant._dialog_last = time.monotonic() - 61
        assistant._dialog_ignored = 0
        assistant._dialog_turns = 1

        assistant._dialog_step()

        self.assertFalse(assistant._dialog_active)
        self.assertEqual(assistant.dialog.stopped, 1)

    def test_gate_receives_dialog_turn_number(self) -> None:
        assistant = self.make()
        assistant.addressee = FakeAddressee(ADDRESSED)
        assistant._dialog_active = True
        assistant._dialog_last = time.monotonic()
        assistant._dialog_turns = 0

        assistant._process("раз", voiced=True)
        assistant._process("два", voiced=True)

        self.assertEqual([item[1] for item in assistant.addressee.seen], [True, False])

    def test_weather_inside_dialog_uses_qa(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA(
            QAAnswer(True, "погода", "Ясно", "Ясно", "weather"),
            kind="weather",
        )
        assistant._dialog_active = True
        message = assistant.run_text("какая погода")
        self.assertEqual(message, "Ясно")
        self.assertEqual(assistant.qa.answered, ["какая погода"])
        self.assertEqual(assistant.dialog.replied, [])

    def test_dialog_exits_after_error_limit(self) -> None:
        assistant = self.make()
        assistant.dialog.error = RuntimeError("boom")
        assistant._dialog_active = True
        with (
            patch("app.main.beep"),
            patch("app.main.log.exception"),
        ):
            assistant.run_text("раз")
            assistant.run_text("два")
        self.assertFalse(assistant._dialog_active)
        self.assertEqual(len(assistant.dialog.replied), 2)
        self.assertEqual(assistant.dialog.stopped, 1)

    def test_streaming_dialog_feeds_chunks_and_does_not_repeat_answer(self) -> None:
        assistant = self.make()
        assistant.tts = StreamingFakeTTS()
        assistant.dialog = StreamingFakeDialog(
            answer="Привет. Как дела?",
            pieces=["Привет. ", "Как дела?"],
        )
        assistant._dialog_active = True
        assistant._dialog_last = time.monotonic()

        message = assistant.run_text("привет")

        self.assertEqual(message, "Привет. Как дела?")
        self.assertEqual(len(assistant.tts.streams), 1)
        self.assertEqual(assistant.tts.streams[0].fed, ["Привет. ", "Как дела?"])
        self.assertEqual(assistant.tts.streams[0].closed, 1)
        self.assertEqual(assistant.tts.said, [])

    def test_streaming_without_tts_speaks_answer_at_end(self) -> None:
        assistant = self.make()
        assistant.dialog = StreamingFakeDialog(
            answer="Привет.",
            pieces=["Привет."],
        )
        assistant._dialog_active = True
        assistant._dialog_last = time.monotonic()

        assistant.run_text("привет")

        self.assertEqual(assistant.tts.said, ["Привет."])

    def test_streaming_error_after_chunks_closes_stream(self) -> None:
        assistant = self.make()
        assistant.tts = StreamingFakeTTS()
        assistant.dialog = StreamingFakeDialog(
            answer="Привет.",
            pieces=["При"],
            error_after=RuntimeError("обрыв"),
        )
        assistant._dialog_last = time.monotonic()
        assistant._dialog_active = True
        with patch("app.main.log.exception"):
            message = assistant.run_text("привет")
        self.assertEqual(message, "Не получилось ответить.")
        self.assertEqual(assistant.tts.streams[0].closed, 1)
        self.assertEqual(assistant.tts.said, ["Не получилось ответить."])

    def test_cleanup_closes_dialog_session(self) -> None:
        assistant = self.make()
        assistant.mic = None
        assistant._hotkey_listener = None
        assistant.browser = SimpleNamespace(stop=lambda: None)
        assistant.tray = None
        assistant._cleanup()
        self.assertEqual(assistant.dialog.stopped, 1)

    def test_status_reports_dialog(self) -> None:
        assistant = self.make()
        assistant.wake = None
        assistant.tray = None
        assistant.cfg = {"wake_word": "хей луна", "hotkey": "ctrl+alt+space"}
        assistant.stt = SimpleNamespace(loaded=False)
        assistant._state_lock = threading.Lock()
        assistant._state = "idle"
        assistant._paused = False
        self.assertFalse(assistant.status()["dialog"])
        assistant._dialog_active = True
        self.assertTrue(assistant.status()["dialog"])


class SilentMic:
    """Микрофон для проверки лимитов записи: всегда отдаёт тишину."""

    def __init__(self, chunk: bytes) -> None:
        self.chunk = chunk
        self.reads = 0

    def read(self, timeout: float = 0) -> bytes:
        self.reads += 1
        return self.chunk


class RecordDialogLimitsTest(unittest.TestCase):
    def make(self, *, dialog: bool) -> Assistant:
        assistant = Assistant.__new__(Assistant)
        # 1600 байт PCM16 при 16 кГц — 50 мс.
        assistant.mic = SilentMic(b"\x00\x00" * 800)
        assistant.cfg = {
            "audio": {
                "energy_threshold": 0.01,
                "noise_multiplier": 2.0,
                "silence_ms": 800,
                "min_speech_ms": 250,
                "max_command_sec": 15,
                "no_speech_timeout_sec": 3,
            },
            "dialog": {
                "silence_ms": 1200,
                "max_utterance_sec": 60,
                "no_speech_timeout_sec": 1,
            },
        }
        assistant._noise_floor = 0.0
        assistant.vad = None
        return assistant

    def test_dialog_uses_its_own_no_speech_timeout(self) -> None:
        assistant = self.make(dialog=True)
        self.assertEqual(assistant._record_command(dialog=True), b"")
        self.assertEqual(assistant.mic.reads, 20)

    def test_command_keeps_audio_timeout(self) -> None:
        assistant = self.make(dialog=False)
        self.assertEqual(assistant._record_command(dialog=False), b"")
        self.assertEqual(assistant.mic.reads, 60)

    def test_pause_stops_recording_immediately(self) -> None:
        assistant = self.make(dialog=False)
        assistant._paused = True
        self.assertEqual(assistant._record_command(dialog=False), b"")
        self.assertEqual(assistant.mic.reads, 0)


class VisionWatchRoutingTest(unittest.TestCase):
    def make(self) -> Assistant:
        assistant = Assistant.__new__(Assistant)
        assistant.router = FakeRouter()
        assistant.planner = FakeLocalPlanner()
        assistant.qa = None
        assistant.vision = FakeVision()
        assistant.image_search = FakeImageSearch()
        assistant.llm = FakeLLM()
        assistant.tts = FakeTTS(enabled=True)
        assistant.vision_watch = FakeVisionWatch()
        assistant.vision_comments = FakeVisionComments()
        assistant.notifications = []
        assistant.notify = lambda message, title="Ассистент": assistant.notifications.append(
            message
        )
        assistant._state_lock = threading.Lock()
        assistant._state = "idle"
        assistant._paused = False
        assistant._dialog_active = False
        assistant.states = []
        assistant.set_state = assistant.states.append
        return assistant

    def test_enable_starts_observation_without_router_or_vision(self) -> None:
        assistant = self.make()

        message = assistant.run_text("включи зрение")

        self.assertTrue(assistant.vision_watch.active)
        self.assertIn("Наблюдение включено", message)
        self.assertEqual(assistant.router.seen, [])
        self.assertEqual(assistant.vision.seen, [])
        self.assertEqual(assistant.vision.captures, 0)
        self.assertEqual(assistant.tts.said, [message])

    def test_enable_twice_reports_already_on(self) -> None:
        assistant = self.make()
        assistant.run_text("включи зрение")

        message = assistant.run_text("включи наблюдение")

        self.assertEqual(assistant.vision_watch.start_calls, 2)
        self.assertEqual(message, "Наблюдение уже включено.")

    def test_disable_stops_observation(self) -> None:
        assistant = self.make()
        assistant.run_text("включи зрение")

        message = assistant.run_text("выключи зрение")

        self.assertFalse(assistant.vision_watch.active)
        self.assertIn("Наблюдение выключено", message)
        self.assertEqual(assistant.router.seen, [])

    def test_disable_when_off_is_safe(self) -> None:
        assistant = self.make()

        message = assistant.run_text("выключи наблюдение")

        self.assertEqual(message, "Наблюдение и так выключено.")

    def test_enable_comments_starts_observation_and_comments(self) -> None:
        assistant = self.make()

        message = assistant.run_text("включи комментарии")

        self.assertTrue(assistant.vision_watch.active)
        self.assertTrue(assistant.vision_comments.active)
        self.assertEqual(assistant.vision_comments.start_calls, 1)
        self.assertIn("Наблюдение и комментарии включены", message)
        self.assertIn("кадра", message)
        self.assertEqual(assistant.router.seen, [])
        self.assertEqual(assistant.vision.seen, [])

    def test_enable_comments_keeps_running_observation(self) -> None:
        assistant = self.make()
        assistant.run_text("включи зрение")

        message = assistant.run_text("включи комментарии")

        self.assertEqual(assistant.vision_watch.start_calls, 1)
        self.assertTrue(assistant.vision_comments.active)
        self.assertIn("Комментарии включены", message)

    def test_disable_comments_keeps_observation(self) -> None:
        assistant = self.make()
        assistant.run_text("включи комментарии")

        message = assistant.run_text("выключи комментарии")

        self.assertFalse(assistant.vision_comments.active)
        self.assertTrue(assistant.vision_watch.active)
        self.assertEqual(message, "Комментарии выключены. Наблюдение продолжается.")

    def test_disable_observation_stops_comments(self) -> None:
        assistant = self.make()
        assistant.run_text("включи комментарии")

        message = assistant.run_text("выключи зрение")

        self.assertFalse(assistant.vision_comments.active)
        self.assertFalse(assistant.vision_watch.active)
        self.assertIn("Комментарии тоже выключены", message)

    def test_comments_status_is_reported(self) -> None:
        assistant = self.make()
        assistant.vision_comments.described = "Комментарии включены: 1 комментарий."

        message = assistant.run_text("статус комментариев")

        self.assertEqual(message, "Комментарии включены: 1 комментарий.")
        self.assertEqual(assistant.router.seen, [])

    def test_comments_disabled_in_config_are_reported(self) -> None:
        assistant = self.make()
        assistant.vision_comments.available = False

        message = assistant.run_text("включи комментарии")

        self.assertFalse(assistant.vision_comments.active)
        self.assertIn("выключены в настройках", message)

    def test_mic_pause_pauses_and_resumes_comments(self) -> None:
        assistant = self.make()
        assistant.mic = None
        assistant.run_text("включи комментарии")

        assistant.toggle_pause()
        self.assertTrue(assistant.vision_comments.paused)

        assistant.toggle_pause()
        self.assertFalse(assistant.vision_comments.paused)

    def test_comment_busy_during_dialog_and_speech(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant._dialog_active = False
        assistant._state = "idle"
        self.assertFalse(assistant._vision_comment_busy())

        assistant._dialog_active = True
        self.assertTrue(assistant._vision_comment_busy())

        assistant._dialog_active = False
        assistant.tts.speaking = True
        self.assertTrue(assistant._vision_comment_busy())

        assistant.tts.speaking = False
        assistant._state = "listening"
        self.assertTrue(assistant._vision_comment_busy())

    def test_timeout_stops_comments_too(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant.vision_watch.auto_off_sec = 60.0
        assistant.vision_comments.active = True

        assistant._vision_watch_stopped("timeout")

        self.assertFalse(assistant.vision_comments.active)
        self.assertIn("комментарии выключены", assistant.notifications[0].lower())

    @staticmethod
    def frame(age: float = 0.0) -> Frame:
        return Frame(
            image=Image.new("RGB", (32, 18), "white"),
            captured_at=time.monotonic() - age,
        )

    @staticmethod
    def watch_event(
        *,
        baseline: bool = False,
        changed: bool = False,
        window_changed: bool = False,
        title: str | None = None,
    ) -> WatchEvent:
        result = None
        if not baseline:
            result = ChangeResult(
                changed=changed,
                score=0.2 if changed else 0.0,
                pixel_diff=0.1 if changed else 0.0,
                hash_distance=3 if changed else 0,
                hash_ratio=0.05 if changed else 0.0,
                reason="content" if changed else "same",
            )
        return WatchEvent(
            at=time.monotonic(),
            changed=changed or window_changed,
            window_changed=window_changed,
            result=result,
            width=320,
            height=180,
            window_title=title,
        )

    def add_key_frames(self, assistant: Assistant, count: int = 2) -> None:
        timeline = assistant.vision_watch.timeline
        now = time.monotonic() - 10.0
        timeline.record(
            self.watch_event(baseline=True, title="Терминал"),
            now=now,
            frame=self.frame(age=10.0),
        )
        for index in range(1, count):
            timeline.record(
                self.watch_event(window_changed=True, title=f"Окно {index}"),
                now=now + index,
                frame=self.frame(age=10.0 - index),
            )

    def test_explicit_question_uses_observed_frame(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        frame = self.frame()
        assistant.vision_watch.last_frame = frame

        message = assistant.run_text("посмотри на экран, что за ошибка")

        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(assistant.vision.frame_answers, 1)
        self.assertEqual(assistant.vision.describe_calls, 0)
        self.assertEqual(assistant.vision.snapshots, 0)
        self.assertEqual(assistant.vision.captures, 0)
        self.assertEqual(assistant.vision.frames, [frame])

    def test_snapshot_only_uses_observed_frame(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant.vision_watch.last_frame = self.frame()

        message = assistant.run_text("посмотри на экран")

        self.assertEqual(message, "Снимок экрана готов. Спрашивай, что на нём.")
        self.assertEqual(assistant.vision.remembered, 1)
        self.assertEqual(assistant.vision.snapshots, 0)
        self.assertEqual(assistant.vision.captures, 0)

    def test_stale_observed_frame_falls_back_to_new_capture(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant.vision_watch.last_frame = self.frame(age=60.0)

        message = assistant.run_text("посмотри на экран, что за ошибка")

        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(assistant.vision.frame_answers, 0)
        self.assertEqual(assistant.vision.describe_calls, 1)

    def test_inactive_observation_ignores_last_frame(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = False
        assistant.vision_watch.last_frame = self.frame()

        assistant.run_text("что на экране")

        self.assertEqual(assistant.vision.frame_answers, 0)
        self.assertEqual(assistant.vision.describe_calls, 1)

    def test_observation_without_frame_uses_new_capture(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True

        assistant.run_text("что на экране")

        self.assertEqual(assistant.vision.frame_answers, 0)
        self.assertEqual(assistant.vision.describe_calls, 1)

    def test_frame_freshness_scales_with_interval(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant.vision_watch.interval_sec = 30.0
        assistant.vision_watch.last_frame = self.frame(age=20.0)

        assistant.run_text("что происходит на экране")

        self.assertEqual(assistant.vision.frame_answers, 1)
        self.assertEqual(assistant.vision.describe_calls, 0)

    def test_observed_frame_stays_for_followup(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant.vision_watch.last_frame = self.frame()
        assistant.run_text("посмотри на экран, что за ошибка")
        assistant.vision.seen.clear()

        message = assistant.run_text("где здесь кнопка")

        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(assistant.vision.seen, ["где здесь кнопка"])
        self.assertEqual(assistant.vision.frame_answers, 1)

    def make_with_visual_model(self, visual: IntentDecision) -> Assistant:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant.vision_watch.last_frame = self.frame()
        assistant.visual_router = VisualRouter({"vision": {"context_ttl_sec": 1800}})
        assistant.intent = FakeIntent(visual=visual)
        return assistant

    @staticmethod
    def model_choice(intent: str, confidence: float = 0.9) -> IntentDecision:
        return IntentDecision(
            intent,
            confidence=confidence,
            raw=intent,
            source=SOURCE_MODEL,
        )

    def test_model_screen_decision_answers_by_observed_frame(self) -> None:
        assistant = self.make_with_visual_model(self.model_choice(SCREEN))

        message = assistant.run_text("почему это не работает")

        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(assistant.vision.frame_answers, 1)
        self.assertEqual(assistant.vision.describe_calls, 0)
        self.assertEqual(assistant.vision.snapshots, 0)
        self.assertEqual(
            assistant.intent.visual_classified,
            [("почему это не работает", False)],
        )

    def test_activity_question_uses_observed_frame(self) -> None:
        assistant = self.make_with_visual_model(self.model_choice(SCREEN))

        message = assistant.run_text("что я сейчас делаю")

        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(assistant.vision.frame_answers, 1)
        self.assertEqual(assistant.vision.describe_calls, 0)
        self.assertEqual(assistant.vision.snapshots, 0)
        self.assertEqual(
            assistant.intent.visual_classified,
            [("что я сейчас делаю", False)],
        )

    def test_activity_question_other_keeps_chat(self) -> None:
        assistant = self.make_with_visual_model(self.model_choice(OTHER))

        with patch("app.main.beep"):
            assistant.run_text("что я сейчас делаю")

        self.assertEqual(
            assistant.intent.visual_classified,
            [("что я сейчас делаю", False)],
        )
        self.assertEqual(assistant.vision.frame_answers, 0)

    def test_activity_question_after_vision_answer_uses_observed_frame(self) -> None:
        assistant = self.make_with_visual_model(self.model_choice(SCREEN))
        # Предыдущий ответ по кадру оставил снимок для уточнений.
        assistant.vision.pending = True

        message = assistant.run_text("что я сейчас делаю")

        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(assistant.vision.frame_answers, 1)
        self.assertEqual(assistant.vision.describe_calls, 0)
        self.assertEqual(
            assistant.intent.visual_classified,
            [("что я сейчас делаю", False)],
        )

    def test_pending_followup_does_not_ask_visual_classifier(self) -> None:
        assistant = self.make_with_visual_model(self.model_choice(SCREEN))
        assistant.vision.pending = True

        message = assistant.run_text("где здесь кнопка")

        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(assistant.intent.visual_classified, [])
        self.assertEqual(assistant.vision.frame_answers, 0)

    def test_model_screen_decision_replaces_clarify_for_ambiguous(self) -> None:
        assistant = self.make_with_visual_model(self.model_choice(SCREEN))

        message = assistant.run_text("что в левом верхнем углу")

        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(assistant.vision.frame_answers, 1)
        self.assertNotIn("Покажи экран", " ".join(assistant.notifications))

    def test_model_other_decision_skips_clarify(self) -> None:
        assistant = self.make_with_visual_model(self.model_choice(OTHER))

        with patch("app.main.beep"):
            message = assistant.run_text("что в левом верхнем углу")

        self.assertEqual(message, "Не понял команду")
        self.assertEqual(assistant.vision.frame_answers, 0)
        self.assertNotIn("Покажи экран", " ".join(assistant.notifications))

    def test_uncertain_model_keeps_old_clarify(self) -> None:
        assistant = self.make_with_visual_model(
            IntentDecision(
                UNKNOWN,
                confidence=0.4,
                raw=SCREEN,
                source=SOURCE_MODEL,
            )
        )

        message = assistant.run_text("что в левом верхнем углу")

        self.assertIn("Покажи экран", message)
        self.assertEqual(assistant.vision.frame_answers, 0)

    def test_general_qa_route_still_asks_the_model(self) -> None:
        assistant = self.make_with_visual_model(self.model_choice(SCREEN))
        assistant.qa = FakeQA(kind="general")

        message = assistant.run_text("почему это не работает")

        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(assistant.vision.frame_answers, 1)
        self.assertEqual(assistant.qa.answered, [])

    def test_general_qa_route_answers_normally_for_other(self) -> None:
        assistant = self.make_with_visual_model(self.model_choice(OTHER))
        assistant.qa = FakeQA(kind="general")

        message = assistant.run_text("что в левом верхнем углу")

        self.assertEqual(message, "Короткий ответ")
        self.assertEqual(assistant.qa.answered, ["что в левом верхнем углу"])
        self.assertEqual(assistant.vision.frame_answers, 0)

    def test_saved_description_does_not_ask_the_model(self) -> None:
        assistant = self.make_with_visual_model(self.model_choice(SCREEN))
        assistant._vision_context_at = time.monotonic()

        message = assistant.run_text("что написано на экране")

        self.assertIn("Покажи экран", message)
        self.assertEqual(assistant.intent.visual_classified, [])
        self.assertEqual(assistant.vision.frame_answers, 0)

    def test_explicit_vision_does_not_ask_the_model(self) -> None:
        assistant = self.make_with_visual_model(self.model_choice(SCREEN))

        message = assistant.run_text("посмотри на экран, что за ошибка")

        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(assistant.vision.frame_answers, 1)
        self.assertEqual(assistant.intent.visual_classified, [])

    def test_command_does_not_ask_the_model(self) -> None:
        assistant = self.make_with_visual_model(self.model_choice(SCREEN))

        with patch("app.main.beep"):
            assistant.run_text("нажми на эту кнопку")

        self.assertEqual(assistant.intent.visual_classified, [])
        self.assertEqual(assistant.vision.frame_answers, 0)

    def test_plain_talk_does_not_ask_the_model(self) -> None:
        assistant = self.make_with_visual_model(self.model_choice(SCREEN))

        with patch("app.main.beep"):
            assistant.run_text("как дела")

        self.assertEqual(assistant.intent.visual_classified, [])
        self.assertEqual(assistant.vision.frame_answers, 0)

    def test_no_fresh_frame_does_not_ask_the_model(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant.visual_router = VisualRouter({"vision": {"context_ttl_sec": 1800}})
        assistant.intent = FakeIntent(visual=self.model_choice(SCREEN))

        message = assistant.run_text("что в левом верхнем углу")

        self.assertIn("Покажи экран", message)
        self.assertEqual(assistant.intent.visual_classified, [])
        self.assertEqual(assistant.vision.frame_answers, 0)

    def test_status_reports_changes_and_window(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant.vision_watch.started_at = time.monotonic() - 10
        assistant.vision_watch.changes = 2
        assistant.vision_watch.window_title = "Chrome — GitHub"

        message = assistant.run_text("статус наблюдения")

        self.assertIn("Экран менялся 2 раза", message)
        self.assertIn("Активное окно: Chrome — GitHub", message)
        self.assertEqual(assistant.router.seen, [])

    def test_what_changed_uses_observation_when_active(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant.vision_watch.started_at = time.monotonic() - 5
        assistant.vision_watch.changes = 3

        message = assistant.run_text("что изменилось")

        self.assertIn("Экран менялся 3 раза", message)
        self.assertEqual(assistant.router.seen, [])

    def test_what_changed_without_observation_suggests_enabling(self) -> None:
        assistant = self.make()

        message = assistant.run_text("что изменилось")

        self.assertIn("включи зрение", message)
        self.assertEqual(assistant.router.seen, [])

    def test_what_changed_in_dialog_falls_through(self) -> None:
        assistant = self.make()

        self.assertIsNone(assistant._vision_watch_reply("что изменилось", in_dialog=True))

    def test_history_reports_timeline_without_model(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant.vision_watch.timeline.record(
            self.watch_event(baseline=True, title="Терминал"),
            now=time.monotonic(),
        )

        message = assistant.run_text("что происходило на экране")

        self.assertIn("наблюдение началось", message)
        self.assertIn("Терминал", message)
        self.assertEqual(assistant.router.seen, [])
        self.assertEqual(assistant.vision.captures, 0)
        self.assertEqual(assistant.vision.describe_calls, 0)

    def test_history_short_phrase_falls_through_in_dialog(self) -> None:
        assistant = self.make()

        self.assertIsNone(assistant._vision_watch_reply("что происходило", in_dialog=True))

    def test_history_explicit_phrase_works_in_dialog(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant.vision_watch.timeline.record(
            self.watch_event(baseline=True, title="Терминал"),
            now=time.monotonic(),
        )

        message = assistant._vision_watch_reply("история наблюдения", in_dialog=True)

        self.assertIsNotNone(message)
        assert message is not None
        self.assertIn("наблюдение началось", message)

    def test_history_without_observation_suggests_enabling(self) -> None:
        assistant = self.make()

        message = assistant.run_text("что происходило на экране")

        self.assertIn("включи зрение", message)
        self.assertEqual(assistant.router.seen, [])

    def test_history_with_events_disabled(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant.vision_watch.timeline = None

        message = assistant.run_text("история наблюдения")

        self.assertIn("выключена", message)

    def test_compare_frames_uses_key_frames_without_capture(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        self.add_key_frames(assistant)

        message = assistant.run_text("сравни кадры")

        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(len(assistant.vision.frame_batches), 1)
        self.assertEqual(len(assistant.vision.frame_batches[0]), 2)
        self.assertEqual(assistant.vision.describe_calls, 0)
        self.assertEqual(assistant.vision.captures, 0)
        self.assertEqual(assistant.router.seen, [])
        self.assertEqual(assistant.tts.said, [message])

    def test_compare_frames_needs_two_key_frames(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        self.add_key_frames(assistant, count=1)

        message = assistant.run_text("что изменилось с начала наблюдения")

        self.assertIn("хотя бы два ключевых кадра", message)
        self.assertEqual(assistant.vision.frame_batches, [])
        self.assertEqual(assistant.vision.captures, 0)

    def test_compare_frames_without_observation_suggests_enabling(self) -> None:
        assistant = self.make()

        message = assistant.run_text("сравни кадры")

        self.assertIn("Включи зрение", message)
        self.assertEqual(assistant.vision.frame_batches, [])
        self.assertEqual(assistant.router.seen, [])

    def test_compare_frames_in_dialog_falls_through_when_inactive(self) -> None:
        assistant = self.make()

        self.assertIsNone(assistant._vision_watch_compare("сравни кадры", in_dialog=True))

    def test_compare_frames_works_in_dialog_when_active(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        self.add_key_frames(assistant)

        message = assistant._vision_watch_compare("сравни кадры наблюдения", in_dialog=True)

        self.assertEqual(message, "На экране открыт браузер")
        self.assertEqual(len(assistant.vision.frame_batches), 1)

    def test_compare_frames_with_events_disabled(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant.vision_watch.timeline = None

        message = assistant.run_text("сравни кадры")

        self.assertIn("Ключевые кадры выключены", message)
        self.assertEqual(assistant.vision.frame_batches, [])

    def test_compare_frames_with_key_frames_disabled(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant.vision_watch.timeline.max_frames = 0

        message = assistant.run_text("сравни кадры")

        self.assertIn("Ключевые кадры выключены", message)
        self.assertEqual(assistant.vision.frame_batches, [])

    def test_compare_frames_failure_is_reported(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True
        assistant.vision.error = RuntimeError("boom")
        self.add_key_frames(assistant)

        with patch("app.main.beep"):
            message = assistant.run_text("сравни кадры")

        self.assertEqual(message, "Не удалось сравнить кадры")
        self.assertEqual(len(assistant.vision.frame_batches), 1)

    def test_compare_is_not_triggered_by_plain_change_question(self) -> None:
        assistant = self.make()
        assistant.vision_watch.active = True

        self.assertIsNone(assistant._vision_watch_compare("что изменилось", in_dialog=False))

    def test_timeout_is_announced(self) -> None:
        assistant = self.make()
        assistant.vision_watch.auto_off_sec = 60.0

        assistant._vision_watch_stopped("timeout")

        self.assertEqual(assistant.notifications, ["Наблюдение выключено: прошло 1 минуту."])
        self.assertEqual(assistant.tts.said, ["Наблюдение выключено: прошло 1 минуту."])

    def test_manual_stop_is_not_announced_twice(self) -> None:
        assistant = self.make()

        assistant._vision_watch_stopped("manual")

        self.assertEqual(assistant.notifications, [])


class VisionWatchConfigTest(unittest.TestCase):
    def test_disabled_vision_has_no_observer(self) -> None:
        assistant = Assistant.__new__(Assistant)
        self.assertIsNone(assistant._make_vision_watch({"vision": {"enabled": False}}))

    def test_disabled_watch_has_no_observer(self) -> None:
        assistant = Assistant.__new__(Assistant)
        self.assertIsNone(
            assistant._make_vision_watch({"vision": {"enabled": True, "watch": {"enabled": False}}})
        )

    def test_defaults_create_inactive_observer(self) -> None:
        assistant = Assistant.__new__(Assistant)

        observer = assistant._make_vision_watch({"vision": {"enabled": True}})

        self.assertIsNotNone(observer)
        assert observer is not None
        self.assertFalse(observer.status().active)
        self.assertEqual(observer.interval_sec, 2.0)
        self.assertEqual(observer.auto_off_sec, 600.0)

    def test_custom_timing_is_applied(self) -> None:
        assistant = Assistant.__new__(Assistant)

        observer = assistant._make_vision_watch(
            {
                "vision": {
                    "enabled": True,
                    "watch": {"interval_sec": 3.5, "auto_off_sec": 120},
                }
            }
        )

        assert observer is not None
        self.assertEqual(observer.interval_sec, 3.5)
        self.assertEqual(observer.auto_off_sec, 120.0)

    def test_events_are_enabled_by_default(self) -> None:
        assistant = Assistant.__new__(Assistant)

        observer = assistant._make_vision_watch({"vision": {"enabled": True}})

        assert observer is not None
        self.assertIsNotNone(observer.timeline)
        assert observer.timeline is not None
        self.assertEqual(observer.timeline.max_events, DEFAULT_MAX_EVENTS)
        self.assertEqual(observer.timeline.idle_after_sec, DEFAULT_IDLE_AFTER_SEC)
        self.assertEqual(observer.timeline.max_frames, DEFAULT_MAX_FRAMES)
        self.assertEqual(observer.timeline.frame_gap_sec, DEFAULT_FRAME_GAP_SEC)

    def test_events_can_be_disabled(self) -> None:
        assistant = Assistant.__new__(Assistant)

        observer = assistant._make_vision_watch(
            {
                "vision": {
                    "enabled": True,
                    "watch": {"events": {"enabled": False}},
                }
            }
        )

        assert observer is not None
        self.assertIsNone(observer.timeline)

    def test_custom_event_settings_are_applied(self) -> None:
        assistant = Assistant.__new__(Assistant)

        observer = assistant._make_vision_watch(
            {
                "vision": {
                    "enabled": True,
                    "watch": {
                        "events": {
                            "max_events": 5,
                            "idle_after_sec": 30,
                            "max_frames": 2,
                            "frame_gap_sec": 7.5,
                        }
                    },
                }
            }
        )

        assert observer is not None
        assert observer.timeline is not None
        self.assertEqual(observer.timeline.max_events, 5)
        self.assertEqual(observer.timeline.idle_after_sec, 30.0)
        self.assertEqual(observer.timeline.max_frames, 2)
        self.assertEqual(observer.timeline.frame_gap_sec, 7.5)

    def test_comments_are_available_but_inactive_by_default(self) -> None:
        assistant = Assistant.__new__(Assistant)

        comments = assistant._make_vision_comments({"vision": {"enabled": True}})

        self.assertTrue(comments.available)
        self.assertFalse(comments.active)
        self.assertTrue(comments.use_frame)
        self.assertEqual(comments.min_interval_sec, 90.0)
        self.assertEqual(comments.max_per_hour, 10)

    def test_comments_can_be_disabled_and_tuned(self) -> None:
        assistant = Assistant.__new__(Assistant)

        comments = assistant._make_vision_comments(
            {
                "vision": {
                    "enabled": True,
                    "watch": {
                        "comments": {
                            "enabled": False,
                            "min_interval_sec": 30,
                            "max_per_hour": 3,
                            "use_frame": False,
                        }
                    },
                }
            }
        )

        self.assertFalse(comments.available)
        self.assertFalse(comments.use_frame)
        self.assertEqual(comments.min_interval_sec, 30.0)
        self.assertEqual(comments.max_per_hour, 3)

    def test_zero_comments_limits_are_preserved(self) -> None:
        assistant = Assistant.__new__(Assistant)

        comments = assistant._make_vision_comments(
            {
                "vision": {
                    "enabled": True,
                    "watch": {
                        "comments": {
                            "min_interval_sec": 0,
                            "max_per_hour": 0,
                        }
                    },
                }
            }
        )

        self.assertEqual(comments.min_interval_sec, 0.0)
        self.assertEqual(comments.max_per_hour, 0)


if __name__ == "__main__":
    unittest.main()
