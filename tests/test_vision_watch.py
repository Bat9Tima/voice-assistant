from __future__ import annotations

import threading
import time
import unittest

from PIL import Image

from app.frame_processor import ChangeResult, FrameProcessor
from app.screen_capture import Frame, ScreenCaptureError
from app.screen_watch import ScreenWatcher, WatchEvent
from app.vision_watch import (
    MIN_INTERVAL_SEC,
    VisionObserver,
    WatchStatus,
    describe_status,
    format_duration,
)


def make_event(
    *,
    at: float = 0.0,
    changed: bool = False,
    title: str | None = None,
    baseline: bool = False,
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
        at=at,
        changed=changed,
        window_changed=False,
        result=result,
        width=320,
        height=180,
        window_title=title,
    )


class FakeWatcher:
    def __init__(self, events: list[WatchEvent]) -> None:
        self.events = list(events)
        self.polls = 0
        self.resets = 0
        self.error: Exception | None = None
        self.frame: Frame | None = None

    def poll(self) -> WatchEvent:
        if self.error is not None:
            raise self.error
        index = min(self.polls, len(self.events) - 1)
        self.polls += 1
        self.frame = Frame(
            image=Image.new("RGB", (32, 18), "white"),
            captured_at=float(self.polls),
        )
        return self.events[index]

    def reset(self) -> None:
        self.resets += 1
        self.frame = None

    @property
    def last_frame(self) -> Frame | None:
        return self.frame


class VisionObserverTest(unittest.TestCase):
    def make(
        self,
        events: list[WatchEvent] | None = None,
        **kwargs: float,
    ) -> tuple[VisionObserver, FakeWatcher]:
        watcher = FakeWatcher(events or [make_event(baseline=True)])
        kwargs.setdefault("interval_sec", 2.0)
        observer = VisionObserver(
            watcher,
            clock=lambda: 100.0,
            **kwargs,
        )
        return observer, watcher

    def test_inactive_tick_does_nothing(self) -> None:
        observer, watcher = self.make()

        self.assertIsNone(observer.tick(100.0))
        self.assertEqual(watcher.polls, 0)
        self.assertFalse(observer.status().active)

    def test_tick_saves_baseline_and_respects_interval(self) -> None:
        events = [
            make_event(baseline=True, title="Терминал"),
            make_event(changed=True, title="Chrome"),
        ]
        observer, watcher = self.make(events)

        self.assertTrue(observer.start(background=False))
        first = observer.tick(100.0)
        early = observer.tick(101.0)
        second = observer.tick(102.0)

        self.assertIsNotNone(first)
        assert first is not None
        self.assertFalse(first.changed)
        self.assertIsNone(early)
        assert second is not None
        self.assertTrue(second.changed)
        self.assertEqual(watcher.polls, 2)
        status = observer.status()
        self.assertEqual(status.polls, 2)
        self.assertEqual(status.changes, 1)
        self.assertEqual(status.last_change_at, 102.0)
        self.assertEqual(status.window_title, "Chrome")
        self.assertIsNotNone(observer.last_frame)

    def test_second_start_returns_false(self) -> None:
        observer, _ = self.make()

        self.assertTrue(observer.start(background=False))
        self.assertFalse(observer.start(background=False))

    def test_auto_off_stops_and_reports_timeout(self) -> None:
        reasons: list[str] = []
        observer, watcher = self.make(
            auto_off_sec=10.0,
            on_stop=reasons.append,
        )
        observer.start(background=False)

        self.assertIsNotNone(observer.tick(105.0))
        self.assertIsNone(observer.tick(110.0))

        self.assertFalse(observer.status().active)
        self.assertEqual(reasons, ["timeout"])
        self.assertIsNone(observer.last_frame)
        self.assertGreaterEqual(watcher.resets, 2)

    def test_auto_off_works_while_paused(self) -> None:
        reasons: list[str] = []
        observer, _ = self.make(auto_off_sec=10.0, on_stop=reasons.append)
        observer.start(background=False)
        observer.pause()

        self.assertIsNone(observer.tick(110.0))

        self.assertFalse(observer.status().active)
        self.assertEqual(reasons, ["timeout"])

    def test_zero_auto_off_disables_timer(self) -> None:
        observer, _ = self.make(auto_off_sec=0.0)
        observer.start(background=False)

        self.assertIsNotNone(observer.tick(10_000.0))
        self.assertTrue(observer.status().active)

    def test_pause_skips_capture_and_resume_starts_fresh(self) -> None:
        events = [make_event(baseline=True), make_event(changed=True)]
        observer, watcher = self.make(events)
        observer.start(background=False)
        observer.tick(100.0)

        self.assertTrue(observer.pause())
        self.assertIsNone(observer.tick(102.0))
        self.assertTrue(observer.status().paused)
        self.assertIsNone(observer.last_frame)
        self.assertEqual(watcher.polls, 1)

        self.assertTrue(observer.resume())
        self.assertIsNotNone(observer.tick(102.0))
        self.assertEqual(watcher.polls, 2)
        self.assertFalse(observer.status().paused)

    def test_capture_error_is_recorded_and_cleared(self) -> None:
        observer, watcher = self.make()
        observer.start(background=False)
        watcher.error = ScreenCaptureError("нет доступа")

        self.assertIsNone(observer.tick(100.0))
        status = observer.status()
        self.assertTrue(status.active)
        self.assertIn("нет доступа", status.error or "")

        watcher.error = None
        self.assertIsNotNone(observer.tick(102.0))
        self.assertIsNone(observer.status().error)

    def test_stop_is_idempotent_and_drops_frames(self) -> None:
        observer, _ = self.make()
        observer.start(background=False)
        observer.tick(100.0)

        self.assertTrue(observer.stop())
        self.assertFalse(observer.status().active)
        self.assertIsNone(observer.last_frame)
        self.assertFalse(observer.stop())

    def test_interval_is_not_below_minimum(self) -> None:
        observer, _ = self.make(interval_sec=0.001)
        self.assertEqual(observer.interval_sec, MIN_INTERVAL_SEC)


class VisionObserverThreadTest(unittest.TestCase):
    def test_thread_captures_until_stopped(self) -> None:
        grabbed = threading.Event()

        class Capture:
            def grab(self) -> Frame:
                grabbed.set()
                return Frame(
                    image=Image.new("RGB", (64, 36), "white"),
                    captured_at=time.monotonic(),
                    window_title="Терминал",
                )

        observer = VisionObserver(
            ScreenWatcher(Capture(), FrameProcessor()),
            interval_sec=MIN_INTERVAL_SEC,
            auto_off_sec=0.0,
        )
        observer.start()
        self.assertTrue(grabbed.wait(2.0), "поток наблюдения не снял кадр")
        observer.stop()

        self.assertFalse(observer.active)
        self.assertIsNone(observer.last_frame)


class FormatDurationTest(unittest.TestCase):
    def test_plural_seconds(self) -> None:
        self.assertEqual(format_duration(0), "0 секунд")
        self.assertEqual(format_duration(1), "1 секунду")
        self.assertEqual(format_duration(2), "2 секунды")
        self.assertEqual(format_duration(5), "5 секунд")
        self.assertEqual(format_duration(11), "11 секунд")
        self.assertEqual(format_duration(21), "21 секунду")

    def test_minutes(self) -> None:
        self.assertEqual(format_duration(60), "1 минуту")
        self.assertEqual(format_duration(61), "1 минуту 1 секунду")
        self.assertEqual(format_duration(125), "2 минуты 5 секунд")
        self.assertEqual(format_duration(121), "2 минуты 1 секунду")


class DescribeStatusTest(unittest.TestCase):
    def status(self, **kwargs: object) -> WatchStatus:
        base: dict[str, object] = {
            "active": True,
            "paused": False,
            "started_at": 100.0,
            "last_poll_at": 100.0,
            "last_change_at": None,
            "polls": 1,
            "changes": 0,
            "window_title": None,
            "error": None,
            "interval_sec": 2.0,
            "auto_off_sec": 600.0,
        }
        base.update(kwargs)
        return WatchStatus(**base)  # type: ignore[arg-type]

    def test_inactive(self) -> None:
        self.assertEqual(
            describe_status(self.status(active=False), now=110.0),
            "Наблюдение выключено.",
        )

    def test_paused(self) -> None:
        self.assertEqual(
            describe_status(self.status(paused=True), now=110.0),
            "Наблюдение на паузе.",
        )

    def test_baseline(self) -> None:
        text = describe_status(self.status(), now=110.0)
        self.assertIn("работает 10 секунд", text)
        self.assertIn("ещё не сравнивался", text)

    def test_changes_and_window(self) -> None:
        text = describe_status(
            self.status(
                polls=4,
                changes=2,
                last_change_at=90.0,
                window_title="Chrome — GitHub",
            ),
            now=100.0,
        )
        self.assertIn("менялся 2 раза", text)
        self.assertIn("Последнее изменение 10 секунд назад", text)
        self.assertIn("Активное окно: Chrome — GitHub", text)

    def test_no_changes(self) -> None:
        text = describe_status(self.status(polls=3), now=130.0)
        self.assertIn("пока не менялся", text)

    def test_long_title_is_shortened(self) -> None:
        text = describe_status(
            self.status(window_title="Очень длинный заголовок " * 5),
            now=100.0,
        )
        self.assertIn("…", text)

    def test_error_is_reported(self) -> None:
        text = describe_status(self.status(error="нет доступа"), now=100.0)
        self.assertIn("Последняя ошибка захвата: нет доступа", text)


if __name__ == "__main__":
    unittest.main()
