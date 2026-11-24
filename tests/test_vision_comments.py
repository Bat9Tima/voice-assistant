from __future__ import annotations

import threading
import unittest

from PIL import Image

from app.screen_capture import Frame
from app.vision_comments import (
    ProactiveCommenter,
    build_text_prompt,
    clean_comment,
    describe_event,
)
from app.vision_events import KIND_CHANGE, KIND_IDLE, KIND_START, KIND_WINDOW, VisionEvent


def make_event(
    kind: str = KIND_WINDOW,
    *,
    at: float = 0.0,
    title: str | None = "Chrome",
    previous: str | None = "Терминал",
    count: int = 1,
    detail: str = "",
) -> VisionEvent:
    return VisionEvent(
        kind,
        at,
        window_title=title,
        previous_title=previous,
        count=count,
        detail=detail,
    )


def make_frame() -> Frame:
    return Frame(image=Image.new("RGB", (16, 9), "white"), captured_at=0.0)


class ProactiveCommenterTest(unittest.TestCase):
    def make(
        self,
        *,
        compose=None,
        available: bool = True,
        use_frame: bool = True,
        min_interval_sec: float = 0.0,
        max_per_hour: int = 10,
        busy=None,
    ) -> tuple[ProactiveCommenter, list[str], threading.Event, list[float]]:
        comments: list[str] = []
        done = threading.Event()
        now = [0.0]

        def on_comment(text: str) -> None:
            comments.append(text)
            done.set()

        commenter = ProactiveCommenter(
            compose=compose or (lambda event, frame: "Проверь ошибку на экране"),
            on_comment=on_comment,
            is_busy=busy,
            clock=lambda: now[0],
            available=available,
            use_frame=use_frame,
            min_interval_sec=min_interval_sec,
            max_per_hour=max_per_hour,
        )
        return commenter, comments, done, now

    def test_disabled_mode_does_not_start(self) -> None:
        commenter, _, _, _ = self.make(available=False)

        self.assertFalse(commenter.start())
        self.assertFalse(commenter.enqueue(make_event()))
        self.assertFalse(commenter.active)

    def test_start_is_idempotent_and_stop_is_safe(self) -> None:
        commenter, _, _, _ = self.make()

        self.assertTrue(commenter.start())
        self.assertFalse(commenter.start())
        self.assertTrue(commenter.active)
        self.assertTrue(commenter.stop())
        self.assertFalse(commenter.active)
        self.assertFalse(commenter.stop())

    def test_worker_composes_and_reports_comment(self) -> None:
        commenter, comments, done, _ = self.make()
        commenter.start()
        try:
            self.assertTrue(commenter.enqueue(make_event(), make_frame(), now=0.0))
            self.assertTrue(done.wait(2.0), "комментарий не сработал")
        finally:
            commenter.stop()

        self.assertEqual(comments, ["Проверь ошибку на экране"])
        status = commenter.status(now=0.0)
        self.assertEqual(status.comments, 1)
        self.assertEqual(status.attempts, 1)

    def test_skip_and_empty_are_not_spoken(self) -> None:
        for answer in ("SKIP", "skip — нечего сказать", "", None):
            with self.subTest(answer=answer):
                commenter, comments, done, _ = self.make(
                    compose=lambda event, frame, answer=answer: answer
                )
                commenter.start()
                try:
                    self.assertTrue(commenter.enqueue(make_event(), now=0.0))
                    # Даём воркеру время обработать событие.
                    self.assertFalse(done.wait(0.4))
                finally:
                    commenter.stop()
                self.assertEqual(comments, [])
                self.assertEqual(commenter.status(now=0.0).attempts, 1)

    def test_min_interval_blocks_next_attempt(self) -> None:
        commenter, _, done, now = self.make(min_interval_sec=100.0)
        commenter.start()
        try:
            now[0] = 100.0
            self.assertTrue(commenter.enqueue(make_event(), now=100.0))
            self.assertTrue(done.wait(2.0))
            now[0] = 150.0
            self.assertFalse(commenter.enqueue(make_event(), now=150.0))
            status = commenter.status(now=150.0)
            self.assertEqual(status.next_allowed_at, 200.0)
        finally:
            commenter.stop()

    def test_hourly_limit_blocks_next_attempt(self) -> None:
        commenter, _, done, now = self.make(max_per_hour=1)
        commenter.start()
        try:
            now[0] = 10.0
            self.assertTrue(commenter.enqueue(make_event(), now=10.0))
            self.assertTrue(done.wait(2.0))
            now[0] = 1000.0
            self.assertFalse(commenter.enqueue(make_event(), now=1000.0))
            self.assertEqual(commenter.status(now=1000.0).attempts, 1)
        finally:
            commenter.stop()

    def test_busy_mode_skips_events(self) -> None:
        commenter, _, _, _ = self.make(busy=lambda: True)
        commenter.start()
        try:
            self.assertFalse(commenter.enqueue(make_event()))
        finally:
            commenter.stop()

    def test_pause_blocks_and_resume_allows(self) -> None:
        commenter, _, _, _ = self.make()
        commenter.start()
        try:
            self.assertTrue(commenter.pause())
            self.assertFalse(commenter.enqueue(make_event()))
            self.assertTrue(commenter.resume())
            self.assertTrue(commenter.enqueue(make_event()))
        finally:
            commenter.stop()

    def test_only_notable_events_are_queued(self) -> None:
        commenter, _, _, _ = self.make()
        commenter.start()
        try:
            self.assertFalse(commenter.enqueue(make_event(KIND_START)))
            self.assertFalse(commenter.enqueue(make_event(KIND_CHANGE)))
            self.assertTrue(commenter.enqueue(make_event(KIND_WINDOW)))
            self.assertTrue(commenter.enqueue(make_event(KIND_IDLE)))
        finally:
            commenter.stop()

    def test_frame_is_passed_only_when_enabled(self) -> None:
        seen: list[Frame | None] = []

        def compose(event: VisionEvent, frame: Frame | None) -> str:
            seen.append(frame)
            return "Комментарий"

        for use_frame in (True, False):
            with self.subTest(use_frame=use_frame):
                seen.clear()
                commenter, _, done, _ = self.make(compose=compose, use_frame=use_frame)
                commenter.start()
                try:
                    frame = make_frame()
                    self.assertTrue(commenter.enqueue(make_event(), frame, now=0.0))
                    self.assertTrue(done.wait(2.0))
                finally:
                    commenter.stop()
                self.assertEqual(len(seen), 1)
                self.assertEqual(seen[0] is frame, use_frame)

    def test_compose_error_is_recorded_without_speaking(self) -> None:
        def boom(event: VisionEvent, frame: Frame | None) -> str:
            raise RuntimeError("boom")

        commenter, comments, done, _ = self.make(compose=boom)
        commenter.start()
        try:
            self.assertTrue(commenter.enqueue(make_event(), now=0.0))
            self.assertFalse(done.wait(0.4))
            self.assertEqual(comments, [])
            self.assertEqual(commenter.status(now=0.0).last_error, "RuntimeError")
        finally:
            commenter.stop()

    def test_describe_reports_state_and_privacy(self) -> None:
        commenter, _, _, _ = self.make()
        self.assertIn("выключены", commenter.describe(now=0.0))
        commenter.start()
        try:
            text = commenter.describe(now=0.0)
        finally:
            commenter.stop()
        self.assertIn("Комментарии включены", text)
        self.assertIn("кадр", text)
        self.assertIn("проверок за час", text)


class CleanCommentTest(unittest.TestCase):
    def test_skip_is_dropped(self) -> None:
        self.assertIsNone(clean_comment("SKIP"))
        self.assertIsNone(clean_comment("skip — нечего сказать"))

    def test_empty_is_dropped(self) -> None:
        self.assertIsNone(clean_comment(None))
        self.assertIsNone(clean_comment("   "))

    def test_prefix_and_quotes_are_removed(self) -> None:
        self.assertEqual(clean_comment("Комментарий: «Проверь ошибку»"), "Проверь ошибку")

    def test_long_comment_is_shortened(self) -> None:
        text = clean_comment("слово " * 100)
        assert text is not None
        self.assertLessEqual(len(text), 300)
        self.assertTrue(text.endswith("…"))


class DescribeEventTest(unittest.TestCase):
    def test_window_event(self) -> None:
        text = describe_event(make_event(KIND_WINDOW, title="Chrome", previous="Терминал"))
        self.assertIn("Терминал", text)
        self.assertIn("Chrome", text)

    def test_idle_event(self) -> None:
        self.assertIn("2 минуты", describe_event(make_event(KIND_IDLE, detail="2 минуты")))

    def test_change_event(self) -> None:
        self.assertIn("3 раз", describe_event(make_event(KIND_CHANGE, count=3)))

    def test_start_event(self) -> None:
        self.assertIn("наблюдение", describe_event(make_event(KIND_START)))


class BuildPromptTest(unittest.TestCase):
    def test_prompt_contains_event_and_history(self) -> None:
        prompt = build_text_prompt(
            make_event(KIND_WINDOW, title="Chrome", previous="Терминал"),
            "Что происходило: 10 секунд назад — активное окно — Chrome.",
        )
        self.assertIn("Chrome", prompt)
        self.assertIn("Терминал", prompt)
        self.assertIn("10 секунд назад", prompt)
        self.assertIn("SKIP", prompt)


if __name__ == "__main__":
    unittest.main()
