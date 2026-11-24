from __future__ import annotations

import unittest

from PIL import Image

from app.frame_processor import FrameProcessor
from app.screen_capture import Frame
from app.screen_watch import ScreenWatcher, format_event


class FakeCapture:
    def __init__(
        self,
        images: list[Image.Image],
        titles: list[str | None] | None = None,
    ) -> None:
        self.images = list(images)
        self.titles = list(titles) if titles is not None else [None] * len(self.images)
        self.calls = 0

    def grab(self) -> Frame:
        index = min(self.calls, len(self.images) - 1)
        self.calls += 1
        return Frame(
            image=self.images[index],
            captured_at=float(self.calls),
            window_title=self.titles[index],
        )


class ScreenWatcherTest(unittest.TestCase):
    def test_first_frame_is_baseline(self) -> None:
        watcher = ScreenWatcher(
            FakeCapture([Image.new("RGB", (320, 180), "white")]),
            FrameProcessor(),
        )

        event = watcher.poll()

        self.assertIsNone(event.result)
        self.assertFalse(event.changed)
        self.assertEqual((event.width, event.height), (320, 180))

    def test_second_identical_frame_is_unchanged(self) -> None:
        watcher = ScreenWatcher(
            FakeCapture(
                [
                    Image.new("RGB", (320, 180), "white"),
                    Image.new("RGB", (320, 180), "white"),
                ]
            ),
            FrameProcessor(),
        )
        watcher.poll()

        event = watcher.poll()

        self.assertIsNotNone(event.result)
        assert event.result is not None
        self.assertFalse(event.result.changed)
        self.assertFalse(event.changed)

    def test_changed_frame_is_reported(self) -> None:
        watcher = ScreenWatcher(
            FakeCapture(
                [
                    Image.new("RGB", (320, 180), "white"),
                    Image.new("RGB", (320, 180), "black"),
                ]
            ),
            FrameProcessor(),
        )
        watcher.poll()

        event = watcher.poll()

        self.assertTrue(event.changed)
        assert event.result is not None
        self.assertEqual(event.result.reason, "content")

    def test_reset_makes_next_frame_baseline(self) -> None:
        watcher = ScreenWatcher(
            FakeCapture(
                [
                    Image.new("RGB", (320, 180), "white"),
                    Image.new("RGB", (320, 180), "black"),
                ]
            ),
            FrameProcessor(),
        )
        watcher.poll()
        watcher.reset()

        event = watcher.poll()

        self.assertIsNone(event.result)
        self.assertFalse(event.changed)

    def test_window_change_with_same_image_is_reported(self) -> None:
        white = Image.new("RGB", (320, 180), "white")
        watcher = ScreenWatcher(
            FakeCapture([white, white.copy()], titles=["Терминал", "Chrome"]),
            FrameProcessor(),
        )
        watcher.poll()

        event = watcher.poll()

        self.assertTrue(event.changed)
        self.assertTrue(event.window_changed)
        assert event.result is not None
        self.assertFalse(event.result.changed)
        text = format_event(event, 2.0)
        self.assertIn("window", text)
        self.assertIn("Chrome", text)

    def test_same_window_title_is_not_reported(self) -> None:
        white = Image.new("RGB", (320, 180), "white")
        watcher = ScreenWatcher(
            FakeCapture([white, white.copy()], titles=["Chrome", "Chrome"]),
            FrameProcessor(),
        )
        watcher.poll()

        event = watcher.poll()

        self.assertFalse(event.changed)
        self.assertFalse(event.window_changed)

    def test_baseline_shows_window_title(self) -> None:
        watcher = ScreenWatcher(
            FakeCapture(
                [Image.new("RGB", (320, 180), "white")],
                titles=["Терминал"],
            ),
            FrameProcessor(),
        )

        text = format_event(watcher.poll(), 0.0)

        self.assertIn("старт", text)
        self.assertIn("Терминал", text)

    def test_last_frame_is_kept_until_reset(self) -> None:
        watcher = ScreenWatcher(
            FakeCapture([Image.new("RGB", (320, 180), "white")]),
            FrameProcessor(),
        )
        self.assertIsNone(watcher.last_frame)

        watcher.poll()

        self.assertIsNotNone(watcher.last_frame)
        watcher.reset()
        self.assertIsNone(watcher.last_frame)

    def test_format_baseline_and_change(self) -> None:
        watcher = ScreenWatcher(
            FakeCapture(
                [
                    Image.new("RGB", (320, 180), "white"),
                    Image.new("RGB", (320, 180), "black"),
                ]
            ),
            FrameProcessor(),
        )

        baseline = format_event(watcher.poll(), 0.0)
        changed = format_event(watcher.poll(), 2.0)

        self.assertIn("старт", baseline)
        self.assertIn("320×180", baseline)
        self.assertIn("изменение", changed)
        self.assertIn("content", changed)


if __name__ == "__main__":
    unittest.main()
