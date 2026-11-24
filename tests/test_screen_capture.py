from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from PIL import Image

from app.screen_capture import Frame, ScreenCapture, ScreenCaptureError


class ScreenCaptureTest(unittest.TestCase):
    def test_grab_resizes_to_max_width(self) -> None:
        source = Image.new("RGB", (2000, 1000), "white")
        with patch("app.screen_capture.ImageGrab.grab", return_value=source) as grab:
            frame = ScreenCapture(max_width=1000).grab()

        grab.assert_called_once_with()
        self.assertEqual(frame.image.size, (1000, 500))
        self.assertEqual((frame.width, frame.height), (1000, 500))
        self.assertGreater(frame.captured_at, 0.0)
        self.assertIsNone(frame.window_title)

    def test_grab_keeps_native_size_without_limit(self) -> None:
        source = Image.new("RGB", (640, 480), "white")
        with patch("app.screen_capture.ImageGrab.grab", return_value=source):
            frame = ScreenCapture(max_width=0).grab()

        self.assertEqual((frame.width, frame.height), (640, 480))

    def test_grab_converts_to_rgb(self) -> None:
        source = Image.new("RGBA", (40, 20), (255, 0, 0, 128))
        with patch("app.screen_capture.ImageGrab.grab", return_value=source):
            frame = ScreenCapture(max_width=0).grab()

        self.assertEqual(frame.image.mode, "RGB")

    def test_grab_wraps_capture_error(self) -> None:
        with patch("app.screen_capture.ImageGrab.grab", side_effect=OSError("нет доступа")):
            with self.assertRaises(ScreenCaptureError) as ctx:
                ScreenCapture().grab()

        self.assertIn("нет доступа", str(ctx.exception))

    def test_frame_is_frozen(self) -> None:
        frame = Frame(image=Image.new("RGB", (10, 10)), captured_at=1.0)
        with self.assertRaises(FrozenInstanceError):
            frame.captured_at = 2.0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
