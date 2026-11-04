from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from app.vad import EndpointTracker, SileroVad, ensure_model


class FakeModel:
    def __init__(self, window: int = 4, answers: list[bool] | None = None) -> None:
        self._window = window
        self.answers = list(answers or [])
        self.calls: list[list[float]] = []
        self.resets = 0

    def window_size(self) -> int:
        return self._window

    def is_speech(self, samples) -> bool:
        self.calls.append(np.asarray(samples, dtype=np.float32).tolist())
        return self.answers.pop(0) if self.answers else False

    def reset(self) -> None:
        self.resets += 1


def pcm(samples: int, value: int = 1000) -> bytes:
    return np.full(samples, value, dtype=np.int16).tobytes()


class SileroVadTest(unittest.TestCase):
    def test_feed_waits_for_full_window(self) -> None:
        model = FakeModel(window=4)
        vad = SileroVad(model)

        self.assertFalse(vad.feed(pcm(3)))
        self.assertEqual(model.calls, [])

        self.assertFalse(vad.feed(pcm(1)))
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(model.calls[0]), 4)

    def test_feed_keeps_remainder_between_calls(self) -> None:
        model = FakeModel(window=4)
        vad = SileroVad(model)

        self.assertFalse(vad.feed(pcm(10)))
        self.assertEqual(len(model.calls), 2)

        self.assertFalse(vad.feed(pcm(2)))
        self.assertEqual(len(model.calls), 3)

    def test_feed_returns_true_if_any_window_has_speech(self) -> None:
        model = FakeModel(window=4, answers=[False, True])
        vad = SileroVad(model)

        self.assertTrue(vad.feed(pcm(8)))

    def test_feed_normalizes_int16(self) -> None:
        model = FakeModel(window=2)
        vad = SileroVad(model)

        vad.feed(pcm(2, 16384))
        self.assertAlmostEqual(model.calls[0][0], 0.5, places=3)

    def test_empty_feed_is_ignored(self) -> None:
        model = FakeModel(window=4)
        vad = SileroVad(model)

        self.assertFalse(vad.feed(b""))
        self.assertEqual(model.calls, [])

    def test_reset_clears_remainder_and_model(self) -> None:
        model = FakeModel(window=4)
        vad = SileroVad(model)

        vad.feed(pcm(3))
        vad.reset()
        self.assertEqual(model.resets, 1)
        self.assertFalse(vad.feed(pcm(1)))
        self.assertEqual(model.calls, [])

    def test_missing_model_returns_none(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing_vad.onnx"
            with self.assertLogs("app.vad", level="WARNING"):
                self.assertIsNone(ensure_model(missing))
                self.assertIsNone(SileroVad.create(model=missing))


class EndpointTrackerTest(unittest.TestCase):
    def make(self, silence_ms: float = 800, min_speech_ms: float = 250) -> EndpointTracker:
        return EndpointTracker(silence_ms=silence_ms, min_speech_ms=min_speech_ms)

    def test_loud_start_and_silence_end(self) -> None:
        tracker = self.make()
        tracker.update(250, vad_speech=False, loud=True)

        self.assertTrue(tracker.started)
        self.assertTrue(tracker.has_speech)

        for _ in range(3):
            tracker.update(250, vad_speech=False, loud=False)

        self.assertAlmostEqual(tracker.silence_ms, 750)
        tracker.update(50, vad_speech=False, loud=False)
        self.assertAlmostEqual(tracker.silence_ms, 800)

    def test_short_noise_is_not_speech(self) -> None:
        tracker = self.make()
        tracker.update(100, vad_speech=False, loud=True)
        tracker.update(800, vad_speech=False, loud=False)

        self.assertTrue(tracker.started)
        self.assertFalse(tracker.has_speech)

    def test_vad_can_start_quiet_speech(self) -> None:
        tracker = self.make()
        tracker.update(300, vad_speech=True, loud=False)

        self.assertTrue(tracker.vad_active)
        self.assertTrue(tracker.started)
        self.assertTrue(tracker.has_speech)
        self.assertAlmostEqual(tracker.vad_ms, 300)

    def test_vad_overrides_loud_background(self) -> None:
        tracker = self.make()
        tracker.update(250, vad_speech=True, loud=False)
        tracker.update(1000, vad_speech=False, loud=True)

        self.assertEqual(tracker.speech_ms, 250)
        self.assertEqual(tracker.silence_ms, 1000)

    def test_loud_fallback_until_vad_hears_speech(self) -> None:
        tracker = self.make()
        tracker.update(500, vad_speech=False, loud=True)
        self.assertFalse(tracker.vad_active)
        self.assertAlmostEqual(tracker.speech_ms, 500)

        tracker.update(250, vad_speech=True, loud=False)
        self.assertTrue(tracker.vad_active)
        self.assertAlmostEqual(tracker.speech_ms, 750)

    def test_silence_accumulates_only_after_start(self) -> None:
        tracker = self.make()
        tracker.update(500, vad_speech=False, loud=False)
        self.assertFalse(tracker.started)
        self.assertEqual(tracker.silence_ms, 0)

    def test_any_speech_resets_silence(self) -> None:
        tracker = self.make()
        tracker.update(250, vad_speech=True, loud=False)
        tracker.update(300, vad_speech=False, loud=False)
        self.assertAlmostEqual(tracker.silence_ms, 300)

        tracker.update(100, vad_speech=True, loud=False)
        self.assertEqual(tracker.silence_ms, 0)


if __name__ == "__main__":
    unittest.main()
