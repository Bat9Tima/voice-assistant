from __future__ import annotations

import json
import unittest

from app.wake import OpenWakeWordDetector, WakeWordDetector, allowed_errors


class AllowedErrorsTest(unittest.TestCase):
    def test_short_word_must_match_exactly(self) -> None:
        self.assertEqual(allowed_errors("луна"), 0)

    def test_similar_short_word_is_not_excused(self) -> None:
        # «уна» отличается от «луна» на одну букву, но это другое слово.
        self.assertEqual(allowed_errors("уна"), 0)

    def test_long_phrase_allows_one_error(self) -> None:
        self.assertEqual(allowed_errors("окей ассистент"), 1)

    def test_long_single_word_allows_one_error(self) -> None:
        self.assertEqual(allowed_errors("ассистент"), 1)


class FakeRecognizer:
    def __init__(self, text: str) -> None:
        self.text = text

    def AcceptWaveform(self, pcm: bytes) -> bool:  # noqa: N802
        return False

    def PartialResult(self) -> str:  # noqa: N802
        return json.dumps({"partial": self.text}, ensure_ascii=False)

    def Reset(self) -> None:  # noqa: N802
        pass


class WakeMatchTest(unittest.TestCase):
    def detector(self, text: str) -> WakeWordDetector:
        detector = WakeWordDetector.__new__(WakeWordDetector)
        detector.aliases = ["луна"]
        detector.recognizer = FakeRecognizer(text)
        return detector

    def test_exact_word_triggers(self) -> None:
        self.assertTrue(self.detector("луна").feed(b"pcm"))

    def test_confusable_does_not_trigger(self) -> None:
        for text in ("уна", "она", "лупа", "руна", "дума", "лунь"):
            with self.subTest(text=text):
                self.assertFalse(self.detector(text).feed(b"pcm"))

    def test_word_inside_command_triggers(self) -> None:
        self.assertTrue(self.detector("луна открой ютуб").feed(b"pcm"))

    def test_empty_audio_is_ignored(self) -> None:
        self.assertFalse(self.detector("луна").feed(b""))


class WakeResetTest(unittest.TestCase):
    def test_reset_creates_fresh_recognizer(self) -> None:
        detector = WakeWordDetector.__new__(WakeWordDetector)
        detector.recognizer = object()
        created = object()
        detector._new_recognizer = lambda: created
        detector.reset()
        self.assertIs(detector.recognizer, created)


class FakeOpenWakeWordModel:
    def __init__(self, score: float) -> None:
        self.score = score
        self.models = {"hey_luna": object()}
        self.reset_called = False

    def predict(self, samples):
        return {"hey_luna": self.score}

    def reset(self) -> None:
        self.reset_called = True


class OpenWakeWordMatchTest(unittest.TestCase):
    def detector(self, score: float) -> OpenWakeWordDetector:
        detector = OpenWakeWordDetector.__new__(OpenWakeWordDetector)
        detector.model = FakeOpenWakeWordModel(score)
        detector.name = "hey_luna"
        detector.threshold = 0.5
        return detector

    def test_score_above_threshold_triggers(self) -> None:
        self.assertTrue(self.detector(0.8).feed(b"\x00\x00"))

    def test_score_below_threshold_is_ignored(self) -> None:
        self.assertFalse(self.detector(0.49).feed(b"\x00\x00"))

    def test_empty_audio_is_ignored(self) -> None:
        self.assertFalse(self.detector(0.9).feed(b""))

    def test_reset_clears_model_state(self) -> None:
        detector = self.detector(0.0)
        detector.reset()
        self.assertTrue(detector.model.reset_called)


if __name__ == "__main__":
    unittest.main()
