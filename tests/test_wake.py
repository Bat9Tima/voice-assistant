from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

from app.wake import (
    DEFAULT_STOP_WORDS,
    BargeInDetector,
    BargeInMatch,
    OpenWakeWordDetector,
    WakeWordDetector,
    allowed_errors,
    condition_pcm,
)


class AllowedErrorsTest(unittest.TestCase):
    def test_short_word_must_match_exactly(self) -> None:
        self.assertEqual(allowed_errors("луна"), 0)

    def test_similar_short_word_is_not_excused(self) -> None:
        # «уна» отличается от «луна» на одну букву, но это другое слово.
        self.assertEqual(allowed_errors("уна"), 0)

    def test_long_phrase_allows_one_error(self) -> None:
        self.assertEqual(allowed_errors("окей ассистент"), 1)

    def test_default_wake_word_allows_one_error(self) -> None:
        # Vosk слышит «хей луна» как «хэ луна»: одна опечатка обязательна.
        self.assertEqual(allowed_errors("хей луна"), 1)
        self.assertEqual(allowed_errors("эй луна"), 0)

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


class ConditionPcmTest(unittest.TestCase):
    def test_quiet_echo_becomes_silence(self) -> None:
        pcm = np.full(400, int(0.003 * 32767), dtype="<i2").tobytes()
        self.assertEqual(condition_pcm(pcm, 0.01, 0.08), b"\x00" * len(pcm))

    def test_quiet_speech_is_amplified(self) -> None:
        pcm = np.full(400, int(0.02 * 32767), dtype="<i2").tobytes()
        prepared = condition_pcm(pcm, 0.01, 0.08)
        samples = np.frombuffer(prepared, dtype="<i2").astype(np.float32) / 32768.0
        level = float(np.sqrt(np.mean(samples**2)))
        self.assertGreater(level, 0.02)
        self.assertLessEqual(level, 0.09)

    def test_loud_audio_is_unchanged(self) -> None:
        pcm = np.full(400, int(0.2 * 32767), dtype="<i2").tobytes()
        self.assertEqual(condition_pcm(pcm, 0.01, 0.08), pcm)

    def test_empty_and_odd_bytes(self) -> None:
        self.assertEqual(condition_pcm(b"", 0.01, 0.08), b"")
        self.assertEqual(condition_pcm(b"\x01", 0.01, 0.08), b"")


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

    def test_alias_is_saved_for_caller(self) -> None:
        detector = self.detector("луна открой ютуб")

        self.assertTrue(detector.feed(b"pcm"))

        self.assertEqual(detector.last_match, "луна")


class WakeResetTest(unittest.TestCase):
    def test_reset_creates_fresh_recognizer(self) -> None:
        detector = WakeWordDetector.__new__(WakeWordDetector)
        detector.recognizer = object()
        created = object()
        detector._new_recognizer = lambda: created
        detector.reset()
        self.assertIs(detector.recognizer, created)


class WakeWarmupTest(unittest.TestCase):
    def test_warm_up_feeds_one_second_of_silence(self) -> None:
        detector = WakeWordDetector.__new__(WakeWordDetector)
        detector.sample_rate = 16000
        fed: list[bytes] = []

        class Recorder:
            def AcceptWaveform(self, pcm: bytes) -> bool:  # noqa: N802
                fed.append(pcm)
                return False

        detector.recognizer = Recorder()
        created = object()
        detector._new_recognizer = lambda: created
        detector._warm_up()
        self.assertEqual(fed, [b"\x00\x00" * 16000])
        self.assertIs(detector.recognizer, created)


class BargeInDetectorInitTest(unittest.TestCase):
    def test_defaults_and_fresh_recognizer(self) -> None:
        model = object()
        with patch("app.wake.vosk.KaldiRecognizer", side_effect=[Mock(), Mock()]) as factory:
            detector = BargeInDetector(model, ["Хей Луна"], min_level=0.005)
            self.assertEqual(factory.call_args.args, (model, 16000))
            self.assertEqual(detector.aliases, ["хей луна"])
            self.assertEqual(tuple(detector.stop_words), DEFAULT_STOP_WORDS)
            self.assertEqual(detector.min_level, 0.005)
            first = detector.recognizer
            detector.reset()
        self.assertIsNot(detector.recognizer, first)
        self.assertEqual(factory.call_count, 2)


class BargeInMatchTest(unittest.TestCase):
    def detector(self, text: str) -> BargeInDetector:
        detector = BargeInDetector.__new__(BargeInDetector)
        detector.aliases = ["хей луна", "эй луна"]
        detector.aliases_folded = ["хеи луна", "еи луна"]
        detector.stop_words = ["стоп", "хватит"]
        detector.min_level = 0.01
        detector.recognizer = FakeRecognizer(text)
        return detector

    def test_stop_word(self) -> None:
        self.assertEqual(self.detector("стоп").feed(b"pcm"), BargeInMatch("stop", "стоп"))

    def test_stop_word_at_end_of_phrase(self) -> None:
        for text in ("ну всё стоп", "стоп пожалуйста", "хватит уже"):
            with self.subTest(text=text):
                match = self.detector(text).feed(b"pcm")
                self.assertIsNotNone(match)
                assert match is not None
                self.assertEqual(match.action, "stop")

    def test_stop_word_inside_sentence_is_ignored(self) -> None:
        self.assertIsNone(self.detector("стоп это не команда").feed(b"pcm"))

    def test_stop_word_is_not_confused_with_plain_speech(self) -> None:
        text = "москва сейчас плюс три градуса цельсия пасмурно ветер три метра в секунду"
        self.assertIsNone(self.detector(text).feed(b"pcm"))

    def test_wake_word(self) -> None:
        self.assertEqual(
            self.detector("хей луна").feed(b"pcm"),
            BargeInMatch("wake", "хей луна"),
        )

    def test_wake_word_with_command(self) -> None:
        self.assertEqual(
            self.detector("эй луна открой ютуб").feed(b"pcm"),
            BargeInMatch("wake", "эй луна"),
        )

    def test_wake_word_after_echo_tail(self) -> None:
        self.assertEqual(
            self.detector("сейчас плюс три хей луна").feed(b"pcm"),
            BargeInMatch("wake", "хей луна"),
        )

    def test_wake_word_with_vosk_spelling(self) -> None:
        # Vosk стабильно пишет «хей луна» как «хэ луна».
        for text in ("хэ луна", "хэ луна новости ирака", "хэй луна"):
            with self.subTest(text=text):
                self.assertEqual(
                    self.detector(text).feed(b"pcm"),
                    BargeInMatch("wake", "хей луна"),
                )

    def test_wake_word_glued_by_vosk(self) -> None:
        for text in ("хейлуна новости", "хэлуна новости ирака"):
            with self.subTest(text=text):
                self.assertEqual(
                    self.detector(text).feed(b"pcm"),
                    BargeInMatch("wake", "хей луна"),
                )

    def test_other_speech_is_ignored(self) -> None:
        for text in ("какая погода", "луна", "хей", "и луна", "unk"):
            with self.subTest(text=text):
                self.assertIsNone(self.detector(text).feed(b"pcm"))

    def test_empty_audio_is_ignored(self) -> None:
        self.assertIsNone(self.detector("стоп").feed(b""))


class GateRecognizer:
    def __init__(self) -> None:
        self.level = 0.0

    def AcceptWaveform(self, pcm: bytes) -> bool:  # noqa: N802
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self.level = float(np.sqrt(np.mean(samples**2)))
        return False

    def PartialResult(self) -> str:  # noqa: N802
        if self.level <= 0.0:
            return json.dumps({"partial": ""})
        return json.dumps({"partial": "стоп"})


class BargeInGateTest(unittest.TestCase):
    def detector(self, recognizer: GateRecognizer) -> BargeInDetector:
        detector = BargeInDetector.__new__(BargeInDetector)
        detector.aliases = ["хей луна"]
        detector.aliases_folded = ["хеи луна"]
        detector.stop_words = ["стоп"]
        detector.min_level = 0.01
        detector.recognizer = recognizer
        return detector

    def test_quiet_echo_does_not_trigger(self) -> None:
        detector = self.detector(GateRecognizer())
        pcm = np.full(400, int(0.003 * 32767), dtype="<i2").tobytes()
        self.assertIsNone(detector.feed(pcm))

    def test_quiet_speech_triggers_after_amplification(self) -> None:
        detector = self.detector(GateRecognizer())
        pcm = np.full(400, int(0.02 * 32767), dtype="<i2").tobytes()
        self.assertEqual(detector.feed(pcm), BargeInMatch("stop", "стоп"))


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
