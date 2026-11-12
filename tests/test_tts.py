from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.tts import (
    PhraseBank,
    TTSService,
    clean_for_speech,
    normalize_pcm,
    split_sentences,
)


def make_pcm(
    seconds: float = 0.5,
    lead: float = 0.2,
    tail: float = 0.2,
    amplitude: float = 0.1,
) -> bytes:
    rate = 44100
    silence_head = np.zeros(int(lead * rate), dtype="<i2")
    time = np.arange(int(seconds * rate)) / rate
    tone = (amplitude * np.sin(2 * np.pi * 220 * time) * 32767).astype("<i2")
    silence_tail = np.zeros(int(tail * rate), dtype="<i2")
    return np.concatenate([silence_head, tone, silence_tail]).tobytes()


def make_service(directory: Path, **overrides) -> TTSService:
    section = {
        "enabled": True,
        "api_key": "test-key",
        "voice": "test-voice",
        "cache_dir": str(directory / "cache"),
        "phrases_dir": str(directory),
        "workers": 2,
    }
    section.update(overrides)
    return TTSService({"tts": section})


class CleanForSpeechTest(unittest.TestCase):
    def test_removes_citations_markdown_and_sources(self) -> None:
        text = (
            "**Курс вырос** на 2% [1]. Подробности [2].\n\n"
            "Источники\nКурс — РБК https://rbc.ru/news"
        )
        self.assertEqual(clean_for_speech(text), "Курс вырос на 2%. Подробности.")

    def test_keeps_plain_text(self) -> None:
        text = "За ночь над Россией сбили 71 беспилотник."
        self.assertEqual(clean_for_speech(text), text)

    def test_empty(self) -> None:
        self.assertEqual(clean_for_speech(""), "")


class SplitSentencesTest(unittest.TestCase):
    def test_splits_sentences(self) -> None:
        text = "Первое предложение здесь. Второе предложение тоже здесь."
        self.assertEqual(
            split_sentences(text),
            ["Первое предложение здесь.", "Второе предложение тоже здесь."],
        )

    def test_merges_short_tail(self) -> None:
        self.assertEqual(split_sentences("Курс вырос. Да."), ["Курс вырос. Да."])

    def test_splits_long_sentence_by_words(self) -> None:
        text = " ".join(["слово"] * 60)
        chunks = split_sentences(text, limit=100)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))

    def test_empty(self) -> None:
        self.assertEqual(split_sentences(""), [])


class NormalizePcmTest(unittest.TestCase):
    def test_trims_silence_and_levels(self) -> None:
        data = normalize_pcm(make_pcm(seconds=0.5, lead=0.3, tail=0.4))
        self.assertIsNotNone(data)
        assert data is not None
        duration = len(data) / 44100
        self.assertGreater(duration, 0.52)
        self.assertLess(duration, 0.64)
        level = 20 * np.log10(float(np.sqrt(np.mean(data**2))))
        self.assertAlmostEqual(level, -20.0, delta=1.0)

    def test_rejects_silence(self) -> None:
        self.assertIsNone(normalize_pcm(b"\x00\x00" * 1000))

    def test_rejects_odd_bytes(self) -> None:
        self.assertIsNone(normalize_pcm(b"\x01\x02\x03"))


class PhraseBankTest(unittest.TestCase):
    def _bank(self, directory: Path) -> PhraseBank:
        for index in range(1, 4):
            (directory / f"done_1_{index}.wav").write_bytes(b"x")
            (directory / f"done_2_{index}.wav").write_bytes(b"x")
        manifest = {
            "phrases": [
                {
                    "id": "done_1",
                    "category": "done",
                    "takes": [
                        {"file": f"done_1_{index}.wav"} for index in range(1, 4)
                    ],
                },
                {
                    "id": "done_2",
                    "category": "done",
                    "takes": [
                        {"file": f"done_2_{index}.wav"} for index in range(1, 4)
                    ],
                },
            ]
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return PhraseBank(directory)

    def test_never_repeats_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank = self._bank(Path(tmp))
            picks = [bank.pick("done") for _ in range(20)]
            self.assertTrue(all(path is not None for path in picks))
            for previous, current in zip(picks, picks[1:], strict=False):
                self.assertNotEqual(previous, current)

    def test_unknown_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank = self._bank(Path(tmp))
            self.assertIsNone(bank.pick("unknown"))

    def test_pick_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank = self._bank(Path(tmp))
            path = bank.pick_id("done_2")
            self.assertIsNotNone(path)
            assert path is not None
            self.assertTrue(path.name.startswith("done_2_"))


class TTSServiceTest(unittest.TestCase):
    def test_disabled_service_is_silent(self) -> None:
        service = TTSService({"tts": {"enabled": False}})
        self.assertFalse(service.phrase("done"))
        self.assertFalse(service.say("Привет."))

    def test_missing_key_is_not_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp), api_key="")
            self.assertFalse(service.available)
            self.assertFalse(service.say("Привет."))

    def test_say_caches_sentences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            played: list[int] = []
            text = "Первое предложение здесь. Второе предложение тоже здесь."
            with patch.object(
                service, "_request", return_value=make_pcm()
            ) as request, patch.object(
                service,
                "_play",
                side_effect=lambda data: played.append(len(data)) or True,
            ):
                self.assertTrue(service.say(text))
                self.assertTrue(service.say(text))
            self.assertEqual(request.call_count, 2)
            self.assertEqual(len(played), 4)

    def test_cache_key_depends_on_voice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = make_service(Path(tmp), voice="one")
            second = make_service(Path(tmp), voice="two")
            self.assertNotEqual(
                first._cache_path("текст"), second._cache_path("текст")
            )

    def test_say_truncates_long_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp), max_chars=40)
            with patch.object(
                service, "_request", return_value=make_pcm()
            ) as request, patch.object(service, "_play", return_value=True):
                service.say("слово " * 100)
            sent = request.call_args.args[0]
            self.assertLessEqual(len(sent), 41)


if __name__ == "__main__":
    unittest.main()
