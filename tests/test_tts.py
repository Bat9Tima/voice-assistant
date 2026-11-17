from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.tts import (
    PhraseBank,
    TTSService,
    _SpeechJob,
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


class FakeStream:
    """Подмена sounddevice.OutputStream: без звука, но с журналом вызовов."""

    def __init__(self, on_write=None) -> None:  # noqa: ANN001
        self.on_write = on_write
        self.writes: list[int] = []
        self.started = False
        self.aborted = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def write(self, data) -> None:  # noqa: ANN001
        self.writes.append(len(data))
        if self.on_write is not None:
            self.on_write()

    def abort(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.aborted = True

    def stop(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.stopped = True

    def close(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.closed = True


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

    def test_short_sentence_with_comma_stays_whole(self) -> None:
        text = "Небо кажется голубым потому, что свет рассеивается."
        self.assertEqual(split_sentences(text), [text])

    def test_long_sentence_prefers_comma(self) -> None:
        text = (
            "Первая часть предложения, которая уже длинная, "
            "вторая часть предложения, которая тоже длинная, "
            "третья часть предложения, которая завершает мысль."
        )
        chunks = split_sentences(text, limit=60)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(chunks[0].endswith(","))
        self.assertLessEqual(len(chunks[0]), 60)
        self.assertTrue(chunks[1].startswith("вторая"))

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
                    "takes": [{"file": f"done_1_{index}.wav"} for index in range(1, 4)],
                },
                {
                    "id": "done_2",
                    "category": "done",
                    "takes": [{"file": f"done_2_{index}.wav"} for index in range(1, 4)],
                },
            ]
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
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
            with (
                patch.object(service, "_request", return_value=make_pcm()) as request,
                patch.object(
                    service,
                    "_play",
                    side_effect=lambda data, job=None: played.append(len(data)) or True,
                ),
            ):
                self.assertTrue(service.say(text))
                self.assertTrue(service.say(text))
                self.assertTrue(service.wait(5))
            self.assertEqual(request.call_count, 2)
            self.assertEqual(len(played), 4)

    def test_say_does_not_block_on_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            started = threading.Event()
            release = threading.Event()

            def slow_request(text: str) -> bytes:
                started.set()
                release.wait(timeout=5)
                return make_pcm()

            with (
                patch.object(service, "_request", side_effect=slow_request),
                patch.object(service, "_play", return_value=True),
            ):
                try:
                    begin = time.monotonic()
                    self.assertTrue(service.say("Привет."))
                    elapsed = time.monotonic() - begin
                    self.assertLess(elapsed, 1.0)
                    self.assertTrue(started.wait(2))
                finally:
                    release.set()
                self.assertTrue(service.wait(5))

    def test_stop_interrupts_speech(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            playing = threading.Event()

            def blocking_play(data, job=None) -> bool:
                playing.set()
                if job is None:
                    return True
                return not job.cancelled.wait(timeout=2)

            with (
                patch.object(service, "_request", return_value=make_pcm()),
                patch.object(service, "_play", side_effect=blocking_play),
            ):
                self.assertTrue(service.say("Первое предложение здесь."))
                self.assertTrue(playing.wait(2))
                self.assertTrue(service.speaking)
                service.stop()
                self.assertTrue(service.wait(2))
                self.assertFalse(service.speaking)

    def test_stop_drops_queued_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            played: list[str] = []
            release = threading.Event()

            def blocking_play(data, job=None) -> bool:
                played.append("play")
                if len(played) == 1:
                    release.wait(timeout=2)
                return True

            with (
                patch.object(service, "_request", return_value=make_pcm()),
                patch.object(service, "_play", side_effect=blocking_play),
            ):
                self.assertTrue(service.say("Первое предложение здесь."))
                self.assertTrue(service.say("Второе предложение здесь."))
                # Даём воркеру дойти до первого фрагмента и встать в _play.
                for _ in range(100):
                    if played:
                        break
                    time.sleep(0.01)
                service.stop()
                release.set()
                self.assertTrue(service.wait(2))
            self.assertEqual(played, ["play"])

    def test_cache_key_depends_on_voice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = make_service(Path(tmp), voice="one")
            second = make_service(Path(tmp), voice="two")
            self.assertNotEqual(first._cache_path("текст"), second._cache_path("текст"))

    def test_say_truncates_long_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp), max_chars=40)
            with (
                patch.object(service, "_request", return_value=make_pcm()) as request,
                patch.object(service, "_play", return_value=True),
            ):
                service.say("слово " * 100)
                self.assertTrue(service.wait(5))
            sent = request.call_args.args[0]
            self.assertLessEqual(len(sent), 41)

    def test_cancelled_job_is_not_played(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            job = _SpeechJob(chunks=["текст"], generation=service._generation)
            job.cancelled.set()
            with patch("sounddevice.OutputStream") as stream:
                self.assertFalse(service._play(np.zeros(100, dtype=np.float32), job))
            stream.assert_not_called()

    def test_playing_marks_service_busy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            job = _SpeechJob(chunks=["текст"], generation=service._generation)
            seen: list[bool] = []
            stream = FakeStream(on_write=lambda: seen.append(service.speaking))

            with patch("sounddevice.OutputStream", return_value=stream):
                self.assertTrue(service._play(np.zeros(100, dtype=np.float32), job))

            self.assertEqual(seen, [True])
            self.assertFalse(service.speaking)
            self.assertTrue(stream.started)
            self.assertTrue(stream.stopped)
            self.assertTrue(stream.closed)
            self.assertFalse(stream.aborted)

    def test_stop_during_playback_aborts_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            job = _SpeechJob(chunks=["текст"], generation=service._generation)
            stream = FakeStream(on_write=service.stop)

            with (
                patch("sounddevice.OutputStream", return_value=stream),
                patch("sounddevice.stop") as legacy_stop,
            ):
                self.assertFalse(service._play(np.zeros(100, dtype=np.float32), job))

            # PortAudio из чужого потока не останавливаем: это роняло процесс.
            legacy_stop.assert_not_called()
            self.assertTrue(stream.aborted)
            self.assertTrue(stream.closed)

    def test_stop_does_not_touch_portaudio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            with patch("sounddevice.stop") as legacy_stop:
                service.stop()
            legacy_stop.assert_not_called()

    def test_phrase_is_cut_by_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            chunks: list[int] = []

            def on_write() -> None:
                chunks.append(1)
                service.stop()

            stream = FakeStream(on_write=on_write)
            with patch("sounddevice.OutputStream", return_value=stream):
                # Фраза идёт без job: отмена ловится по смене поколения.
                self.assertFalse(service._play(np.zeros(10, dtype=np.float32)))
            self.assertEqual(len(chunks), 1)
            self.assertTrue(stream.aborted)


class TTSStreamTest(unittest.TestCase):
    def test_stream_is_none_when_disabled(self) -> None:
        self.assertIsNone(TTSService({"tts": {"enabled": False}}).stream())
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(make_service(Path(tmp), api_key="").stream())

    def test_stream_speaks_sentences_as_they_arrive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            played: list[str] = []
            requested: list[str] = []

            def fake_request(text: str) -> bytes:
                requested.append(text)
                return make_pcm()

            with (
                patch.object(service, "_request", side_effect=fake_request),
                patch.object(
                    service,
                    "_play",
                    side_effect=lambda data, job=None: played.append("play") or True,
                ),
            ):
                stream = service.stream()
                self.assertIsNotNone(stream)
                stream.feed("Первое предложение. Вто")
                deadline = time.monotonic() + 2
                while not played and time.monotonic() < deadline:
                    time.sleep(0.01)
                # Пока поток открыт, уже звучит только первое предложение.
                self.assertEqual(played, ["play"])
                self.assertEqual(requested, ["Первое предложение."])

                stream.feed("рое предложение.")
                stream.close()
                self.assertTrue(service.wait(5))
            self.assertEqual(played, ["play", "play"])
            self.assertEqual(requested, ["Первое предложение.", "Второе предложение."])

    def test_stream_flushes_remainder_on_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            requested: list[str] = []
            with (
                patch.object(
                    service,
                    "_request",
                    side_effect=lambda text: requested.append(text) or make_pcm(),
                ),
                patch.object(service, "_play", return_value=True),
            ):
                stream = service.stream()
                stream.feed("Без точки в конце")
                stream.close()
                self.assertTrue(service.wait(5))
            self.assertEqual(requested, ["Без точки в конце"])

    def test_stream_respects_max_chars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp), max_chars=40)
            requested: list[str] = []
            with (
                patch.object(
                    service,
                    "_request",
                    side_effect=lambda text: requested.append(text) or make_pcm(),
                ),
                patch.object(service, "_play", return_value=True),
            ):
                stream = service.stream()
                stream.feed("Первое длинное предложение здесь. Второе предложение тоже здесь.")
                stream.close()
                self.assertTrue(service.wait(5))
            self.assertEqual(requested, ["Первое длинное предложение здесь."])
            self.assertLessEqual(sum(len(text) for text in requested), 40)

    def test_stop_drops_stream_chunks_fed_after_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            playing = threading.Event()
            played: list[str] = []

            def blocking_play(data, job=None) -> bool:
                played.append("play")
                playing.set()
                if job is None:
                    return True
                return not job.cancelled.wait(timeout=2)

            with (
                patch.object(service, "_request", return_value=make_pcm()),
                patch.object(service, "_play", side_effect=blocking_play),
            ):
                stream = service.stream()
                stream.feed("Первое предложение здесь. ")
                self.assertTrue(playing.wait(2))
                service.stop()
                stream.feed("Второе предложение здесь.")
                stream.close()
                self.assertTrue(service.wait(2))
            self.assertEqual(len(played), 1)

    def test_stream_does_not_cut_short_sentence_at_comma(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            requested: list[str] = []
            with (
                patch.object(
                    service,
                    "_request",
                    side_effect=lambda text: requested.append(text) or make_pcm(),
                ),
                patch.object(service, "_play", return_value=True),
            ):
                stream = service.stream()
                text = "Небо кажется голубым потому, что свет рассеивается."
                stream.feed(text)
                stream.close()
                self.assertTrue(service.wait(5))
            self.assertEqual(requested, [text])

    def test_stream_does_not_start_on_comma(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            requested: list[str] = []
            with (
                patch.object(
                    service,
                    "_request",
                    side_effect=lambda text: requested.append(text) or make_pcm(),
                ),
                patch.object(service, "_play", return_value=True),
            ):
                stream = service.stream()
                stream.feed("Привет, как дела")
                time.sleep(0.2)
                self.assertEqual(requested, [])
                stream.feed("?")
                stream.close()
                self.assertTrue(service.wait(5))
            self.assertEqual(requested, ["Привет, как дела?"])

    def test_stream_long_sentence_falls_back_to_comma(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            requested: list[str] = []
            with (
                patch.object(
                    service,
                    "_request",
                    side_effect=lambda text: requested.append(text) or make_pcm(),
                ),
                patch.object(service, "_play", return_value=True),
            ):
                stream = service.stream()
                stream.feed(
                    "Первая часть предложения, которая уже длинная, "
                    "вторая часть предложения, которая тоже длинная, "
                    "третья часть предложения, которая добавляет детали, "
                    "четвёртая часть предложения, которая завершает мысль."
                )
                stream.close()
                self.assertTrue(service.wait(5))
            self.assertGreaterEqual(len(requested), 2)
            self.assertTrue(requested[0].endswith(","))
            self.assertLessEqual(len(requested[0]), 180)

    def test_stream_short_first_sentence_stays_whole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            requested: list[str] = []
            with (
                patch.object(
                    service,
                    "_request",
                    side_effect=lambda text: requested.append(text) or make_pcm(),
                ),
                patch.object(service, "_play", return_value=True),
            ):
                stream = service.stream()
                stream.feed(
                    "Привет. Дальше идёт длинное предложение, "
                    "которое целиком уйдёт в озвучку вторым фрагментом."
                )
                stream.close()
                self.assertTrue(service.wait(5))
            self.assertEqual(requested[0], "Привет.")

    def test_stream_job_waits_for_more_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            with (
                patch.object(service, "_request", return_value=make_pcm()) as request,
                patch.object(service, "_play", return_value=True),
            ):
                stream = service.stream()
                stream.feed("Привет")
                # Точки нет: предложение ещё не готово, синтеза нет.
                time.sleep(0.2)
                request.assert_not_called()
                stream.close()
                self.assertTrue(service.wait(5))

    def test_sealed_job_plays_tail_missed_by_first_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            job = _SpeechJob(generation=service._generation)
            played: list[str] = []
            calls = {"count": 0}
            real_snapshot = job.snapshot

            def snapshot() -> list[str]:
                # Первый снимок — до добавления хвоста, дальше как есть.
                calls["count"] += 1
                if calls["count"] == 1:
                    return []
                return real_snapshot()

            with (
                patch.object(job, "snapshot", side_effect=snapshot),
                patch.object(service, "_request", return_value=make_pcm()),
                patch.object(
                    service,
                    "_play",
                    side_effect=lambda data, job=None: played.append("play") or True,
                ),
            ):
                job.add(["Хвост после закрытия."])
                job.seal()
                service._run_job(job)
            self.assertEqual(played, ["play"])

    def test_stream_skips_failed_chunk_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            played: list[float] = []
            values = {
                "Первое предложение здесь.": 1.0,
                "Второе предложение здесь.": 2.0,
                "Третье предложение здесь.": 3.0,
            }

            def fake_synth(text: str) -> np.ndarray:
                if text == "Второе предложение здесь.":
                    raise RuntimeError("boom")
                return np.array([values[text]], dtype=np.float32)

            def fake_play(data, job=None) -> bool:  # noqa: ANN001
                played.append(float(data[0]))
                return True

            with (
                patch.object(service, "_synth_cached", side_effect=fake_synth),
                patch.object(service, "_play", side_effect=fake_play),
            ):
                stream = service.stream()
                stream.feed(
                    "Первое предложение здесь. Второе предложение здесь. Третье предложение здесь."
                )
                stream.close()
                self.assertTrue(service.wait(5))
            self.assertEqual(played, [1.0, 3.0])

    def test_stream_skips_playback_error_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            played: list[float] = []
            values = {
                "Первое предложение здесь.": 1.0,
                "Второе предложение здесь.": 2.0,
                "Третье предложение здесь.": 3.0,
            }

            def fake_synth(text: str) -> np.ndarray:
                return np.array([values[text]], dtype=np.float32)

            def fake_play(data, job=None) -> bool:  # noqa: ANN001
                value = float(data[0])
                if value == 1.0:
                    return False
                played.append(value)
                return True

            with (
                patch.object(service, "_synth_cached", side_effect=fake_synth),
                patch.object(service, "_play", side_effect=fake_play),
            ):
                stream = service.stream()
                stream.feed(
                    "Первое предложение здесь. Второе предложение здесь. Третье предложение здесь."
                )
                stream.close()
                self.assertTrue(service.wait(5))
            self.assertEqual(played, [2.0, 3.0])

    def test_stream_skips_chunk_that_takes_too_long(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp), chunk_wait_sec=1.0)
            played: list[float] = []
            release = threading.Event()
            values = {
                "Первое предложение здесь.": 1.0,
                "Второе предложение здесь.": 2.0,
                "Третье предложение здесь.": 3.0,
            }

            def fake_synth(text: str) -> np.ndarray:
                if text == "Второе предложение здесь.":
                    release.wait(timeout=5)
                return np.array([values[text]], dtype=np.float32)

            def fake_play(data, job=None) -> bool:  # noqa: ANN001
                played.append(float(data[0]))
                return True

            try:
                with (
                    patch.object(service, "_synth_cached", side_effect=fake_synth),
                    patch.object(service, "_play", side_effect=fake_play),
                ):
                    stream = service.stream()
                    stream.feed(
                        "Первое предложение здесь. "
                        "Второе предложение здесь. "
                        "Третье предложение здесь."
                    )
                    stream.close()
                    self.assertTrue(service.wait(5))
            finally:
                release.set()
            self.assertEqual(played, [1.0, 3.0])

    def test_stream_prefetches_next_chunk_while_current_plays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(Path(tmp))
            requested: list[str] = []
            playing = threading.Event()
            release = threading.Event()

            def fake_synth(text: str) -> np.ndarray:
                requested.append(text)
                return np.zeros(10, dtype=np.float32)

            def fake_play(data, job=None) -> bool:  # noqa: ANN001
                playing.set()
                release.wait(timeout=5)
                return True

            with (
                patch.object(service, "_synth_cached", side_effect=fake_synth),
                patch.object(service, "_play", side_effect=fake_play),
            ):
                stream = service.stream()
                try:
                    # Пробел в конце нужен, чтобы вторая точка стала границей
                    # предложения ещё до close().
                    stream.feed("Первое предложение здесь. Второе предложение здесь. ")
                    self.assertTrue(playing.wait(2))
                    deadline = time.monotonic() + 2
                    while len(requested) < 2 and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertEqual(
                        requested,
                        [
                            "Первое предложение здесь.",
                            "Второе предложение здесь.",
                        ],
                    )
                finally:
                    stream.close()
                    release.set()
            self.assertTrue(service.wait(5))


if __name__ == "__main__":
    unittest.main()
