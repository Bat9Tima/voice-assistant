from __future__ import annotations

import io
import json
import unittest
import urllib.error
import wave
from unittest.mock import Mock, patch

from app.stt import (
    DeepgramError,
    DeepgramTranscriber,
    GigaAMTranscriber,
    Transcript,
    create_transcriber,
)

PCM = b"\x00\x00" * 1600


def make(**kwargs) -> DeepgramTranscriber:
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("fallback", None)
    return DeepgramTranscriber(**kwargs)


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class DeepgramTranscriberTest(unittest.TestCase):
    def test_parses_text_and_sends_wav(self) -> None:
        transcribe = make()
        with patch.object(transcribe, "_request", return_value={"text": "Привет, Luna"}) as request:
            result = transcribe.transcribe_details(PCM)

        self.assertEqual(result.text, "Привет, Luna")
        self.assertEqual(result.language, "multi")
        wav = request.call_args.args[0]
        self.assertTrue(wav.startswith(b"RIFF"))
        with wave.open(io.BytesIO(wav), "rb") as stream:
            self.assertEqual(stream.getnchannels(), 1)
            self.assertEqual(stream.getsampwidth(), 2)
            self.assertEqual(stream.getframerate(), 16000)
            self.assertEqual(stream.getnframes(), 1600)

    def test_body_contains_model_language_and_hints(self) -> None:
        transcribe = make()
        boundary, body = transcribe._build_body(b"RIFFdata")
        text = body.decode("utf-8", "replace")

        self.assertIn(f"--{boundary}", text)
        self.assertIn('name="model"\r\n\r\ndeepgram/nova-3', text)
        self.assertIn('name="language"\r\n\r\nmulti', text)
        self.assertEqual(text.count('name="language_hint"'), 2)
        self.assertIn('name="file"; filename="command.wav"', text)
        self.assertIn("RIFFdata", text)

    def test_fallback_on_request_error(self) -> None:
        fallback = Mock()
        fallback.transcribe_details.return_value = Transcript(text="локально")
        transcribe = make(fallback=fallback)
        with patch.object(transcribe, "_request", side_effect=DeepgramError("нет сети")):
            result = transcribe.transcribe_details(PCM)

        self.assertEqual(result.text, "локально")
        fallback.transcribe_details.assert_called_once_with(PCM)

    def test_fallback_on_empty_response(self) -> None:
        fallback = Mock()
        fallback.transcribe_details.return_value = Transcript(text="локально")
        transcribe = make(fallback=fallback)
        with patch.object(transcribe, "_request", return_value={"text": "  "}):
            result = transcribe.transcribe_details(PCM)

        self.assertEqual(result.text, "локально")
        fallback.transcribe_details.assert_called_once_with(PCM)

    @patch.dict("os.environ", {"PROXYAPI_KEY": "", "DEEPGRAM_API_KEY": ""})
    def test_missing_key_uses_fallback(self) -> None:
        fallback = Mock()
        fallback.transcribe_details.return_value = Transcript(text="локально")
        transcribe = make(api_key="", fallback=fallback)
        with patch.object(transcribe, "_request") as request:
            result = transcribe.transcribe_details(PCM)

        self.assertEqual(result.text, "локально")
        request.assert_not_called()

    @patch.dict("os.environ", {"PROXYAPI_KEY": "", "DEEPGRAM_API_KEY": ""})
    def test_missing_key_without_fallback_raises(self) -> None:
        transcribe = make(api_key="")
        with self.assertRaises(DeepgramError):
            transcribe.transcribe_details(PCM)

    def test_empty_pcm_does_not_call_api(self) -> None:
        transcribe = make()
        with patch.object(transcribe, "_request") as request:
            result = transcribe.transcribe_details(b"")

        self.assertEqual(result.text, "")
        request.assert_not_called()

    def test_http_error_is_reported(self) -> None:
        transcribe = make()
        error = urllib.error.HTTPError(
            "https://example.test",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":"bad key"}'),
        )
        with (
            patch("app.stt.urllib.request.urlopen", side_effect=error),
            self.assertRaises(DeepgramError) as caught,
        ):
            transcribe._request(b"RIFFdata")

        self.assertIn("401", str(caught.exception))
        self.assertIn("bad key", str(caught.exception))

    def test_request_parses_json(self) -> None:
        transcribe = make()
        response = FakeResponse(json.dumps({"text": "ok"}).encode())
        with patch("app.stt.urllib.request.urlopen", return_value=response):
            payload = transcribe._request(b"RIFFdata")

        self.assertEqual(payload, {"text": "ok"})

    def test_request_rejects_non_json(self) -> None:
        transcribe = make()
        response = FakeResponse(b"not json")
        with (
            patch("app.stt.urllib.request.urlopen", return_value=response),
            self.assertRaises(DeepgramError),
        ):
            transcribe._request(b"RIFFdata")

    def test_request_rejects_non_object(self) -> None:
        transcribe = make()
        response = FakeResponse(b'[{"text": "ok"}]')
        with (
            patch("app.stt.urllib.request.urlopen", return_value=response),
            self.assertRaises(DeepgramError),
        ):
            transcribe._request(b"RIFFdata")

    def test_network_error_is_reported(self) -> None:
        transcribe = make()
        with (
            patch("app.stt.urllib.request.urlopen", side_effect=OSError("нет сети")),
            self.assertRaises(DeepgramError) as caught,
        ):
            transcribe._request(b"RIFFdata")

        self.assertIn("нет сети", str(caught.exception))

    @patch.dict("os.environ", {"PROXYAPI_KEY": "env-key"})
    def test_key_can_come_from_environment(self) -> None:
        transcribe = make(api_key="")
        self.assertEqual(transcribe._api_key, "env-key")
        with patch.object(transcribe, "_request", return_value={"text": "ok"}):
            self.assertEqual(transcribe.transcribe(PCM), "ok")

    def test_load_is_noop_and_english_is_available(self) -> None:
        transcribe = make()
        transcribe.load()

        self.assertTrue(transcribe.english_loaded)

    @patch.dict("os.environ", {"PROXYAPI_KEY": "", "DEEPGRAM_API_KEY": ""})
    def test_loaded_reflects_key_or_fallback(self) -> None:
        self.assertTrue(make().loaded)
        self.assertTrue(make(api_key="", fallback=Mock()).loaded)
        self.assertFalse(make(api_key="").loaded)


class CreateDeepgramTranscriberTest(unittest.TestCase):
    def test_builds_from_config(self) -> None:
        cfg = {
            "audio": {"sample_rate": 8000},
            "stt": {
                "engine": "deepgram",
                "num_threads": 2,
                "deepgram": {
                    "model": "deepgram/nova-3",
                    "language": "multi",
                    "language_hints": ["ru", "en"],
                    "timeout_sec": 7,
                    "api_key": "config-key",
                },
            },
        }
        transcribe = create_transcriber(cfg)

        self.assertIsInstance(transcribe, DeepgramTranscriber)
        self.assertEqual(transcribe.model, "deepgram/nova-3")
        self.assertEqual(transcribe.language, "multi")
        self.assertEqual(transcribe.language_hints, ["ru", "en"])
        self.assertEqual(transcribe.timeout, 7)
        self.assertEqual(transcribe.sample_rate, 8000)
        self.assertIsInstance(transcribe.fallback, GigaAMTranscriber)

    def test_fallback_can_be_disabled(self) -> None:
        cfg = {
            "stt": {
                "engine": "deepgram",
                "deepgram": {"api_key": "config-key", "fallback": False},
            }
        }
        transcribe = create_transcriber(cfg)

        self.assertIsInstance(transcribe, DeepgramTranscriber)
        self.assertIsNone(transcribe.fallback)

    def test_language_hints_accept_comma_string(self) -> None:
        cfg = {
            "stt": {
                "engine": "deepgram",
                "deepgram": {"api_key": "config-key", "language_hints": "ru, en"},
            }
        }
        transcribe = create_transcriber(cfg)

        self.assertEqual(transcribe.language_hints, ["ru", "en"])

    def test_unknown_engine_lists_deepgram(self) -> None:
        with self.assertRaises(ValueError) as caught:
            create_transcriber({"stt": {"engine": "unknown"}})

        self.assertIn("deepgram", str(caught.exception))
