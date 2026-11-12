from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from app.cloud_llm import CloudLLM, LLMError


class FakeResponse:
    def __init__(self, body: bytes = b"", status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args) -> bool:
        return False

    def read(self) -> bytes:
        return self.body


def make_llm(**overrides) -> CloudLLM:
    section = {
        "enabled": True,
        "base_url": "https://api.proxyapi.ru/v1",
        "api_key": "test-key",
        "model": "anthropic/claude-haiku-5-5",
        "fallback_models": ["xiaomi/mimo-v2.6-flash"],
        "reasoning_effort": "minimal",
        "max_completion_tokens": 2000,
        "temperature": 0.2,
        "timeout_sec": 30,
        "retries": 0,
    }
    section.update(overrides)
    return CloudLLM({"cloud_llm": section})


def response_body(text: str = "Ответ") -> bytes:
    return json.dumps(
        {
            "model": "anthropic/claude-haiku-5-5",
            "choices": [{"message": {"content": text}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")


class CloudLLMConfigTest(unittest.TestCase):
    def test_available_with_key(self) -> None:
        self.assertTrue(make_llm().available())

    def test_unavailable_without_key(self) -> None:
        self.assertFalse(make_llm(api_key="").available())

    def test_unavailable_when_disabled(self) -> None:
        self.assertFalse(make_llm(enabled=False).available())

    def test_tts_key_is_used_as_fallback(self) -> None:
        llm = CloudLLM(
            {
                "cloud_llm": {"api_key": "", "model": "anthropic/claude-haiku-5-5"},
                "tts": {"api_key": "tts-key"},
            }
        )
        self.assertEqual(llm.api_key, "tts-key")
        self.assertTrue(llm.available())

    def test_fallback_models_exclude_primary(self) -> None:
        llm = make_llm(
            fallback_models=[
                "anthropic/claude-haiku-5-5",
                "xiaomi/mimo-v2.6-flash",
                "",
            ]
        )
        self.assertEqual(llm.fallback_models, ["xiaomi/mimo-v2.6-flash"])

    def test_completion_limit_has_floor(self) -> None:
        self.assertEqual(make_llm(max_completion_tokens=300).max_completion_tokens, 2000)


class CloudLLMAnswerTest(unittest.TestCase):
    def test_answer_without_context_uses_answer_prompt(self) -> None:
        llm = make_llm()
        with patch.object(llm, "_complete", return_value="Ответ") as complete:
            self.assertEqual(llm.answer("что такое фотосинтез"), "Ответ")
        model, system, user = complete.call_args.args
        self.assertEqual(model, "anthropic/claude-haiku-5-5")
        self.assertIn("голосовой ассистент", system)
        self.assertNotIn("источников", system)
        self.assertEqual(user, "что такое фотосинтез")

    def test_answer_with_context_uses_rag_prompt(self) -> None:
        llm = make_llm()
        with patch.object(llm, "_complete", return_value="Ответ [1]") as complete:
            llm.answer("вопрос", "[1] Источник\nтекст")
        _, system, user = complete.call_args.args
        self.assertIn("по источникам", system)
        self.assertIn("ссылку", system)
        self.assertIn("Источники:", user)
        self.assertIn("[1] Источник", user)

    def test_answer_short_uses_short_prompt(self) -> None:
        llm = make_llm()
        with patch.object(llm, "_complete", return_value="Ответ") as complete:
            llm.answer("вопрос", style="short")
        self.assertIn("1–2", complete.call_args.args[1])

    def test_answer_news_uses_news_prompt(self) -> None:
        llm = make_llm()
        with patch.object(llm, "_complete", return_value="Ответ [1]") as complete:
            llm.answer("вопрос", "[1] Источник", style="news")
        system = complete.call_args.args[1]
        self.assertIn("3–4", system)
        self.assertIn("с новой строки", system)
        self.assertNotIn("за последнюю неделю", system)

    def test_answer_detailed_uses_detailed_prompt(self) -> None:
        llm = make_llm()
        with patch.object(llm, "_complete", return_value="Ответ") as complete:
            llm.answer("вопрос", style="detailed")
        self.assertIn("4–6", complete.call_args.args[1])

    def test_answer_floors_completion_tokens(self) -> None:
        llm = make_llm()
        with patch.object(llm, "_complete", return_value="Ответ") as complete:
            llm.answer("вопрос", max_tokens=400)
        self.assertEqual(complete.call_args.kwargs["max_tokens"], 2000)

    def test_answer_passes_temperature_and_timeout(self) -> None:
        llm = make_llm()
        with patch.object(llm, "_complete", return_value="Ответ") as complete:
            llm.answer("вопрос", temperature=0.5, timeout=42)
        self.assertEqual(complete.call_args.kwargs["temperature"], 0.5)
        self.assertEqual(complete.call_args.kwargs["timeout"], 42)

    def test_article_model_from_config(self) -> None:
        llm = make_llm(article_model="anthropic/claude-haiku-5-5")
        with patch.object(llm, "_complete", return_value="Ответ") as complete:
            llm.answer("вопрос", "[1] Источник", style="article")
        self.assertEqual(
            complete.call_args.args[0], "anthropic/claude-haiku-5-5"
        )

    def test_model_override(self) -> None:
        llm = make_llm()
        with patch.object(llm, "_complete", return_value="Ответ") as complete:
            llm.answer("вопрос", model="qwen/qwen3.8-flash")
        self.assertEqual(complete.call_args.args[0], "qwen/qwen3.8-flash")

    def test_empty_question(self) -> None:
        with self.assertRaises(LLMError):
            make_llm().answer("  ")

    def test_disabled(self) -> None:
        with self.assertRaises(LLMError):
            make_llm(enabled=False).answer("вопрос")

    def test_missing_key(self) -> None:
        with self.assertRaises(LLMError):
            make_llm(api_key="").answer("вопрос")


class CloudLLMFallbackTest(unittest.TestCase):
    def test_fallback_after_empty_answer(self) -> None:
        llm = make_llm()
        calls: list[str] = []

        def fake_complete(model, *args, **kwargs):
            calls.append(model)
            if model == "anthropic/claude-haiku-5-5":
                raise LLMError("пустой ответ модели")
            return "Ответ запасной"

        with patch.object(llm, "_complete", side_effect=fake_complete):
            self.assertEqual(llm.answer("вопрос"), "Ответ запасной")
        self.assertEqual(
            calls,
            ["anthropic/claude-haiku-5-5", "xiaomi/mimo-v2.6-flash"],
        )

    def test_retryable_error_is_retried(self) -> None:
        llm = make_llm(retries=1, fallback_models=[])
        calls: list[str] = []

        def fake_complete(model, *args, **kwargs):
            calls.append(model)
            if len(calls) == 1:
                raise LLMError("временная ошибка", retryable=True)
            return "Ответ"

        with patch.object(llm, "_complete", side_effect=fake_complete), patch(
            "app.cloud_llm.time.sleep"
        ):
            self.assertEqual(llm.answer("вопрос"), "Ответ")
        self.assertEqual(calls, ["anthropic/claude-haiku-5-5"] * 2)

    def test_non_retryable_error_goes_to_fallback(self) -> None:
        llm = make_llm(retries=2)
        calls: list[str] = []

        def fake_complete(model, *args, **kwargs):
            calls.append(model)
            if model == "anthropic/claude-haiku-5-5":
                raise LLMError("HTTP 400", retryable=False)
            return "Ответ запасной"

        with patch.object(llm, "_complete", side_effect=fake_complete):
            self.assertEqual(llm.answer("вопрос"), "Ответ запасной")
        self.assertEqual(
            calls,
            ["anthropic/claude-haiku-5-5", "xiaomi/mimo-v2.6-flash"],
        )

    def test_all_models_failed(self) -> None:
        llm = make_llm()
        with patch.object(
            llm, "_complete", side_effect=LLMError("пустой ответ модели")
        ):
            with self.assertRaises(LLMError) as ctx:
                llm.answer("вопрос")
        self.assertIn("модель не ответила", str(ctx.exception))


class CloudLLMHttpTest(unittest.TestCase):
    def test_payload_and_response(self) -> None:
        captured: dict = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["auth"] = request.get_header("Authorization")
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse(body=response_body("Ответ"))

        llm = make_llm()
        with patch(
            "app.cloud_llm.urllib.request.urlopen", side_effect=fake_urlopen
        ):
            text = llm._complete(
                "anthropic/claude-haiku-5-5",
                "system",
                "user",
                max_tokens=2000,
                temperature=0.2,
                timeout=30,
            )
        self.assertEqual(text, "Ответ")
        self.assertEqual(
            captured["url"], "https://api.proxyapi.ru/v1/chat/completions"
        )
        self.assertEqual(captured["auth"], "Bearer test-key")
        payload = captured["payload"]
        self.assertEqual(payload["model"], "anthropic/claude-haiku-5-5")
        self.assertEqual(payload["max_completion_tokens"], 2000)
        self.assertEqual(payload["reasoning_effort"], "minimal")
        self.assertEqual(payload["temperature"], 0.2)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(captured["timeout"], 30)

    def test_content_parts_are_joined(self) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "Первая "},
                                {"type": "text", "text": "часть"},
                            ]
                        }
                    }
                ]
            }
        ).encode("utf-8")
        llm = make_llm()
        with patch(
            "app.cloud_llm.urllib.request.urlopen",
            return_value=FakeResponse(body=body),
        ):
            text = llm._complete(
                "model",
                "system",
                "user",
                max_tokens=2000,
                temperature=0.2,
                timeout=30,
            )
        self.assertEqual(text, "Первая часть")

    def test_http_400_is_not_retried(self) -> None:
        error = urllib.error.HTTPError(
            "https://api.proxyapi.ru/v1/chat/completions",
            400,
            "Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error": "bad model"}'),
        )
        llm = make_llm(retries=2, fallback_models=[])
        with patch(
            "app.cloud_llm.urllib.request.urlopen", side_effect=error
        ) as urlopen:
            with self.assertRaises(LLMError) as ctx:
                llm.answer("вопрос")
        self.assertEqual(urlopen.call_count, 1)
        self.assertIn("HTTP 400", str(ctx.exception))
        self.assertIn("bad model", str(ctx.exception))

    def test_http_500_is_retried(self) -> None:
        error = urllib.error.HTTPError(
            "https://api.proxyapi.ru/v1/chat/completions",
            500,
            "Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b"boom"),
        )
        llm = make_llm(retries=1, fallback_models=[])
        with patch(
            "app.cloud_llm.urllib.request.urlopen",
            side_effect=[error, FakeResponse(body=response_body("Ответ"))],
        ) as urlopen, patch("app.cloud_llm.time.sleep"):
            self.assertEqual(llm.answer("вопрос"), "Ответ")
        self.assertEqual(urlopen.call_count, 2)

    def test_empty_response_is_an_error(self) -> None:
        body = json.dumps(
            {"choices": [{"message": {"content": "  "}}]}
        ).encode("utf-8")
        llm = make_llm(retries=0, fallback_models=[])
        with patch(
            "app.cloud_llm.urllib.request.urlopen",
            return_value=FakeResponse(body=body),
        ):
            with self.assertRaises(LLMError) as ctx:
                llm.answer("вопрос")
        self.assertIn("пустой ответ", str(ctx.exception))

    def test_missing_choices_is_an_error(self) -> None:
        body = json.dumps({"choices": []}).encode("utf-8")
        llm = make_llm(retries=0, fallback_models=[])
        with patch(
            "app.cloud_llm.urllib.request.urlopen",
            return_value=FakeResponse(body=body),
        ):
            with self.assertRaises(LLMError) as ctx:
                llm.answer("вопрос")
        self.assertIn("choices", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
