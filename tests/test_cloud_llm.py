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


class FakeStreamResponse:
    """SSE-ответ: итерируется построчно, как http.client.HTTPResponse."""

    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines

    def __enter__(self) -> FakeStreamResponse:
        return self

    def __exit__(self, *args) -> bool:
        return False

    def __iter__(self):
        return iter(self.lines)


def sse_lines(*texts: str, done: bool = True) -> list[bytes]:
    lines = [
        (
            "data: "
            + json.dumps(
                {"choices": [{"delta": {"content": text}}]},
                ensure_ascii=False,
            )
            + "\n\n"
        ).encode("utf-8")
        for text in texts
    ]
    if done:
        lines.append(b"data: [DONE]\n\n")
    return lines


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


def responses_body(text: str = "Ответ") -> bytes:
    return json.dumps(
        {
            "model": "anthropic/claude-haiku-5-5",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 5},
                "output_tokens_details": {"reasoning_tokens": 3},
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")


def responses_sse(
    *texts: str,
    done: bool = True,
    search: bool = True,
) -> list[bytes]:
    events = [{"type": "response.created"}]
    if search:
        events.append({"type": "response.web_search_call.searching"})
    events.extend({"type": "response.output_text.delta", "delta": text} for text in texts)
    events.append(
        {
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                }
            },
        }
    )
    lines = [
        ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode("utf-8")
        for event in events
    ]
    if done:
        lines.append(b"data: [DONE]\n\n")
    return lines


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

    def test_answer_without_context_includes_datetime(self) -> None:
        llm = make_llm()
        now = "Сегодня 13 ноября 2026 года, пятница. Сейчас 20:47."
        with (
            patch("app.cloud_llm.datetime_context", return_value=now),
            patch.object(llm, "_complete", return_value="Ответ") as complete,
        ):
            llm.answer("что такое фотосинтез")
        _, system, _ = complete.call_args.args
        self.assertIn(now, system)

    def test_answer_with_history_includes_datetime(self) -> None:
        llm = make_llm()
        now = "Сегодня 13 ноября 2026 года, пятница. Сейчас 20:47."
        with (
            patch("app.cloud_llm.datetime_context", return_value=now),
            patch.object(llm, "_complete", return_value="Ответ") as complete,
        ):
            llm.answer("а почему?", history="Прошлый ответ")
        _, system, _ = complete.call_args.args
        self.assertIn(now, system)

    def test_answer_with_context_has_no_datetime(self) -> None:
        llm = make_llm()
        now = "Сегодня 13 ноября 2026 года, пятница. Сейчас 20:47."
        with (
            patch("app.cloud_llm.datetime_context", return_value=now),
            patch.object(llm, "_complete", return_value="Ответ [1]") as complete,
        ):
            llm.answer("вопрос", "[1] Источник")
        _, system, _ = complete.call_args.args
        self.assertNotIn(now, system)

    def test_answer_with_context_uses_rag_prompt(self) -> None:
        llm = make_llm()
        with patch.object(llm, "_complete", return_value="Ответ [1]") as complete:
            llm.answer("вопрос", "[1] Источник\nтекст")
        _, system, user = complete.call_args.args
        self.assertIn("по источникам", system)
        self.assertIn("ссылку", system)
        self.assertIn("Источники:", user)
        self.assertIn("[1] Источник", user)

    def test_answer_with_history_uses_followup_prompt(self) -> None:
        llm = make_llm()
        history = (
            "Предыдущий вопрос: что такое фотосинтез\n"
            "Предыдущий ответ: Процесс, при котором растения выделяют кислород."
        )
        with patch.object(llm, "_complete", return_value="Ответ") as complete:
            llm.answer("а почему это важно?", history=history)
        _, system, user = complete.call_args.args
        self.assertIn("продолжение разговора", system)
        self.assertIn("новые детали", system)
        self.assertIn(history, user)
        self.assertIn("Уточняющий вопрос: а почему это важно?", user)

    def test_answer_with_context_and_history_keeps_rag_prompt(self) -> None:
        llm = make_llm()
        with patch.object(llm, "_complete", return_value="Ответ [1]") as complete:
            llm.answer("а подробнее?", "[1] Источник", history="Прошлый ответ")
        _, system, user = complete.call_args.args
        self.assertIn("по источникам", system)
        self.assertNotIn("продолжение разговора", system)
        self.assertIn("Прошлый ответ", user)
        self.assertIn("Вопрос: а подробнее?", user)

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
        self.assertEqual(complete.call_args.args[0], "anthropic/claude-haiku-5-5")

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

        with (
            patch.object(llm, "_complete", side_effect=fake_complete),
            patch("app.cloud_llm.time.sleep"),
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
        with patch.object(llm, "_complete", side_effect=LLMError("пустой ответ модели")):
            with self.assertRaises(LLMError) as ctx:
                llm.answer("вопрос")
        self.assertIn("модель не ответила", str(ctx.exception))


class CloudLLMChatTest(unittest.TestCase):
    def test_chat_builds_messages_with_system(self) -> None:
        llm = make_llm()
        captured: dict = {}

        def fake_post(model, messages, **kwargs):
            captured["model"] = model
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "Ответ"

        with patch.object(llm, "_post", side_effect=fake_post):
            result = llm.chat(
                [{"role": "user", "content": "привет"}],
                system="правила",
                temperature=0.5,
                timeout=10,
            )
        self.assertEqual(result, "Ответ")
        self.assertEqual(captured["model"], "anthropic/claude-haiku-5-5")
        self.assertEqual(
            captured["messages"],
            [
                {"role": "system", "content": "правила"},
                {"role": "user", "content": "привет"},
            ],
        )
        self.assertEqual(captured["kwargs"]["temperature"], 0.5)
        self.assertEqual(captured["kwargs"]["timeout"], 10)
        self.assertEqual(captured["kwargs"]["max_tokens"], 2000)

    def test_chat_skips_empty_and_unknown_messages(self) -> None:
        llm = make_llm()
        captured: dict = {}

        def fake_post(model, messages, **kwargs):
            captured["messages"] = messages
            return "Ответ"

        with patch.object(llm, "_post", side_effect=fake_post):
            llm.chat(
                [
                    {"role": "tool", "content": "не в роль"},
                    {"role": "user", "content": "  привет  "},
                    {"role": "assistant", "content": "   "},
                ]
            )
        self.assertEqual(captured["messages"], [{"role": "user", "content": "привет"}])

    def test_chat_empty_messages(self) -> None:
        with self.assertRaises(LLMError):
            make_llm().chat([])

    def test_chat_disabled(self) -> None:
        with self.assertRaises(LLMError):
            make_llm(enabled=False).chat([{"role": "user", "content": "привет"}])

    def test_chat_missing_key(self) -> None:
        with self.assertRaises(LLMError):
            make_llm(api_key="").chat([{"role": "user", "content": "привет"}])

    def test_chat_uses_fallback_model(self) -> None:
        llm = make_llm()
        calls: list[str] = []

        def fake_post(model, messages, **kwargs):
            calls.append(model)
            if model == "anthropic/claude-haiku-5-5":
                raise LLMError("пустой ответ модели")
            return "Ответ запасной"

        with patch.object(llm, "_post", side_effect=fake_post):
            self.assertEqual(
                llm.chat([{"role": "user", "content": "привет"}]),
                "Ответ запасной",
            )
        self.assertEqual(
            calls,
            ["anthropic/claude-haiku-5-5", "xiaomi/mimo-v2.6-flash"],
        )


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
        with patch("app.cloud_llm.urllib.request.urlopen", side_effect=fake_urlopen):
            text = llm._complete(
                "anthropic/claude-haiku-5-5",
                "system",
                "user",
                max_tokens=2000,
                temperature=0.2,
                timeout=30,
            )
        self.assertEqual(text, "Ответ")
        self.assertEqual(captured["url"], "https://api.proxyapi.ru/v1/chat/completions")
        self.assertEqual(captured["auth"], "Bearer test-key")
        payload = captured["payload"]
        self.assertEqual(payload["model"], "anthropic/claude-haiku-5-5")
        self.assertEqual(payload["max_completion_tokens"], 2000)
        self.assertEqual(payload["reasoning_effort"], "minimal")
        self.assertEqual(payload["temperature"], 0.2)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(captured["timeout"], 30)

    def test_model_params_override_and_remove_fields(self) -> None:
        llm = make_llm(
            model_params={
                "openai/gpt-6-luna": {
                    "temperature": None,
                    "reasoning_effort": "none",
                }
            }
        )

        payload = llm._payload(
            "openai/gpt-6-luna",
            [{"role": "user", "content": "привет"}],
            max_tokens=2000,
            temperature=0.2,
            stream=False,
        )

        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["max_completion_tokens"], 2000)

    def test_model_params_do_not_affect_other_models(self) -> None:
        llm = make_llm(model_params={"openai/gpt-6-luna": {"temperature": None}})

        payload = llm._payload(
            "xiaomi/mimo-v2.6-flash",
            [{"role": "user", "content": "привет"}],
            max_tokens=2000,
            temperature=0.2,
            stream=False,
        )

        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["reasoning_effort"], "minimal")

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
        with patch("app.cloud_llm.urllib.request.urlopen", side_effect=error) as urlopen:
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
        with (
            patch(
                "app.cloud_llm.urllib.request.urlopen",
                side_effect=[error, FakeResponse(body=response_body("Ответ"))],
            ) as urlopen,
            patch("app.cloud_llm.time.sleep"),
        ):
            self.assertEqual(llm.answer("вопрос"), "Ответ")
        self.assertEqual(urlopen.call_count, 2)

    def test_empty_response_is_an_error(self) -> None:
        body = json.dumps({"choices": [{"message": {"content": "  "}}]}).encode("utf-8")
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


class CloudLLMStreamTest(unittest.TestCase):
    def test_stream_sends_sse_payload(self) -> None:
        llm = make_llm(retries=0, fallback_models=[])
        captured = {}

        def fake_urlopen(request, timeout=None):  # noqa: ANN001
            captured["request"] = request
            return FakeStreamResponse(sse_lines("Ответ"))

        with patch("app.cloud_llm.urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertEqual(
                list(llm.chat_stream([{"role": "user", "content": "привет"}])),
                ["Ответ"],
            )

        payload = json.loads(captured["request"].data)
        self.assertTrue(payload["stream"])
        self.assertEqual(
            captured["request"].headers["Accept"],
            "text/event-stream",
        )

    def test_stream_yields_deltas_in_order_and_skips_noise(self) -> None:
        llm = make_llm(retries=0, fallback_models=[])
        lines = [
            b": keep-alive\n\n",
            b"event: message\n\n",
            b"data: not json\n\n",
            json.dumps(
                {"choices": [{"delta": {"role": "assistant"}}]},
                ensure_ascii=False,
            ).encode("utf-8"),
            *sse_lines("При", "вет", ". Как дела?"),
        ]
        with patch(
            "app.cloud_llm.urllib.request.urlopen",
            return_value=FakeStreamResponse(lines),
        ):
            text = "".join(llm.chat_stream([{"role": "user", "content": "привет"}]))

        self.assertEqual(text, "Привет. Как дела?")

    def test_stream_falls_back_before_first_chunk(self) -> None:
        llm = make_llm(retries=0)
        calls: list[str] = []

        def fake_stream(model, messages, **kwargs):  # noqa: ANN001
            calls.append(model)
            if model == llm.model:
                raise LLMError("boom")
            yield "запас"

        with patch.object(llm, "_post_stream", side_effect=fake_stream):
            self.assertEqual(
                list(llm.chat_stream([{"role": "user", "content": "привет"}])),
                ["запас"],
            )

        self.assertEqual(calls, [llm.model, llm.fallback_models[0]])

    def test_stream_does_not_fall_back_after_first_chunk(self) -> None:
        llm = make_llm(retries=0)
        calls: list[str] = []

        def fake_stream(model, messages, **kwargs):  # noqa: ANN001
            calls.append(model)
            if model == llm.model:
                yield "начало"
                raise LLMError("обрыв")
            yield "запас"

        with patch.object(llm, "_post_stream", side_effect=fake_stream):
            with self.assertRaises(LLMError) as ctx:
                list(llm.chat_stream([{"role": "user", "content": "привет"}]))

        self.assertIn("поток прерван", str(ctx.exception))
        self.assertEqual(calls, [llm.model])

    def test_stream_empty_answer_is_an_error(self) -> None:
        llm = make_llm(retries=0, fallback_models=[])
        with patch(
            "app.cloud_llm.urllib.request.urlopen",
            return_value=FakeStreamResponse([b"data: [DONE]\n\n"]),
        ):
            with self.assertRaises(LLMError) as ctx:
                list(llm.chat_stream([{"role": "user", "content": "привет"}]))

        self.assertIn("пустой ответ", str(ctx.exception))

    def test_stream_requires_messages_and_key(self) -> None:
        with self.assertRaises(LLMError):
            make_llm().chat_stream([])
        with self.assertRaises(LLMError):
            make_llm(api_key="").chat_stream([{"role": "user", "content": "привет"}])


class CloudLLMWebSearchTest(unittest.TestCase):
    def test_web_search_enabled_by_default(self) -> None:
        self.assertTrue(make_llm().web_search_enabled)

    def test_web_search_can_be_disabled(self) -> None:
        llm = make_llm(web_search={"enabled": False})
        self.assertFalse(llm.web_search_enabled)

    def test_responses_payload_maps_fields(self) -> None:
        llm = make_llm(
            model_params={
                "openai/gpt-6-luna": {
                    "temperature": None,
                    "reasoning_effort": "none",
                }
            }
        )
        payload = llm._responses_payload(
            "openai/gpt-6-luna",
            [
                {"role": "system", "content": "правила"},
                {"role": "user", "content": "вопрос"},
                {"role": "assistant", "content": "ответ"},
            ],
            max_tokens=2000,
            temperature=0.5,
            stream=False,
        )

        self.assertEqual(payload["instructions"], "правила")
        self.assertEqual(
            payload["input"],
            [
                {"role": "user", "content": "вопрос"},
                {"role": "assistant", "content": "ответ"},
            ],
        )
        self.assertEqual(payload["tools"], [{"type": "web_search"}])
        self.assertEqual(payload["max_output_tokens"], 2000)
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["reasoning"], {"effort": "none"})
        self.assertFalse(payload["stream"])

    def test_chat_with_web_search_uses_responses_endpoint(self) -> None:
        captured: dict = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(body=responses_body("Ответ"))

        llm = make_llm()
        with patch("app.cloud_llm.urllib.request.urlopen", side_effect=fake_urlopen):
            text = llm.chat(
                [{"role": "user", "content": "привет"}],
                system="правила",
                web_search=True,
            )

        self.assertEqual(text, "Ответ")
        self.assertEqual(captured["url"], "https://api.proxyapi.ru/v1/responses")
        payload = captured["payload"]
        self.assertEqual(payload["instructions"], "правила")
        self.assertEqual(payload["tools"], [{"type": "web_search"}])
        self.assertEqual(payload["input"], [{"role": "user", "content": "привет"}])

    def test_answer_with_web_search_adds_prompt_suffix(self) -> None:
        captured: dict = {}

        def fake_urlopen(request, timeout=None):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(body=responses_body("Ответ"))

        llm = make_llm()
        with patch("app.cloud_llm.urllib.request.urlopen", side_effect=fake_urlopen):
            text = llm.answer("кто такой никс", web_search=True)

        self.assertEqual(text, "Ответ")
        instructions = captured["payload"]["instructions"]
        self.assertIn("инструмент поиска", instructions)

    def test_web_search_does_not_use_fallback_models(self) -> None:
        error = urllib.error.HTTPError(
            "https://api.proxyapi.ru/v1/responses",
            500,
            "Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b"boom"),
        )
        llm = make_llm(retries=0)
        with patch(
            "app.cloud_llm.urllib.request.urlopen",
            side_effect=error,
        ) as urlopen:
            with self.assertRaises(LLMError):
                llm.chat([{"role": "user", "content": "привет"}], web_search=True)

        self.assertEqual(urlopen.call_count, 1)
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["model"], llm.model)

    def test_responses_stream_reads_text_deltas(self) -> None:
        llm = make_llm(retries=0)
        with patch(
            "app.cloud_llm.urllib.request.urlopen",
            return_value=FakeStreamResponse(responses_sse("При", "вет")),
        ):
            text = "".join(
                llm.chat_stream(
                    [{"role": "user", "content": "привет"}],
                    web_search=True,
                )
            )

        self.assertEqual(text, "Привет")

    def test_responses_stream_buffers_and_cleans_citations(self) -> None:
        llm = make_llm(retries=0)
        lines = responses_sse(
            "Ответ. ([community.creative-assembly.com](",
            "https://example.com/forum?source=openai))",
        )
        with patch(
            "app.cloud_llm.urllib.request.urlopen",
            return_value=FakeStreamResponse(lines),
        ):
            chunks = list(
                llm.chat_stream(
                    [{"role": "user", "content": "привет"}],
                    web_search=True,
                )
            )

        self.assertEqual(chunks, ["Ответ."])

    def test_responses_stream_without_search_still_streams_deltas(self) -> None:
        llm = make_llm(retries=0)
        lines = responses_sse("При", "вет", search=False)
        with patch(
            "app.cloud_llm.urllib.request.urlopen",
            return_value=FakeStreamResponse(lines),
        ):
            chunks = list(
                llm.chat_stream(
                    [{"role": "user", "content": "привет"}],
                    web_search=True,
                )
            )

        self.assertEqual(chunks, ["При", "вет"])

    def test_responses_stream_failed_event_is_an_error(self) -> None:
        lines = [
            (
                "data: "
                + json.dumps(
                    {
                        "type": "response.failed",
                        "response": {"error": {"message": "поиск недоступен"}},
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            ).encode("utf-8"),
        ]
        llm = make_llm(retries=0)
        with patch(
            "app.cloud_llm.urllib.request.urlopen",
            return_value=FakeStreamResponse(lines),
        ):
            with self.assertRaises(LLMError) as ctx:
                list(
                    llm.chat_stream(
                        [{"role": "user", "content": "привет"}],
                        web_search=True,
                    )
                )

        self.assertIn("поиск недоступен", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
