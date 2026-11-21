from __future__ import annotations

import unittest
from unittest.mock import patch

from app.cloud_llm import CloudLLM, LLMError
from app.vision import DEFAULT_TRIGGERS, ScreenVision, VisionError, match_request


class MatchRequestTest(unittest.TestCase):
    def test_exact_phrase(self) -> None:
        self.assertEqual(match_request("Посмотри на экран", DEFAULT_TRIGGERS), (True, ""))

    def test_question_after_trigger(self) -> None:
        self.assertEqual(
            match_request("посмотри на экран, что за ошибка?", DEFAULT_TRIGGERS),
            (True, "что за ошибка"),
        )

    def test_service_words_are_stripped(self) -> None:
        self.assertEqual(
            match_request("посмотри на экран и скажи какая ошибка", DEFAULT_TRIGGERS),
            (True, "какая ошибка"),
        )

    def test_question_without_verb(self) -> None:
        self.assertEqual(match_request("что на экране", DEFAULT_TRIGGERS), (True, ""))

    def test_filler_before_question(self) -> None:
        self.assertEqual(match_request("а что на экране?", DEFAULT_TRIGGERS), (True, ""))

    def test_merged_verb_and_question(self) -> None:
        self.assertEqual(
            match_request("посмотри что на экране", DEFAULT_TRIGGERS),
            (True, ""),
        )

    def test_trigger_only_at_start(self) -> None:
        self.assertEqual(
            match_request("включи видео на экране", DEFAULT_TRIGGERS),
            (False, ""),
        )
        self.assertEqual(match_request("экран", DEFAULT_TRIGGERS), (False, ""))

    def test_short_question_is_not_enough(self) -> None:
        self.assertEqual(match_request("что ты видишь", DEFAULT_TRIGGERS), (False, ""))

    def test_empty_triggers(self) -> None:
        self.assertEqual(match_request("посмотри на экран", []), (False, ""))


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.available_result = True

    def available(self) -> bool:
        return self.available_result

    def answer_image(self, image: bytes, prompt: str, **kwargs) -> str:
        self.calls.append({"image": image, "prompt": prompt, **kwargs})
        return "На экране открыт браузер"


class ScreenVisionTest(unittest.TestCase):
    def make(self, **section) -> tuple[ScreenVision, FakeLLM]:
        cfg = {
            "vision": {
                "enabled": True,
                "model": "openai/gpt-6-luna",
                "max_width": 1280,
                "jpeg_quality": 75,
                "max_tokens": 2000,
                "timeout_sec": 60,
                "prompt": "Опиши экран",
                "triggers": ["посмотри на экран"],
                **section,
            }
        }
        llm = FakeLLM()
        vision = ScreenVision(cfg, llm)  # type: ignore[arg-type]
        vision._capture = lambda: b"jpeg-bytes"  # type: ignore[method-assign]
        return vision, llm

    def test_default_prompt(self) -> None:
        vision, llm = self.make()
        self.assertEqual(vision.describe(), "На экране открыт браузер")
        call = llm.calls[0]
        self.assertEqual(call["image"], b"jpeg-bytes")
        self.assertEqual(call["prompt"], "Опиши экран")
        self.assertEqual(call["model"], "openai/gpt-6-luna")
        self.assertEqual(call["max_tokens"], 2000)
        self.assertEqual(call["timeout"], 60)

    def test_request_becomes_question(self) -> None:
        vision, llm = self.make()
        vision.describe("что за ошибка")
        self.assertIn("что за ошибка", llm.calls[0]["prompt"])
        self.assertIn("по картинке", llm.calls[0]["prompt"])

    def test_disabled(self) -> None:
        vision, _ = self.make(enabled=False)
        with self.assertRaises(VisionError):
            vision.describe()
        self.assertFalse(vision.available())

    def test_missing_model(self) -> None:
        vision, _ = self.make(model="")
        with self.assertRaises(VisionError):
            vision.describe()

    def test_unavailable_llm(self) -> None:
        vision, llm = self.make()
        llm.available_result = False
        self.assertFalse(vision.available())


class AnswerImageTest(unittest.TestCase):
    def make_llm(self, **section) -> CloudLLM:
        cfg = {
            "cloud_llm": {
                "enabled": True,
                "api_key": "test-key",
                "model": "openai/gpt-6-luna",
                "retries": 0,
                **section,
            }
        }
        return CloudLLM(cfg)

    def test_payload_has_data_url(self) -> None:
        llm = self.make_llm()
        with patch.object(llm, "_post", return_value="ответ") as post:
            result = llm.answer_image(b"jpeg", "что тут", model="openai/gpt-6-luna")
        self.assertEqual(result, "ответ")
        model, messages = post.call_args.args[:2]
        self.assertEqual(model, "openai/gpt-6-luna")
        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "что тут"})
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_retryable_error_is_repeated(self) -> None:
        llm = self.make_llm(retries=1)
        error = LLMError("HTTP 429", retryable=True)
        with (
            patch.object(llm, "_post", side_effect=[error, "ответ"]) as post,
            patch("app.cloud_llm.time.sleep"),
        ):
            self.assertEqual(llm.answer_image(b"jpeg", "что тут"), "ответ")
        self.assertEqual(post.call_count, 2)

    def test_missing_key(self) -> None:
        llm = self.make_llm(api_key="")
        with self.assertRaises(LLMError):
            llm.answer_image(b"jpeg", "что тут")

    def test_empty_image(self) -> None:
        llm = self.make_llm()
        with self.assertRaises(LLMError):
            llm.answer_image(b"", "что тут")


if __name__ == "__main__":
    unittest.main()
