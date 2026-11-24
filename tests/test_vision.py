from __future__ import annotations

import base64
import io
import unittest
from unittest.mock import patch

from PIL import Image

from app.cloud_llm import CloudLLM, LLMError
from app.screen_capture import Frame
from app.vision import (
    DEFAULT_TRIGGERS,
    ScreenVision,
    VisionError,
    match_followup,
    match_request,
)


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

    def test_observation_phrases(self) -> None:
        for phrase in (
            "что сейчас на экране",
            "что происходит на экране",
            "что сейчас происходит на экране",
            "опиши экран",
            "опиши что на экране",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(match_request(phrase, DEFAULT_TRIGGERS), (True, ""))


class FollowupMatchTest(unittest.TestCase):
    def test_examples(self) -> None:
        for phrase in (
            "где здесь кнопка",
            "что тут написано",
            "что написано",
            "где кнопка",
            "что это",
            "что за ошибка",
            "покажи где кнопка",
            "прочитай что тут написано",
            "объясни что здесь",
            "тут что",
            "а здесь ссылка",
            "а что в левом верхнем углу",
            "что в правом нижнем углу",
            "что слева",
            "что на фоне",
            "какого цвета кнопка",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(match_followup(phrase))

    def test_general_questions_are_not_followups(self) -> None:
        for phrase in (
            "какая погода",
            "что нового",
            "включи видео на экране",
            "открой это",
            "нажми на кнопку",
            "посмотри на экран",
            "перемести окно в левый верхний угол",
            "что в центре города",
            "",
        ):
            with self.subTest(phrase=phrase):
                self.assertFalse(match_followup(phrase))


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.available_result = True

    def available(self) -> bool:
        return self.available_result

    def answer_image(self, image: bytes, prompt: str, **kwargs) -> str:
        self.calls.append({"image": image, "prompt": prompt, **kwargs})
        return "На экране открыт браузер"

    def answer_images(self, images: list[bytes], prompt: str, **kwargs) -> str:
        self.calls.append({"images": list(images), "prompt": prompt, **kwargs})
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

    def test_request_includes_dialog_context(self) -> None:
        vision, llm = self.make()
        vision.snapshot()
        vision.describe_pending(
            "найди область определения",
            context="Пользователь: что ты номер 353 видишь\nЛуна: Номер 353",
        )
        prompt = llm.calls[0]["prompt"]
        self.assertIn("найди область определения", prompt)
        self.assertIn("Контекст разговора", prompt)
        self.assertIn("номер 353", prompt)

    def test_default_prompt_includes_dialog_context(self) -> None:
        vision, llm = self.make()
        vision.describe(context="Пользователь: покажи пример")
        prompt = llm.calls[0]["prompt"]
        self.assertEqual(prompt, "Опиши экран Контекст разговора: Пользователь: покажи пример")

    def test_snapshot_then_pending_question(self) -> None:
        vision, llm = self.make()
        vision.snapshot()
        self.assertTrue(vision.has_pending())
        self.assertEqual(
            vision.describe_pending("где здесь кнопка"),
            "На экране открыт браузер",
        )
        call = llm.calls[0]
        self.assertEqual(call["image"], b"jpeg-bytes")
        self.assertIn("где здесь кнопка", call["prompt"])

    def test_describe_keeps_frame_for_followup(self) -> None:
        vision, llm = self.make()
        vision.describe()
        self.assertTrue(vision.has_pending())
        vision.describe_pending("что тут написано")
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[0]["image"], llm.calls[1]["image"])

    def test_pending_question_without_snapshot(self) -> None:
        vision, _ = self.make()
        with self.assertRaises(VisionError):
            vision.describe_pending("что тут")

    def test_pending_expires(self) -> None:
        vision, _ = self.make()
        vision.snapshot()
        vision._pending_at -= vision.followup_ttl + 1
        self.assertFalse(vision.has_pending())
        with self.assertRaises(VisionError):
            vision.describe_pending("что тут")

    def test_capture_does_not_require_model(self) -> None:
        vision, _ = self.make(model="")
        self.assertEqual(vision.capture(), b"jpeg-bytes")
        self.assertEqual(vision.pending_image(), b"jpeg-bytes")

    def test_pending_image_expires(self) -> None:
        vision, _ = self.make()
        vision.capture()
        vision._pending_at -= vision.followup_ttl + 1
        self.assertIsNone(vision.pending_image())

    def test_snapshot_still_requires_llm(self) -> None:
        vision, llm = self.make()
        llm.available_result = False
        with self.assertRaises(VisionError):
            vision.snapshot()

    @staticmethod
    def frame(width: int = 64, height: int = 36) -> Frame:
        return Frame(
            image=Image.new("RGB", (width, height), "white"),
            captured_at=1.0,
        )

    def test_describe_frame_uses_existing_frame(self) -> None:
        vision, llm = self.make()
        vision._capture = lambda: self.fail("повторный захват")  # type: ignore[method-assign]

        answer = vision.describe_frame(self.frame(), "что за ошибка")

        self.assertEqual(answer, "На экране открыт браузер")
        call = llm.calls[0]
        self.assertTrue(call["image"].startswith(b"\xff\xd8"))
        self.assertIn("что за ошибка", call["prompt"])

    def test_describe_frame_keeps_frame_for_followup(self) -> None:
        vision, llm = self.make()
        vision.describe_frame(self.frame(), "что тут написано")
        self.assertTrue(vision.has_pending())

        vision.describe_pending("а это что")

        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[0]["image"], llm.calls[1]["image"])

    def test_remember_frame_does_not_call_model(self) -> None:
        vision, llm = self.make()

        image = vision.remember_frame(self.frame())

        self.assertTrue(image.startswith(b"\xff\xd8"))
        self.assertEqual(llm.calls, [])
        self.assertTrue(vision.has_pending())

    def test_remember_frame_resizes_like_capture(self) -> None:
        vision, _ = self.make(max_width=32)

        image = vision.remember_frame(self.frame(width=64, height=36))

        with Image.open(io.BytesIO(image)) as decoded:
            self.assertEqual(decoded.size, (32, 18))

    def test_describe_frame_requires_model(self) -> None:
        vision, _ = self.make(model="")
        with self.assertRaises(VisionError):
            vision.describe_frame(self.frame())

    def test_describe_frame_disabled(self) -> None:
        vision, _ = self.make(enabled=False)
        with self.assertRaises(VisionError):
            vision.describe_frame(self.frame())

    def test_describe_frames_sends_several_images(self) -> None:
        vision, llm = self.make()
        frames = [self.frame(), self.frame(), self.frame()]

        answer = vision.describe_frames(frames, "что изменилось")

        self.assertEqual(answer, "На экране открыт браузер")
        call = llm.calls[0]
        self.assertEqual(len(call["images"]), 3)
        for image in call["images"]:
            self.assertTrue(image.startswith(b"\xff\xd8"))
        self.assertIn("что изменилось", call["prompt"])
        self.assertIn("ключевых кадров", call["prompt"])

    def test_describe_frames_limits_to_three(self) -> None:
        vision, llm = self.make()

        vision.describe_frames([self.frame() for _ in range(4)], "сравни")

        self.assertEqual(len(llm.calls[0]["images"]), 3)

    def test_describe_frames_includes_dialog_context(self) -> None:
        vision, llm = self.make()

        vision.describe_frames(
            [self.frame(), self.frame()],
            "сравни",
            context="Пользователь: что тут происходит",
        )

        self.assertIn("Контекст разговора", llm.calls[0]["prompt"])
        self.assertIn("что тут происходит", llm.calls[0]["prompt"])

    def test_describe_frames_rejects_empty(self) -> None:
        vision, llm = self.make()

        with self.assertRaises(VisionError):
            vision.describe_frames([])

        self.assertEqual(llm.calls, [])

    def test_describe_frames_requires_model(self) -> None:
        vision, _ = self.make(model="")

        with self.assertRaises(VisionError):
            vision.describe_frames([self.frame(), self.frame()])

    def test_snapshot_only_for_imperative(self) -> None:
        vision, _ = self.make(triggers=list(DEFAULT_TRIGGERS))
        self.assertTrue(vision.wants_snapshot("посмотри на экран"))
        self.assertFalse(vision.wants_snapshot("что на экране"))
        self.assertFalse(vision.wants_snapshot("посмотри что на экране"))
        self.assertFalse(vision.wants_snapshot("посмотри на экран, что за ошибка"))

    def test_match_followup_method(self) -> None:
        vision, _ = self.make()
        self.assertTrue(vision.match_followup("где здесь кнопка"))
        self.assertFalse(vision.match_followup("какая погода"))

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

    def test_payload_has_all_data_urls(self) -> None:
        llm = self.make_llm()

        with patch.object(llm, "_post", return_value="ответ") as post:
            result = llm.answer_images([b"one", b"two"], "сравни кадры")

        self.assertEqual(result, "ответ")
        content = post.call_args.args[1][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "сравни кадры"})
        self.assertEqual(len(content), 3)
        for item, expected in zip(content[1:], (b"one", b"two"), strict=True):
            url = item["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))
            self.assertEqual(base64.b64decode(url.split(",", 1)[1]), expected)

    def test_empty_image_list_is_rejected(self) -> None:
        llm = self.make_llm()

        with self.assertRaises(LLMError):
            llm.answer_images([], "что тут")

    def test_any_empty_image_is_rejected(self) -> None:
        llm = self.make_llm()

        with self.assertRaises(LLMError):
            llm.answer_images([b"one", b""], "что тут")


if __name__ == "__main__":
    unittest.main()
