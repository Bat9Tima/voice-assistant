from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from app.intent_classifier import (
    ANSWER,
    DIALOG,
    INFO,
    MODE_ACTIVE,
    MODE_SHADOW,
    SOURCE_FALLBACK,
    SOURCE_MODEL,
    SOURCE_RULES,
    UNKNOWN,
    DialogState,
    IntentClassifier,
    IntentDecision,
)


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def make_cfg(**intent: object) -> dict:
    section = {
        "enabled": True,
        "mode": MODE_SHADOW,
        "api_key": "test-key",
        "model": "openai/gpt-6-luna-decisions",
    }
    section.update(intent)
    return {"intent": section}


def response_body(
    choice: str,
    confidence: float,
    probabilities: dict[str, float] | None = None,
) -> bytes:
    return json.dumps(
        {
            "id": "gen-dec-test",
            "model": "openai/gpt-6-luna-decisions-20260101",
            "answers": {
                "intent": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": confidence,
                    "probabilities": probabilities or {choice: confidence},
                }
            },
            "usage": {"input_tokens": 312, "output_tokens": 0},
        },
        ensure_ascii=False,
    ).encode("utf-8")


class DialogStateTest(unittest.TestCase):
    def test_keeps_last_question_from_reply(self) -> None:
        now = [100.0]
        state = DialogState(ttl_sec=120, clock=lambda: now[0])

        state.note_reply("Слушаю. Как ты себя чувствуешь?")

        self.assertTrue(state.awaiting_answer)
        self.assertEqual(state.question(), "Как ты себя чувствуешь?")

    def test_reply_without_question_clears_state(self) -> None:
        state = DialogState(ttl_sec=120, clock=lambda: 100.0)
        state.note_question("Как дела?")

        state.note_reply("Понятно, спасибо.")

        self.assertFalse(state.awaiting_answer)
        self.assertEqual(state.question(), "")

    def test_question_expires(self) -> None:
        now = [100.0]
        state = DialogState(ttl_sec=30, clock=lambda: now[0])
        state.note_question("Как дела?")

        now[0] += 31

        self.assertFalse(state.awaiting_answer)
        self.assertEqual(state.question(), "")

    def test_zero_ttl_disables_state(self) -> None:
        state = DialogState(ttl_sec=0, clock=lambda: 100.0)
        state.note_question("Как дела?")

        self.assertFalse(state.awaiting_answer)

    def test_note_answer_and_clear(self) -> None:
        state = DialogState(ttl_sec=120, clock=lambda: 100.0)
        state.note_question("Как дела?")
        state.note_answer()
        self.assertFalse(state.awaiting_answer)

        state.note_question("Что нового?")
        state.clear()
        self.assertFalse(state.awaiting_answer)


class IntentClassifierHttpTest(unittest.TestCase):
    def test_payload_and_decision(self) -> None:
        captured: dict = {}

        def fake_urlopen(request, timeout=None):  # noqa: ANN001
            captured["url"] = request.full_url
            captured["auth"] = request.get_header("Authorization")
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse(response_body(ANSWER, 0.99, {ANSWER: 0.99, DIALOG: 0.01}))

        classifier = IntentClassifier(make_cfg())
        classifier.state.note_question("Как ты себя чувствуешь?")

        with patch(
            "app.intent_classifier.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            decision = classifier.classify(
                "да говорил я такой ну вроде бы сейчас получше стало",
                in_dialog=True,
            )

        self.assertEqual(decision.intent, ANSWER)
        self.assertEqual(decision.raw, ANSWER)
        self.assertEqual(decision.confidence, 0.99)
        self.assertEqual(decision.source, SOURCE_MODEL)
        self.assertEqual(decision.input_tokens, 312)
        self.assertEqual(captured["url"], "https://api.proxyapi.ru/v1/decisions")
        self.assertEqual(captured["auth"], "Bearer test-key")
        self.assertEqual(captured["timeout"], 4.0)

        payload = captured["payload"]
        self.assertEqual(payload["model"], "openai/gpt-6-luna-decisions")
        self.assertEqual(
            payload["state"]["utterance"], ("да говорил я такой ну вроде бы сейчас получше стало")
        )
        self.assertTrue(payload["state"]["awaiting_answer"])
        self.assertEqual(
            payload["state"]["last_luna_question"],
            "Как ты себя чувствуешь?",
        )
        self.assertEqual(payload["questions"]["intent"]["type"], "choice")
        self.assertEqual(
            set(payload["questions"]["intent"]["criteria"]),
            {"command", "control", "answer", "dialog", "info", "clarify"},
        )

    def test_low_confidence_becomes_unknown(self) -> None:
        classifier = IntentClassifier(make_cfg(threshold=0.6))
        body = response_body(DIALOG, 0.41, {DIALOG: 0.53, ANSWER: 0.40})

        with patch(
            "app.intent_classifier.urllib.request.urlopen",
            return_value=FakeResponse(body),
        ):
            decision = classifier.classify("а да нормально", in_dialog=True)

        self.assertEqual(decision.intent, UNKNOWN)
        self.assertEqual(decision.raw, DIALOG)
        self.assertEqual(decision.confidence, 0.41)
        self.assertTrue(classifier.blocks_search(decision))

    def test_http_error_falls_back_and_keeps_search(self) -> None:
        error = urllib.error.HTTPError(
            "https://api.proxyapi.ru/v1/decisions",
            500,
            "Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b'{"detail": "boom"}'),
        )
        classifier = IntentClassifier(make_cfg())

        with patch(
            "app.intent_classifier.urllib.request.urlopen",
            side_effect=error,
        ):
            decision = classifier.classify("мне новое хобби", in_dialog=True)

        self.assertEqual(decision.intent, UNKNOWN)
        self.assertEqual(decision.source, SOURCE_FALLBACK)
        self.assertIn("HTTP 500", decision.error)
        self.assertFalse(classifier.blocks_search(decision))

    def test_broken_json_falls_back(self) -> None:
        classifier = IntentClassifier(make_cfg())

        with patch(
            "app.intent_classifier.urllib.request.urlopen",
            return_value=FakeResponse(b"not json"),
        ):
            decision = classifier.classify("мне новое хобби", in_dialog=True)

        self.assertEqual(decision.source, SOURCE_FALLBACK)
        self.assertFalse(classifier.blocks_search(decision))

    def test_explicit_search_skips_network(self) -> None:
        classifier = IntentClassifier(make_cfg())

        with patch(
            "app.intent_classifier.urllib.request.urlopen",
        ) as urlopen:
            decision = classifier.classify("поищи рецепт борща", in_dialog=True)

        self.assertEqual(decision.intent, INFO)
        self.assertEqual(decision.source, SOURCE_RULES)
        urlopen.assert_not_called()
        self.assertFalse(classifier.blocks_search(decision))

    def test_missing_key_disables_network(self) -> None:
        classifier = IntentClassifier({"intent": {"enabled": True, "mode": "shadow"}})

        self.assertFalse(classifier.available())
        with patch("app.intent_classifier.urllib.request.urlopen") as urlopen:
            decision = classifier.classify("привет", in_dialog=True)
        self.assertEqual(decision.intent, UNKNOWN)
        self.assertEqual(decision.source, SOURCE_FALLBACK)
        urlopen.assert_not_called()


class IntentClassifierRulesTest(unittest.TestCase):
    def test_blocks_only_model_conversation(self) -> None:
        classifier = IntentClassifier(make_cfg())

        def decision(intent: str, source: str) -> IntentDecision:
            return IntentDecision(intent=intent, source=source)

        for intent in (ANSWER, DIALOG, "clarify", UNKNOWN):
            with self.subTest(intent=intent):
                self.assertTrue(classifier.blocks_search(decision(intent, SOURCE_MODEL)))
        for intent in (INFO, "command", "control"):
            with self.subTest(intent=intent):
                self.assertFalse(classifier.blocks_search(decision(intent, SOURCE_MODEL)))
        self.assertFalse(classifier.blocks_search(decision(ANSWER, SOURCE_RULES)))
        self.assertFalse(classifier.blocks_search(decision(UNKNOWN, SOURCE_FALLBACK)))

    def test_mode_typo_does_not_enable_active(self) -> None:
        classifier = IntentClassifier(make_cfg(mode="activ"))
        self.assertEqual(classifier.mode, MODE_SHADOW)

    def test_active_mode_is_read_from_config(self) -> None:
        classifier = IntentClassifier(make_cfg(mode=MODE_ACTIVE))
        self.assertEqual(classifier.mode, MODE_ACTIVE)

    def test_key_falls_back_to_cloud_llm(self) -> None:
        classifier = IntentClassifier(
            {
                "intent": {"enabled": True, "mode": MODE_SHADOW},
                "cloud_llm": {"api_key": "cloud-key"},
            }
        )
        self.assertEqual(classifier.api_key, "cloud-key")
        self.assertTrue(classifier.available())

    def test_key_falls_back_to_tts(self) -> None:
        classifier = IntentClassifier(
            {
                "intent": {"enabled": True, "mode": MODE_SHADOW},
                "tts": {"api_key": "tts-key"},
            }
        )
        self.assertEqual(classifier.api_key, "tts-key")

    def test_observe_does_nothing_without_key(self) -> None:
        classifier = IntentClassifier({"intent": {"enabled": True, "mode": "shadow"}})
        with patch("app.intent_classifier.threading.Thread") as thread:
            classifier.observe("привет", in_dialog=True)
        thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()
