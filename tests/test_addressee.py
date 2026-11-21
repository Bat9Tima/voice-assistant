from __future__ import annotations

import json
import unittest
import urllib.error
from unittest.mock import patch

from app.addressee import (
    ADDRESSED,
    BACKGROUND,
    CLOSING,
    MODE_ACTIVE,
    MODE_OFF,
    SOURCE_FALLBACK,
    SOURCE_MODEL,
    SOURCE_RULES,
    AddresseeGate,
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


def make_cfg(**addressee: object) -> dict:
    section = {
        "enabled": True,
        "mode": MODE_ACTIVE,
        "api_key": "test-key",
        "model": "openai/gpt-6-luna-decisions",
    }
    section.update(addressee)
    return {"dialog": {"addressee": section}}


def response_body(
    choice: str,
    confidence: float,
    probabilities: dict[str, float] | None = None,
) -> bytes:
    return json.dumps(
        {
            "id": "gen-addr-test",
            "model": "openai/gpt-6-luna-decisions-20260101",
            "answers": {
                "addressee": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": confidence,
                    "probabilities": probabilities or {choice: confidence},
                }
            },
            "usage": {"input_tokens": 180, "output_tokens": 0},
        },
        ensure_ascii=False,
    ).encode("utf-8")


class AddresseeRulesTest(unittest.TestCase):
    def test_name_is_addressed_without_network(self) -> None:
        gate = AddresseeGate(make_cfg())
        with patch("app.addressee.urllib.request.urlopen") as urlopen:
            decision = gate.classify("Луна, как дела?")
        urlopen.assert_not_called()
        self.assertEqual(decision.kind, ADDRESSED)
        self.assertEqual(decision.source, SOURCE_RULES)

    def test_closing_phrase_is_closing(self) -> None:
        gate = AddresseeGate(make_cfg())
        decision = gate.classify("Ну всё, спасибо.")
        self.assertEqual(decision.kind, CLOSING)
        self.assertEqual(decision.source, SOURCE_RULES)

    def test_background_marker_is_background(self) -> None:
        gate = AddresseeGate(make_cfg())
        decision = gate.classify("Я не тебе, я по телефону.")
        self.assertEqual(decision.kind, BACKGROUND)
        self.assertEqual(decision.source, SOURCE_RULES)

    def test_not_with_you_marker_is_background(self) -> None:
        gate = AddresseeGate(make_cfg())
        decision = gate.classify("Нет, я не про него, я вообще сейчас не с тобой говорил.")
        self.assertEqual(decision.kind, BACKGROUND)
        self.assertEqual(decision.source, SOURCE_RULES)

    def test_phone_call_is_background(self) -> None:
        gate = AddresseeGate(make_cfg())
        decision = gate.classify("Алло, да слушаю, да да, нормально.")
        self.assertEqual(decision.kind, BACKGROUND)
        self.assertEqual(decision.source, SOURCE_RULES)

    def test_other_name_with_second_person_is_background(self) -> None:
        gate = AddresseeGate(make_cfg(other_names=["максим", "макс"]))
        decision = gate.classify("Бля, Максим, ты заебал уже меня.")
        self.assertEqual(decision.kind, BACKGROUND)
        self.assertEqual(decision.source, SOURCE_RULES)

    def test_repeated_other_name_is_background(self) -> None:
        gate = AddresseeGate(make_cfg(other_names=["максимка"]))
        decision = gate.classify("Максимка, Максимка, ну ты где?")
        self.assertEqual(decision.kind, BACKGROUND)
        self.assertEqual(decision.source, SOURCE_RULES)

    def test_other_name_in_third_person_goes_to_model(self) -> None:
        gate = AddresseeGate(make_cfg(other_names=["максим"]))
        with patch(
            "app.addressee.urllib.request.urlopen",
            return_value=FakeResponse(response_body(ADDRESSED, 0.9)),
        ) as urlopen:
            decision = gate.classify("Максим вчера сказал, что погода будет хорошая.")
        urlopen.assert_called_once()
        self.assertEqual(decision.kind, ADDRESSED)

    def test_bare_all_is_not_closing(self) -> None:
        gate = AddresseeGate(make_cfg())
        with patch(
            "app.addressee.urllib.request.urlopen",
            return_value=FakeResponse(response_body(ADDRESSED, 0.9)),
        ):
            decision = gate.classify("все")
        self.assertEqual(decision.kind, ADDRESSED)

    def test_first_turn_is_addressed(self) -> None:
        gate = AddresseeGate(make_cfg())
        with patch("app.addressee.urllib.request.urlopen") as urlopen:
            decision = gate.classify("как дела", first_turn=True)
        urlopen.assert_not_called()
        self.assertEqual(decision.kind, ADDRESSED)
        self.assertEqual(decision.source, SOURCE_RULES)

    def test_first_turn_does_not_override_phone(self) -> None:
        gate = AddresseeGate(make_cfg())
        decision = gate.classify("Але, парни, че вы тут?", first_turn=True)
        self.assertEqual(decision.kind, BACKGROUND)
        self.assertEqual(decision.source, SOURCE_RULES)

    def test_first_turn_does_not_override_other_name(self) -> None:
        gate = AddresseeGate(make_cfg(other_names=["максим"]))
        decision = gate.classify("Максим, ты заебал", first_turn=True)
        self.assertEqual(decision.kind, BACKGROUND)
        self.assertEqual(decision.source, SOURCE_RULES)

    def test_off_mode_does_not_call_model(self) -> None:
        gate = AddresseeGate(make_cfg(mode=MODE_OFF))
        with patch("app.addressee.urllib.request.urlopen") as urlopen:
            decision = gate.classify("как дела")
        urlopen.assert_not_called()
        self.assertEqual(decision.kind, ADDRESSED)
        self.assertEqual(decision.source, SOURCE_FALLBACK)


class AddresseeHttpTest(unittest.TestCase):
    def test_payload_and_background_decision(self) -> None:
        captured: dict = {}

        def fake_urlopen(request, timeout=None):  # noqa: ANN001
            captured["url"] = request.full_url
            captured["auth"] = request.get_header("Authorization")
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse(response_body(BACKGROUND, 0.93))

        gate = AddresseeGate(make_cfg())
        with patch("app.addressee.urllib.request.urlopen", side_effect=fake_urlopen):
            decision = gate.classify("смотри какая скидка", last_question="Что смотрим?")

        self.assertEqual(decision.kind, BACKGROUND)
        self.assertEqual(decision.source, SOURCE_MODEL)
        self.assertEqual(captured["url"], "https://api.proxyapi.ru/v1/decisions")
        self.assertEqual(captured["auth"], "Bearer test-key")
        self.assertEqual(captured["timeout"], 2.0)
        self.assertEqual(captured["payload"]["questions"]["addressee"]["type"], "choice")
        self.assertEqual(captured["payload"]["state"]["utterance"], "смотри какая скидка")
        self.assertEqual(captured["payload"]["state"]["last_luna_question"], "Что смотрим?")

    def test_low_confidence_background_is_addressed(self) -> None:
        gate = AddresseeGate(make_cfg())
        with patch(
            "app.addressee.urllib.request.urlopen",
            return_value=FakeResponse(response_body(BACKGROUND, 0.41)),
        ):
            decision = gate.classify("что-то невнятное")
        self.assertEqual(decision.kind, ADDRESSED)
        self.assertEqual(decision.raw, BACKGROUND)
        self.assertTrue(decision.weak)

    def test_medium_confidence_background_is_background(self) -> None:
        gate = AddresseeGate(make_cfg())
        with patch(
            "app.addressee.urllib.request.urlopen",
            return_value=FakeResponse(response_body(BACKGROUND, 0.55)),
        ):
            decision = gate.classify("она на фоне играет, заебала")
        self.assertEqual(decision.kind, BACKGROUND)
        self.assertFalse(decision.weak)

    def test_borderline_background_is_addressed_weak(self) -> None:
        gate = AddresseeGate(make_cfg())
        with patch(
            "app.addressee.urllib.request.urlopen",
            return_value=FakeResponse(response_body(BACKGROUND, 0.49)),
        ):
            decision = gate.classify("как где")
        self.assertEqual(decision.kind, ADDRESSED)
        self.assertTrue(decision.weak)

    def test_uncertain_addressed_is_weak(self) -> None:
        gate = AddresseeGate(make_cfg())
        with patch(
            "app.addressee.urllib.request.urlopen",
            return_value=FakeResponse(response_body(ADDRESSED, 0.55)),
        ):
            decision = gate.classify("ну да, наверное")
        self.assertEqual(decision.kind, ADDRESSED)
        self.assertTrue(decision.weak)

    def test_confident_addressed_is_not_weak(self) -> None:
        gate = AddresseeGate(make_cfg())
        with patch(
            "app.addressee.urllib.request.urlopen",
            return_value=FakeResponse(response_body(ADDRESSED, 0.91)),
        ):
            decision = gate.classify("расскажи, как прошёл день")
        self.assertEqual(decision.kind, ADDRESSED)
        self.assertFalse(decision.weak)

    def test_network_error_is_addressed(self) -> None:
        gate = AddresseeGate(make_cfg())
        with patch(
            "app.addressee.urllib.request.urlopen",
            side_effect=urllib.error.URLError("offline"),
        ):
            decision = gate.classify("как дела")
        self.assertEqual(decision.kind, ADDRESSED)
        self.assertEqual(decision.source, SOURCE_FALLBACK)
        self.assertIn("offline", decision.error)

    def test_unknown_choice_is_addressed(self) -> None:
        gate = AddresseeGate(make_cfg())
        with patch(
            "app.addressee.urllib.request.urlopen",
            return_value=FakeResponse(response_body("noise", 0.9)),
        ):
            decision = gate.classify("непонятно что")
        self.assertEqual(decision.kind, ADDRESSED)
        self.assertEqual(decision.source, SOURCE_FALLBACK)
        self.assertIn("noise", decision.error)

    def test_malformed_body_is_addressed(self) -> None:
        gate = AddresseeGate(make_cfg())
        with patch(
            "app.addressee.urllib.request.urlopen",
            return_value=FakeResponse(b"not json"),
        ):
            decision = gate.classify("как дела")
        self.assertEqual(decision.kind, ADDRESSED)
        self.assertEqual(decision.source, SOURCE_FALLBACK)


if __name__ == "__main__":
    unittest.main()
