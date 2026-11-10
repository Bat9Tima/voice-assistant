from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from app.actions import CommandResult
from app.local_llm import LocalLLMError, LocalPlan
from app.main import Assistant


class FakeRouter:
    def __init__(self, result: CommandResult | None = None) -> None:
        self.result = result
        self.seen: list[str] = []
        self.plan_result = CommandResult(True, True, "Выполнено")
        self.plans: list[tuple[str, dict]] = []

    def handle(self, text: str) -> CommandResult | None:
        self.seen.append(text)
        return self.result

    def execute_plan(self, action: str, args: dict) -> CommandResult:
        self.plans.append((action, args))
        return self.plan_result


class FakeBrain:
    def __init__(self, answer: str = "ответ агента") -> None:
        self.answer = answer
        self.asked: list[str] = []

    def ask(self, text: str) -> str:
        self.asked.append(text)
        return self.answer


class FakeLocalLLM:
    def __init__(
        self, plan: LocalPlan | None = None, error: Exception | None = None
    ) -> None:
        self.plan_result = plan
        self.error = error
        self.seen: list[str] = []

    def plan(self, text: str) -> LocalPlan | None:
        self.seen.append(text)
        if self.error is not None:
            raise self.error
        return self.plan_result


class FakeTray:
    def __init__(self, status: dict[str, Any]) -> None:
        self._status = status

    def status(self) -> dict[str, Any]:
        return self._status


class TrayStatusTest(unittest.TestCase):
    @staticmethod
    def make(tray: Any) -> Assistant:
        assistant = Assistant.__new__(Assistant)
        assistant.tray = tray
        assistant.cfg = {"wake_word": "хей луна", "hotkey": "ctrl+alt+space"}
        assistant.stt = SimpleNamespace(loaded=False)
        assistant.wake = None
        assistant._state_lock = threading.Lock()
        assistant._state = "starting"
        assistant._paused = False
        return assistant

    def test_status_exposes_tray_state(self) -> None:
        self.assertIsNone(self.make(None).status()["tray"])

        tray_status = {
            "available": True,
            "running": True,
            "error": None,
            "restarts": 2,
        }
        status = self.make(FakeTray(tray_status)).status()
        self.assertEqual(
            set(status["tray"]), {"available", "running", "error", "restarts"}
        )
        self.assertEqual(status["tray"], tray_status)


class AgentTriggerRoutingTest(unittest.TestCase):
    def make(self, result: CommandResult | None = None) -> Assistant:
        assistant = Assistant.__new__(Assistant)
        assistant.router = FakeRouter(result)
        assistant.brain = FakeBrain()
        assistant.local_llm = FakeLocalLLM()
        assistant.agent_triggers = ["спроси агента", "передай агенту", "агент"]
        assistant.notifications = []
        assistant.states = []
        assistant.notify = lambda message, title="Ассистент": assistant.notifications.append(
            message
        )
        assistant.set_state = lambda state: assistant.states.append(state)
        return assistant

    def test_unknown_command_does_not_call_agent(self) -> None:
        assistant = self.make()
        with patch("app.main.beep") as beep:
            message = assistant.run_text("сделай что-нибудь странное")
        self.assertEqual(message, "Не понял команду")
        self.assertEqual(assistant.brain.asked, [])
        self.assertEqual(assistant.router.seen, ["сделай что нибудь странное"])
        beep.assert_called_once_with(320, 120)

    def test_explicit_agent_call(self) -> None:
        assistant = self.make()
        message = assistant.run_text("Спроси агента, какая погода?")
        self.assertEqual(message, "ответ агента")
        self.assertEqual(assistant.brain.asked, ["какая погода"])
        self.assertEqual(assistant.router.seen, [])

    def test_agent_trigger_without_request(self) -> None:
        assistant = self.make()
        with patch("app.main.beep") as beep:
            message = assistant.run_text("агент")
        self.assertEqual(message, "Что спросить у агента?")
        self.assertEqual(assistant.brain.asked, [])
        beep.assert_not_called()

    def test_router_command_with_word_agent_is_not_trigger(self) -> None:
        assistant = self.make(CommandResult(True, True, "Открываю YouTube"))
        message = assistant.run_text("включи фильм агент 007")
        self.assertEqual(message, "Открываю YouTube")
        self.assertEqual(assistant.brain.asked, [])
        self.assertEqual(assistant.router.seen, ["включи фильм агент 007"])

    def test_canonicalizes_text_before_routing(self) -> None:
        assistant = self.make(CommandResult(True, True, "Открываю Google"))
        message = assistant.run_text("открой гугель")
        self.assertEqual(message, "Открываю Google")
        self.assertEqual(assistant.router.seen, ["открой гугл"])

    def test_free_form_command_uses_local_llm(self) -> None:
        assistant = self.make()
        assistant.local_llm = FakeLocalLLM(
            LocalPlan("youtube", {"query": "lo-fi", "index": 1})
        )
        assistant.router.plan_result = CommandResult(
            True, True, "Включаю на YouTube: lo-fi"
        )
        message = assistant.run_text("вруби лофай на ютубе")
        self.assertEqual(message, "Включаю на YouTube: lo-fi")
        self.assertEqual(assistant.local_llm.seen, ["вруби лоуфай на ютубе"])
        self.assertEqual(
            assistant.router.plans,
            [("youtube", {"query": "lo-fi", "index": 1})],
        )

    def test_local_llm_none_falls_back(self) -> None:
        assistant = self.make()
        with patch("app.main.beep") as beep:
            message = assistant.run_text("сделай что-нибудь странное")
        self.assertEqual(message, "Не понял команду")
        self.assertEqual(
            assistant.local_llm.seen, ["сделай что нибудь странное"]
        )
        beep.assert_called_once_with(320, 120)

    def test_router_command_skips_local_llm(self) -> None:
        assistant = self.make(CommandResult(True, True, "Открываю YouTube"))
        assistant.run_text("открой ютуб")
        self.assertEqual(assistant.local_llm.seen, [])

    def test_local_llm_error_falls_back(self) -> None:
        assistant = self.make()
        assistant.local_llm = FakeLocalLLM(error=LocalLLMError("нет сервера"))
        with patch("app.main.beep") as beep:
            message = assistant.run_text("вруби музыку")
        self.assertEqual(message, "Не понял команду")
        beep.assert_called_once_with(320, 120)

    def test_agent_trigger_skips_local_llm(self) -> None:
        assistant = self.make()
        assistant.run_text("спроси агента, какая погода")
        self.assertEqual(assistant.local_llm.seen, [])


if __name__ == "__main__":
    unittest.main()
