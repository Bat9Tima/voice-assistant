from __future__ import annotations

import unittest
from unittest.mock import patch

from app.actions import CommandResult
from app.main import Assistant


class FakeRouter:
    def __init__(self, result: CommandResult | None = None) -> None:
        self.result = result
        self.seen: list[str] = []

    def handle(self, text: str) -> CommandResult | None:
        self.seen.append(text)
        return self.result


class FakeBrain:
    def __init__(self, answer: str = "ответ агента") -> None:
        self.answer = answer
        self.asked: list[str] = []

    def ask(self, text: str) -> str:
        self.asked.append(text)
        return self.answer


class AgentTriggerRoutingTest(unittest.TestCase):
    def make(self, result: CommandResult | None = None) -> Assistant:
        assistant = Assistant.__new__(Assistant)
        assistant.router = FakeRouter(result)
        assistant.brain = FakeBrain()
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


if __name__ == "__main__":
    unittest.main()
