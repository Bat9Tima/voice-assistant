from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.brain import OpenCodeBrain, OpenCodeError
from app.config import ROOT


def make_brain(
    model: str = "opencode/deepseek-v4.1-flash-free",
) -> OpenCodeBrain:
    brain = OpenCodeBrain.__new__(OpenCodeBrain)
    brain.cfg = {}
    brain.agent = "voice"
    brain.model = model
    brain.standalone = False
    brain.timeout = 5
    brain.cli = "opencode"
    brain.signature = ""
    brain._save_session = lambda session_id: None
    return brain


class BrainCommandTest(unittest.TestCase):
    def test_command_includes_model(self) -> None:
        cmd = make_brain()._command("привет", "ses_1")
        self.assertIn("--model", cmd)
        self.assertEqual(
            cmd[cmd.index("--model") + 1],
            "opencode/deepseek-v4.1-flash-free",
        )

    def test_command_omits_empty_model(self) -> None:
        cmd = make_brain(model="")._command("привет", None)
        self.assertNotIn("--model", cmd)

    def test_command_includes_agent_and_session(self) -> None:
        cmd = make_brain()._command("привет", "ses_1")
        self.assertEqual(cmd[cmd.index("--agent") + 1], "voice")
        self.assertEqual(cmd[cmd.index("--session") + 1], "ses_1")
        self.assertEqual(cmd[-1], "привет")


class BrainRunTest(unittest.TestCase):
    def test_text_event_is_returned(self) -> None:
        stdout = (
            '{"type":"text","sessionID":"ses_1","part":'
            '{"type":"text","text":"Четыре."}}\n'
        )
        completed = subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")
        with patch("app.brain.subprocess.run", return_value=completed):
            self.assertEqual(make_brain()._run("два плюс два", None), "Четыре.")

    def test_error_event_is_reported(self) -> None:
        stdout = (
            '{"type":"error","sessionID":"ses_1","error":'
            '{"type":"provider.auth","message":'
            '"This model is not available in your country"}}\n'
        )
        completed = subprocess.CompletedProcess([], 1, stdout=stdout, stderr="")
        with patch("app.brain.subprocess.run", return_value=completed):
            with self.assertRaises(OpenCodeError) as ctx:
                make_brain()._run("привет", None)
        self.assertIn("not available in your country", str(ctx.exception))

    def test_missing_session_returns_none(self) -> None:
        completed = subprocess.CompletedProcess(
            [], 1, stdout="", stderr="session not found"
        )
        with patch("app.brain.subprocess.run", return_value=completed):
            self.assertIsNone(make_brain()._run("привет", "ses_1"))

    def test_location_not_found_returns_none(self) -> None:
        stdout = (
            '{"type":"error","sessionID":"ses_1","error":'
            '{"type":"unknown","message":'
            '"Location not found: C:\\\\old"}}\n'
        )
        completed = subprocess.CompletedProcess([], 1, stdout=stdout, stderr="")
        with patch("app.brain.subprocess.run", return_value=completed):
            self.assertIsNone(make_brain()._run("привет", "ses_1"))

    def test_location_not_found_without_session_is_reported(self) -> None:
        stdout = (
            '{"type":"error","sessionID":"ses_1","error":'
            '{"type":"unknown","message":'
            '"Location not found: C:\\\\old"}}\n'
        )
        completed = subprocess.CompletedProcess([], 1, stdout=stdout, stderr="")
        with patch("app.brain.subprocess.run", return_value=completed):
            with self.assertRaises(OpenCodeError):
                make_brain()._run("привет", None)


class BrainSessionTest(unittest.TestCase):
    def _brain(self, signature: str) -> OpenCodeBrain:
        brain = OpenCodeBrain.__new__(OpenCodeBrain)
        brain.cfg = {}
        brain.agent = "voice"
        brain.model = ""
        brain.signature = signature
        brain.session_id = None
        return brain

    def test_prompt_change_drops_old_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            path.write_text(
                json.dumps({"session_id": "ses_1", "signature": "old"}),
                encoding="utf-8",
            )
            with patch("app.brain.SESSION_FILE", path):
                self.assertIsNone(self._brain("new")._load_session())
                self.assertEqual(self._brain("old")._load_session(), "ses_1")

    def test_session_without_signature_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            path.write_text(json.dumps({"session_id": "ses_1"}), encoding="utf-8")
            with patch("app.brain.SESSION_FILE", path):
                self.assertIsNone(self._brain("new")._load_session())
                # Если промпт недоступен, отпечатка нет и сессия сохраняется.
                self.assertEqual(self._brain("")._load_session(), "ses_1")

    def test_other_project_root_drops_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            path.write_text(
                json.dumps(
                    {"session_id": "ses_1", "signature": "abc", "root": "C:\\old"}
                ),
                encoding="utf-8",
            )
            with patch("app.brain.SESSION_FILE", path), patch(
                "app.brain.ROOT", Path(tmp) / "new"
            ):
                self.assertIsNone(self._brain("abc")._load_session())

    def test_saved_session_keeps_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            with patch("app.brain.SESSION_FILE", path):
                brain = self._brain("abc")
                brain._save_session("ses_2")
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(data["session_id"], "ses_2")
                self.assertEqual(data["signature"], "abc")
                self.assertEqual(data["root"], str(ROOT))


if __name__ == "__main__":
    unittest.main()
