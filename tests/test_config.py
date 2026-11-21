from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.config import load_config


class LlmPresetTest(unittest.TestCase):
    def load(self, **overrides) -> dict:
        data = {"llm": {"preset": "gpt"}}
        data.update(overrides)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return load_config(path)

    def test_default_preset_is_gpt(self) -> None:
        cfg = self.load()
        self.assertEqual(cfg["cloud_llm"]["model"], "openai/gpt-6-luna")
        self.assertEqual(cfg["dialog"]["addressee"]["model"], "openai/gpt-6-luna-decisions")
        self.assertEqual(cfg["intent"]["model"], "openai/gpt-6-luna-decisions")
        self.assertIsNone(cfg["cloud_llm"]["model_params"]["openai/gpt-6-luna"]["temperature"])

    def test_grok_preset_switches_chat_model(self) -> None:
        cfg = self.load(llm={"preset": "grok"})
        self.assertEqual(cfg["cloud_llm"]["model"], "x-ai/grok-4.3")
        self.assertEqual(cfg["cloud_llm"]["base_url"], "https://api.proxyapi.ru/v1")
        self.assertEqual(cfg["dialog"]["addressee"]["model"], "openai/gpt-6-luna-decisions")

    def test_preset_overrides_explicit_model(self) -> None:
        cfg = self.load(llm={"preset": "grok"}, cloud_llm={"model": "custom/other"})
        self.assertEqual(cfg["cloud_llm"]["model"], "x-ai/grok-4.3")

    def test_unknown_preset_keeps_defaults(self) -> None:
        with self.assertLogs("app.config", level="WARNING"):
            cfg = self.load(llm={"preset": "nope"})
        self.assertEqual(cfg["cloud_llm"]["model"], "openai/gpt-6-luna")

    def test_empty_preset_skips_overlay(self) -> None:
        cfg = self.load(llm={"preset": ""}, cloud_llm={"model": "custom/other"})
        self.assertEqual(cfg["cloud_llm"]["model"], "custom/other")


if __name__ == "__main__":
    unittest.main()
