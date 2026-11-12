from __future__ import annotations

import inspect
import json
import re
import unittest

from app import browser_cli
from app.actions import SITE_URLS, ActionRouter
from app.catalog import (
    ACTIONS,
    action_names,
    catalog_by_cli,
    catalog_by_tool,
    validate_catalog,
)
from app.config import ROOT
from app.local_planner import SITE_KEYS
from app.tools import build_default_registry


def load_intents() -> list[dict]:
    data = json.loads((ROOT / "intents.json").read_text(encoding="utf-8-sig"))
    return list(data["intents"])


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = text.replace("«", "").replace("»", "").replace("**", "")
    return re.sub(r"\s+", " ", text)


class CatalogIntegrityTest(unittest.TestCase):
    def test_catalog_has_no_problems(self) -> None:
        self.assertEqual(validate_catalog(), [])

    def test_action_names_are_unique(self) -> None:
        names = action_names()
        self.assertEqual(len(names), len(set(names)))

    def test_every_intent_action_has_spec(self) -> None:
        intents = load_intents()
        catalog = set(action_names())
        for intent in intents:
            with self.subTest(intent=intent["id"]):
                self.assertIn(intent["action"], catalog)

    def test_every_intent_is_mapped_once(self) -> None:
        intents = load_intents()
        intent_ids = {intent["id"] for intent in intents}
        covered = [name for action in ACTIONS for name in action.intents]
        self.assertEqual(set(covered), intent_ids)
        self.assertEqual(len(covered), len(set(covered)))

    def test_every_spec_intent_exists(self) -> None:
        intent_ids = {intent["id"] for intent in load_intents()}
        for action in ACTIONS:
            for intent_id in action.intents:
                with self.subTest(action=action.name, intent=intent_id):
                    self.assertIn(intent_id, intent_ids)

    def test_every_intent_handler_has_spec(self) -> None:
        handlers = {
            name.removeprefix("_intent_")
            for name, _ in inspect.getmembers(
                ActionRouter, predicate=inspect.isfunction
            )
            if name.startswith("_intent_")
        }
        intent_actions = {intent["action"] for intent in load_intents()}
        self.assertEqual(handlers - set(action_names()), set())
        self.assertEqual(intent_actions - handlers, set())
        self.assertEqual(len(handlers), len(intent_actions))


class CatalogToolCoverageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_default_registry(SITE_URLS, SITE_KEYS)

    def test_every_tool_has_spec(self) -> None:
        self.assertEqual(
            set(self.registry.names()), set(catalog_by_tool())
        )

    def test_every_spec_tool_exists(self) -> None:
        for action in ACTIONS:
            if action.tool is None:
                continue
            with self.subTest(action=action.name):
                self.assertIsNotNone(self.registry.get(action.tool))


class CatalogCliCoverageTest(unittest.TestCase):
    def test_cli_commands_match_catalog(self) -> None:
        source = inspect.getsource(browser_cli.main)
        pattern = re.compile(
            r'action\s+(?:in\s+\{([^}]*)\}|==\s+"([^"]+)")'
        )
        commands: set[str] = set()
        for block, single in pattern.findall(source):
            commands.update(re.findall(r'"([^"]+)"', block))
            if single:
                commands.add(single)
        self.assertEqual(commands, set(catalog_by_cli()))

    def test_every_cli_command_has_action(self) -> None:
        for command, actions in catalog_by_cli().items():
            with self.subTest(command=command):
                self.assertTrue(actions)


class CatalogReadmeCoverageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.readme = normalize(
            (ROOT / "README.md").read_text(encoding="utf-8-sig")
        )

    def test_readme_covers_every_documented_action(self) -> None:
        missing = [
            action.name
            for action in ACTIONS
            if action.documented
            and normalize(action.doc_example) not in self.readme
        ]
        self.assertEqual(missing, [])

    def test_documented_actions_have_examples(self) -> None:
        for action in ACTIONS:
            if not action.documented:
                continue
            with self.subTest(action=action.name):
                self.assertTrue(action.doc_example)
                self.assertTrue(action.examples)


if __name__ == "__main__":
    unittest.main()
