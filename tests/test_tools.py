from __future__ import annotations

import unittest

from app.actions import SITE_URLS
from app.local_llm import (
    REGISTRY,
    SITE_KEYS,
    SYSTEM_PROMPT,
    LocalLLM,
    LocalPlan,
    _build_plan,
)
from app.tools import Tool, ToolParam, ToolRegistry, build_default_registry


class ToolParamTest(unittest.TestCase):
    def test_hint_for_enum_is_sorted(self) -> None:
        param = ToolParam(
            "direction", "string", enum=frozenset({"up", "down"})
        )
        self.assertEqual(param.prompt_hint(), '"down|up"')

    def test_hint_for_integer_range(self) -> None:
        param = ToolParam("index", "integer", minimum=1, maximum=20)
        self.assertEqual(param.prompt_hint(), "1..20")

    def test_hint_for_integer_without_range(self) -> None:
        self.assertEqual(ToolParam("count", "integer").prompt_hint(), "1")

    def test_hint_for_boolean(self) -> None:
        self.assertEqual(ToolParam("play", "boolean").prompt_hint(), "true")

    def test_custom_hint_wins(self) -> None:
        param = ToolParam("url", "string", hint='"https://..."')
        self.assertEqual(param.prompt_hint(), '"https://..."')


class ToolPromptTest(unittest.TestCase):
    def test_tool_without_params(self) -> None:
        self.assertEqual(
            Tool("pause", "пауза.").prompt_line(),
            "- pause — пауза.",
        )

    def test_tool_with_params(self) -> None:
        tool = Tool(
            "tab",
            "переключить вкладку.",
            (
                ToolParam(
                    "index",
                    "integer",
                    required=True,
                    minimum=1,
                    maximum=99,
                ),
            ),
        )
        self.assertEqual(
            tool.prompt_line(),
            '- tab: {"index": 1..99} — переключить вкладку.',
        )


class ToolRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry(
            [
                Tool("pause", "пауза."),
                Tool(
                    "search",
                    "поиск.",
                    (
                        ToolParam(
                            "query",
                            "string",
                            required=True,
                            max_length=10,
                        ),
                        ToolParam("limit", "integer", default=5, minimum=1),
                        ToolParam("safe", "boolean", default=True),
                    ),
                ),
            ]
        )

    def test_get_and_names(self) -> None:
        self.assertIsNotNone(self.registry.get("pause"))
        self.assertIsNone(self.registry.get("missing"))
        self.assertEqual(self.registry.names(), ("pause", "search"))
        self.assertEqual(len(self.registry.tools()), 2)

    def test_register_replaces_tool(self) -> None:
        self.registry.register(Tool("pause", "новая пауза."))
        tool = self.registry.get("pause")
        self.assertIsNotNone(tool)
        assert tool is not None
        self.assertEqual(tool.description, "новая пауза.")

    def test_unknown_action(self) -> None:
        self.assertIsNone(self.registry.validate("launch", {}))

    def test_tool_without_params(self) -> None:
        self.assertEqual(self.registry.validate("pause", {"x": 1}), {})

    def test_required_argument(self) -> None:
        self.assertIsNone(self.registry.validate("search", {}))
        self.assertIsNone(self.registry.validate("search", {"query": "  "}))

    def test_defaults_are_filled(self) -> None:
        self.assertEqual(
            self.registry.validate("search", {"query": "кот"}),
            {"query": "кот", "limit": 5, "safe": True},
        )

    def test_extra_arguments_are_dropped(self) -> None:
        self.assertEqual(
            self.registry.validate(
                "search", {"query": "кот", "evil": "rm -rf"}
            ),
            {"query": "кот", "limit": 5, "safe": True},
        )

    def test_string_is_trimmed(self) -> None:
        self.assertEqual(
            self.registry.validate("search", {"query": "  два   слова "}),
            {"query": "два слова", "limit": 5, "safe": True},
        )

    def test_string_max_length(self) -> None:
        self.assertIsNone(
            self.registry.validate("search", {"query": "x" * 11})
        )

    def test_integer_from_string(self) -> None:
        self.assertEqual(
            self.registry.validate("search", {"query": "кот", "limit": "7"}),
            {"query": "кот", "limit": 7, "safe": True},
        )

    def test_integer_rejects_garbage(self) -> None:
        self.assertIsNone(
            self.registry.validate("search", {"query": "кот", "limit": "много"})
        )

    def test_integer_rejects_bool(self) -> None:
        self.assertIsNone(
            self.registry.validate("search", {"query": "кот", "limit": True})
        )

    def test_integer_below_minimum(self) -> None:
        self.assertIsNone(
            self.registry.validate("search", {"query": "кот", "limit": 0})
        )

    def test_boolean_parsing(self) -> None:
        for value in (False, "false", "нет", 0, "no"):
            with self.subTest(value=value):
                result = self.registry.validate(
                    "search", {"query": "кот", "safe": value}
                )
                self.assertIsNotNone(result)
                self.assertFalse(result["safe"])

    def test_boolean_rejects_unknown_text(self) -> None:
        self.assertIsNone(
            self.registry.validate(
                "search", {"query": "кот", "safe": "может быть"}
            )
        )

    def test_enum_and_lower(self) -> None:
        registry = ToolRegistry(
            [
                Tool(
                    "scroll",
                    "прокрутка.",
                    (
                        ToolParam(
                            "direction",
                            "string",
                            required=True,
                            enum=frozenset({"up", "down"}),
                            lower=True,
                        ),
                    ),
                )
            ]
        )
        self.assertEqual(
            registry.validate("scroll", {"direction": "UP"}),
            {"direction": "up"},
        )
        self.assertIsNone(registry.validate("scroll", {"direction": "left"}))

    def test_clamp(self) -> None:
        registry = ToolRegistry(
            [
                Tool(
                    "youtube",
                    "видео.",
                    (
                        ToolParam(
                            "index",
                            "integer",
                            default=1,
                            minimum=1,
                            maximum=20,
                            clamp=True,
                        ),
                    ),
                )
            ]
        )
        self.assertEqual(registry.validate("youtube", {"index": 0}), {"index": 1})
        self.assertEqual(
            registry.validate("youtube", {"index": 100}), {"index": 20}
        )

    def test_custom_validator(self) -> None:
        def validate_url(args: dict) -> dict | None:
            url = str(args.get("url") or "")
            return args if url.startswith("https://") else None

        registry = ToolRegistry(
            [
                Tool(
                    "goto",
                    "адрес.",
                    (ToolParam("url", "string", required=True),),
                    validate=validate_url,
                )
            ]
        )
        self.assertEqual(
            registry.validate("goto", {"url": "https://example.com"}),
            {"url": "https://example.com"},
        )
        self.assertIsNone(
            registry.validate("goto", {"url": "javascript:alert(1)"})
        )


class DefaultRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_default_registry(SITE_URLS, SITE_KEYS)

    def test_expected_tools_exist(self) -> None:
        expected = {
            "open_site",
            "goto",
            "youtube",
            "soundcloud",
            "soundcloud_likes",
            "google",
            "play",
            "pause",
            "stop",
            "seek",
            "next",
            "prev",
            "newtab",
            "closetab",
            "tab",
            "back",
            "forward",
            "reload",
            "scroll",
            "volume",
            "volume_up",
            "volume_down",
            "mute",
            "unmute",
            "status",
        }
        self.assertEqual(set(self.registry.names()), expected)

    def test_open_site(self) -> None:
        self.assertEqual(
            self.registry.validate("open_site", {"site": "ЮТУБ"}),
            {"site": "ютуб"},
        )
        self.assertIsNone(
            self.registry.validate("open_site", {"site": "неизвестный сайт"})
        )

    def test_goto(self) -> None:
        self.assertEqual(
            self.registry.validate("goto", {"url": "https://example.com"}),
            {"url": "https://example.com"},
        )
        self.assertIsNone(
            self.registry.validate("goto", {"url": "javascript:alert(1)"})
        )
        self.assertIsNone(
            self.registry.validate("goto", {"url": "https://exa mple.com"})
        )

    def test_soundcloud_likes_defaults(self) -> None:
        self.assertEqual(
            self.registry.validate("soundcloud_likes", {}),
            {"query": "", "play": True},
        )
        self.assertEqual(
            self.registry.validate(
                "soundcloud_likes", {"query": " отпусти меня ", "play": "false"}
            ),
            {"query": "отпусти меня", "play": False},
        )

    def test_media_site_optional(self) -> None:
        for name in ("play", "pause", "stop"):
            self.assertEqual(self.registry.validate(name, {}), {}, name)
            self.assertEqual(
                self.registry.validate(name, {"site": "YouTube"}),
                {"site": "youtube"},
                name,
            )
            self.assertEqual(
                self.registry.validate(name, {"site": " soundcloud "}),
                {"site": "soundcloud"},
                name,
            )
            self.assertIsNone(
                self.registry.validate(name, {"site": "vk"}), name
            )

    def test_seek_defaults(self) -> None:
        self.assertEqual(
            self.registry.validate("seek", {}),
            {"seconds": 10, "direction": "forward"},
        )
        self.assertEqual(
            self.registry.validate(
                "seek", {"seconds": 30, "direction": "backward"}
            ),
            {"seconds": 30, "direction": "backward"},
        )
        self.assertEqual(
            self.registry.validate(
                "seek", {"seconds": "7", "site": "YouTube"}
            ),
            {"seconds": 7, "direction": "forward", "site": "youtube"},
        )

    def test_seek_rejects_bad_values(self) -> None:
        self.assertIsNone(self.registry.validate("seek", {"seconds": 0}))
        self.assertIsNone(self.registry.validate("seek", {"seconds": 3601}))
        self.assertIsNone(self.registry.validate("seek", {"seconds": "много"}))
        self.assertIsNone(
            self.registry.validate("seek", {"direction": "sideways"})
        )

    def test_volume_range(self) -> None:
        self.assertEqual(
            self.registry.validate("volume", {"level": 30}), {"level": 30}
        )
        self.assertIsNone(self.registry.validate("volume", {"level": 101}))
        self.assertIsNone(self.registry.validate("volume", {"level": -1}))

    def test_tab_requires_index(self) -> None:
        self.assertIsNone(self.registry.validate("tab", {}))
        self.assertEqual(self.registry.validate("tab", {"index": 2}), {"index": 2})
        self.assertIsNone(self.registry.validate("tab", {"index": 0}))

    def test_youtube_clamps_index(self) -> None:
        self.assertEqual(
            self.registry.validate("youtube", {"query": "lo-fi"}),
            {"query": "lo-fi", "index": 1},
        )
        self.assertEqual(
            self.registry.validate(
                "youtube", {"query": "lo-fi", "index": 100}
            ),
            {"query": "lo-fi", "index": 20},
        )


class PromptFromRegistryTest(unittest.TestCase):
    def test_prompt_mentions_every_tool(self) -> None:
        for name in REGISTRY.names():
            with self.subTest(name=name):
                self.assertIn(f"- {name}", SYSTEM_PROMPT)

    def test_prompt_mentions_none(self) -> None:
        self.assertIn("- none", SYSTEM_PROMPT)

    def test_prompt_lists_site_keys(self) -> None:
        self.assertIn("ютуб", SYSTEM_PROMPT)
        self.assertIn("саундклауд", SYSTEM_PROMPT)

    def test_prompt_describes_arguments(self) -> None:
        self.assertIn('"query": "..."', SYSTEM_PROMPT)
        self.assertIn('"play": true', SYSTEM_PROMPT)
        self.assertIn('"index": 1..99', SYSTEM_PROMPT)

    def test_custom_registry_changes_prompt_and_plan(self) -> None:
        registry = ToolRegistry([Tool("ping", "проверка.")])
        llm = LocalLLM(
            {"llm": {"enabled": True, "model": "test"}}, registry=registry
        )
        self.assertIn("- ping", llm.system_prompt)
        self.assertNotIn("- youtube", llm.system_prompt)
        self.assertEqual(
            _build_plan({"action": "ping"}, registry),
            LocalPlan("ping", {}),
        )
        self.assertIsNone(_build_plan({"action": "youtube", "query": "x"}, registry))


if __name__ == "__main__":
    unittest.main()
