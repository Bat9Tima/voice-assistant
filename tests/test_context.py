from __future__ import annotations

import unittest

from app.context import CommandContext


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class CommandContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock()
        self.context = CommandContext(ttl_sec=60, clock=self.clock)

    def test_remembers_search_and_media_independently(self) -> None:
        self.context.remember("search", "youtube_search", site="youtube", query="котики", index=1)
        self.context.remember("media", "pause", site="youtube")
        self.assertEqual(self.context.get("search").action, "youtube_search")
        self.assertEqual(self.context.get("search").slots["query"], "котики")
        self.assertEqual(self.context.get("media").action, "pause")
        self.assertTrue(self.context.matches("search"))
        self.assertTrue(self.context.matches("media"))
        self.assertFalse(self.context.matches("results"))

    def test_new_entry_replaces_only_its_kind(self) -> None:
        self.context.remember("search", "google_search", query="погода", index=1)
        self.context.remember("volume", "system_volume", target="system")
        self.context.remember("search", "youtube_search", query="музыка", index=1)
        self.assertEqual(self.context.get("search").slots["query"], "музыка")
        self.assertEqual(self.context.get("volume").slots["target"], "system")

    def test_entry_expires_by_ttl(self) -> None:
        self.context.remember("search", "google_search", query="погода", index=1)
        self.clock.advance(59)
        self.assertIsNotNone(self.context.get("search"))
        self.clock.advance(2)
        self.assertIsNone(self.context.get("search"))
        self.assertFalse(self.context.matches("search"))

    def test_clear_removes_all_kinds(self) -> None:
        self.context.remember("search", "google_search", query="погода", index=1)
        self.context.remember("volume", "system_volume", target="system")
        self.context.clear()
        self.assertIsNone(self.context.get("search"))
        self.assertIsNone(self.context.get("volume"))

    def test_unknown_kind_fails(self) -> None:
        with self.assertRaises(ValueError):
            self.context.remember("unknown", "action")


if __name__ == "__main__":
    unittest.main()
