from __future__ import annotations

import unittest

from app.actions import ActionRouter
from app.config import DEFAULTS, load_config


class FakeBrowser:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, action: str, timeout: float = 90, **args):
        self.calls.append((action, args))
        return {
            "ok": True,
            "data": {
                "url": "https://example.com",
                "title": "Example",
                "tabs": [{"index": 1, "title": "Example", "url": "https://example.com"}],
            },
        }

    @property
    def last(self) -> tuple[str, dict]:
        return self.calls[-1]


class RoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        cfg = load_config()
        self.browser = FakeBrowser()
        self.router = ActionRouter(cfg, self.browser)

    def test_youtube_query(self) -> None:
        result = self.router.handle("включи на ютубе Rick Astley Never Gonna Give You Up")
        self.assertIsNotNone(result)
        self.assertTrue(result.ok)
        self.assertEqual(self.browser.last[0], "youtube")
        self.assertEqual(
            self.browser.last[1]["query"], "rick astley never gonna give you up"
        )

    def test_open_youtube_without_query(self) -> None:
        result = self.router.handle("открой ютуб")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.last[0], "goto")
        self.assertEqual(self.browser.last[1]["url"], "https://www.youtube.com")

    def test_video_phrase(self) -> None:
        self.router.handle("включи видео lo-fi hip hop")
        self.assertEqual(self.browser.last[0], "youtube")
        self.assertEqual(self.browser.last[1]["query"], "lo fi hip hop")

    def test_search(self) -> None:
        self.router.handle("загугли погода москва")
        self.assertEqual(self.browser.last[0], "google")
        self.assertEqual(self.browser.last[1]["query"], "погода москва")

    def test_find_on_youtube(self) -> None:
        self.router.handle("найди на ютубе котиков")
        self.assertEqual(self.browser.last[0], "youtube")
        self.assertEqual(self.browser.last[1]["query"], "котиков")

    def test_tab(self) -> None:
        self.router.handle("переключись на вкладку два")
        self.assertEqual(self.browser.last[0], "tab")
        self.assertEqual(self.browser.last[1]["index"], 2)

    def test_pause(self) -> None:
        self.router.handle("поставь на паузу")
        self.assertEqual(self.browser.last[0], "pause")

    def test_plain_question_goes_to_llm(self) -> None:
        self.assertIsNone(self.router.handle("сколько будет два плюс два"))

    def test_site_with_domain(self) -> None:
        self.router.handle("открой example.com")
        self.assertEqual(self.browser.last[0], "goto")
        self.assertEqual(self.browser.last[1]["url"], "https://example.com")

    def test_unknown_open_becomes_google(self) -> None:
        self.router.handle("открой расписание электричек")
        self.assertEqual(self.browser.last[0], "google")


class ConfigTest(unittest.TestCase):
    def test_defaults_present(self) -> None:
        cfg = load_config()
        self.assertIn(cfg["wake_word"], cfg["wake_aliases"])
        self.assertTrue(cfg["browser"]["profile_dir"])
        self.assertEqual(DEFAULTS["wake_word"], "хей луна")
        self.assertFalse(DEFAULTS["llm"]["enabled"])
        self.assertIn("11434", DEFAULTS["llm"]["base_url"])
        self.assertEqual(DEFAULTS["llm"]["model"], "qwen2.5:3b-instruct")


if __name__ == "__main__":
    unittest.main()
