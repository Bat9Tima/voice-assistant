from __future__ import annotations

import unittest

from app.browser import BrowserController


def item(index: int, title: str) -> dict:
    return {"index": index, "title": title, "playable": True}


class SoundCloudLikesMatchTest(unittest.TestCase):
    def test_finds_russian_substring(self) -> None:
        items = [
            item(0, "Скука - uncle pecos"),
            item(1, "Билборды - Скучно (Hardstyle)"),
        ]
        found = BrowserController._find_soundcloud_like(items, "скучно")
        self.assertIsNotNone(found)
        self.assertEqual(found["index"], 1)

    def test_matches_transliterated_title(self) -> None:
        items = [
            item(0, "Liza Evans - Ревную"),
            item(1, "EVA EVA"),
        ]
        found = BrowserController._find_soundcloud_like(items, "ева ева")
        self.assertIsNotNone(found)
        self.assertEqual(found["index"], 1)

    def test_matches_stt_ending(self) -> None:
        items = [
            item(0, "Скука - uncle pecos"),
            item(1, "Скучно"),
        ]
        found = BrowserController._find_soundcloud_like(items, "скучна")
        self.assertIsNotNone(found)
        self.assertEqual(found["index"], 1)

    def test_matches_two_fuzzy_words(self) -> None:
        items = [
            item(0, "билборды - острые лезвия"),
            item(1, "Билборды - Скучно (Hardstyle)"),
        ]
        found = BrowserController._find_soundcloud_like(items, "билборды скучна")
        self.assertIsNotNone(found)
        self.assertEqual(found["index"], 1)

    def test_does_not_match_unrelated(self) -> None:
        items = [item(0, "Liza Evans - Ревную")]
        self.assertIsNone(
            BrowserController._find_soundcloud_like(items, "зима холода")
        )

    def test_filter_queries_include_prefix_fallback(self) -> None:
        queries = BrowserController._soundcloud_filter_queries("скучна")
        self.assertIn("скучна", queries)
        self.assertIn("скуч", queries)
