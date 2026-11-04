from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app.actions import ActionRouter
from app.dictionary import Dictionary, get_dictionary, reset_dictionary
from app.text import canonicalize

DATA = {
    "version": 1,
    "terms": [
        {
            "id": "google",
            "kind": "site",
            "canonical": "гугл",
            "aliases": ["гугель", {"text": "google", "replace": False}],
        },
        {
            "id": "maps",
            "kind": "site",
            "canonical": "карты",
            "aliases": ["яндекс карты", "карты"],
        },
    ],
}


class DictionaryApplyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dictionary = Dictionary.from_data(DATA)

    def test_replaces_safe_alias(self) -> None:
        self.assertEqual(self.dictionary.apply("открой гугель"), "открой гугл")

    def test_keeps_alias_without_replace(self) -> None:
        self.assertEqual(self.dictionary.apply("открой google"), "открой google")

    def test_longest_alias_wins(self) -> None:
        self.assertEqual(
            self.dictionary.apply("открой яндекс  карты"),
            "открой карты",
        )

    def test_respects_word_boundaries(self) -> None:
        self.assertEqual(
            self.dictionary.apply("гугельчик и яндекс карточки"),
            "гугельчик и яндекс карточки",
        )

    def test_find_returns_terms_and_positions(self) -> None:
        matches = self.dictionary.find("открой гугель и яндекс карты")
        self.assertEqual([item.term.id for item in matches], ["google", "maps"])
        self.assertEqual(matches[0].text, "гугель")
        self.assertEqual(matches[0].start, 7)
        self.assertEqual(matches[1].text, "яндекс карты")
        self.assertEqual(matches[1].start, 16)

    def test_find_includes_non_replaced_alias(self) -> None:
        matches = self.dictionary.find("открой google")
        self.assertEqual([item.term.id for item in matches], ["google"])


class DictionaryValidationTest(unittest.TestCase):
    def test_duplicate_alias_with_other_canonical_fails(self) -> None:
        data = {
            "version": 1,
            "terms": [
                {"id": "one", "canonical": "один", "aliases": ["слово"]},
                {"id": "two", "canonical": "два", "aliases": ["слово"]},
            ],
        }
        with self.assertRaises(ValueError):
            Dictionary.from_data(data)

    def test_missing_fields_fail(self) -> None:
        for data in (
            {"version": 1, "terms": [{"canonical": "x", "aliases": ["x"]}]},
            {"version": 1, "terms": [{"id": "x", "aliases": ["x"]}]},
            {"version": 1, "terms": [{"id": "x", "canonical": "x"}]},
        ):
            with self.assertRaises(ValueError):
                Dictionary.from_data(data)

    def test_unknown_version_fails(self) -> None:
        with self.assertRaises(ValueError):
            Dictionary.from_data({"version": 99, "terms": []})

    def test_missing_file_gives_empty_dictionary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dictionary = Dictionary.load(Path(tmp) / "missing.json")
        self.assertEqual(dictionary.terms, ())
        self.assertEqual(dictionary.apply("открой гугель"), "открой гугель")

    def test_broken_json_gives_empty_dictionary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dictionary.json"
            path.write_text("{ not json", encoding="utf-8")
            dictionary = Dictionary.load(path)
        self.assertEqual(dictionary.terms, ())

    def test_saved_dictionary_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dictionary.json"
            path.write_text(json.dumps(DATA, ensure_ascii=False), encoding="utf-8")
            dictionary = Dictionary.load(path)
        self.assertEqual(dictionary.apply("открой гугель"), "открой гугл")


class ShippedDictionaryTest(unittest.TestCase):
    def test_dictionary_has_terms(self) -> None:
        self.assertGreaterEqual(len(get_dictionary().terms), 20)

    def test_canonicalizes_site_variants(self) -> None:
        cases = {
            "открой гугель": "открой гугл",
            "открой ютьюб": "открой ютуб",
            "открой в контакте": "открой вконтакте",
            "открой телеграмм": "открой телеграм",
            "открой чат гпт": "открой чатгпт",
            "открой яндекс карты": "открой карты",
            "открой авиту": "открой авито",
            "открой вайлд беррис": "открой вайлдберриз",
        }
        for source, expected in cases.items():
            self.assertEqual(canonicalize(source), expected)

    def test_keeps_regular_phrases(self) -> None:
        cases = (
            "найди в гугле кота",
            "найди на ютубе лоуфай",
            "громкость пять",
            "включи песню про телегу",
            "открой вкладку",
        )
        for text in cases:
            self.assertEqual(canonicalize(text), text)

    def test_canonicalization_is_fast(self) -> None:
        reset_dictionary()
        text = "открой яндекс карты и включи на ютубе лоуфай"
        start = time.perf_counter()
        for _ in range(2000):
            canonicalize(text)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 0.5)


class FakeBrowser:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, action: str, timeout: float = 90, **args) -> dict:
        self.calls.append((action, args))
        return {"ok": True, "data": {}}


class RouterDictionaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.browser = FakeBrowser()
        self.router = ActionRouter({}, self.browser)

    def test_opens_site_from_asr_variant(self) -> None:
        cases = {
            "открой гугель": "google.com",
            "открой в контакте": "vk.com",
            "открой ютьюб": "youtube.com",
            "открой яндекс карты": "yandex.ru/maps",
            "открой чат гпт": "chatgpt.com",
            "открой гугл почту": "mail.google.com",
        }
        for command, domain in cases.items():
            self.browser.calls.clear()
            result = self.router.handle(command)
            self.assertIsNotNone(result, command)
            self.assertTrue(result.ok, command)
            action, args = self.browser.calls[-1]
            self.assertEqual(action, "goto", command)
            self.assertIn(domain, args["url"], command)


if __name__ == "__main__":
    unittest.main()
