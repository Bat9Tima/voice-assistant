from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.context import CommandContext
from app.intents import IntentSet


def volume_percent(raw: str) -> int | None:
    try:
        return int(raw)
    except ValueError:
        return None


def make_set(intents: list[dict], context: CommandContext | None = None):
    data = {"version": 1, "intents": intents}
    return IntentSet.from_data(
        data,
        site_names=["ютуб", "гугл", "яндекс музыка"],
        volume_parser=volume_percent,
    )


class IntentMatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intents = make_set(
            [
                {
                    "id": "open_site",
                    "priority": 50,
                    "patterns": ["открой {site}", "open {site}"],
                    "slots": {"site": "site"},
                    "action": "open_site",
                },
                {
                    "id": "google_search",
                    "priority": 40,
                    "patterns": ["найди {query}"],
                    "slots": {"query": "query"},
                    "action": "google_search",
                },
                {
                    "id": "switch_tab",
                    "priority": 40,
                    "patterns": ["вкладка {tab}"],
                    "slots": {"tab": "tab"},
                    "action": "switch_tab",
                },
                {
                    "id": "open_result",
                    "priority": 70,
                    "patterns": ["открой {index} ссылку"],
                    "slots": {"index": "ordinal"},
                    "action": "open_result",
                },
                {
                    "id": "open_saved_result",
                    "priority": 80,
                    "requires": "results",
                    "patterns": ["{index}"],
                    "slots": {"index": "ordinal_word"},
                    "action": "open_result",
                },
                {
                    "id": "volume_set",
                    "priority": 40,
                    "patterns": ["громкость {level}"],
                    "slots": {"level": "volume"},
                    "action": "volume_set",
                },
            ]
        )

    def test_matches_site(self) -> None:
        match = self.intents.match("открой ютуб")
        self.assertEqual(match.intent.id, "open_site")
        self.assertEqual(match.slots["site"], "ютуб")

    def test_matches_multiword_site(self) -> None:
        match = self.intents.match("открой яндекс музыка")
        self.assertEqual(match.slots["site"], "яндекс музыка")

    def test_unknown_site_does_not_match(self) -> None:
        self.assertIsNone(self.intents.match("открой кота"))

    def test_slot_query_and_original_text(self) -> None:
        match = self.intents.match("найди дешевые мониторы")
        self.assertEqual(match.intent.id, "google_search")
        self.assertEqual(match.slots["query"], "дешевые мониторы")

    def test_tab_number_words(self) -> None:
        self.assertEqual(self.intents.match("вкладка два").slots["tab"], 2)
        self.assertEqual(self.intents.match("вкладка 3").slots["tab"], 3)
        self.assertIsNone(self.intents.match("вкладка 11"))

    def test_ordinal_words_and_digits(self) -> None:
        self.assertEqual(self.intents.match("открой вторую ссылку").slots["index"], 2)
        self.assertEqual(self.intents.match("открой 3 ссылку").slots["index"], 3)

    def test_volume_slot_uses_parser(self) -> None:
        self.assertEqual(self.intents.match("громкость 40").slots["level"], 40)
        self.assertIsNone(self.intents.match("громкость максимум"))

    def test_context_requirement(self) -> None:
        self.assertIsNone(self.intents.match("вторую"))
        context = CommandContext(ttl_sec=60)
        context.remember("results", "google_search", index=1)
        match = self.intents.match("вторую", context)
        self.assertEqual(match.intent.id, "open_saved_result")
        self.assertEqual(match.slots["index"], 2)

    def test_context_ordinal_word_rejects_digits(self) -> None:
        context = CommandContext(ttl_sec=60)
        context.remember("results", "google_search", index=1)
        self.assertIsNone(self.intents.match("2", context))


class IntentPriorityTest(unittest.TestCase):
    def test_higher_priority_wins(self) -> None:
        intents = make_set(
            [
                {
                    "id": "low",
                    "priority": 10,
                    "patterns": ["повтори"],
                    "action": "low",
                },
                {
                    "id": "high",
                    "priority": 90,
                    "patterns": ["повтори"],
                    "action": "high",
                },
            ]
        )
        self.assertEqual(intents.match("повтори").intent.id, "high")

    def test_priority_order_is_stable_for_equal_values(self) -> None:
        intents = make_set(
            [
                {"id": "first", "patterns": ["тест"], "action": "first"},
                {"id": "second", "patterns": ["тест"], "action": "second"},
            ]
        )
        self.assertEqual(intents.match("тест").intent.id, "first")


class IntentValidationTest(unittest.TestCase):
    def test_duplicate_ids_fail(self) -> None:
        with self.assertRaises(ValueError):
            make_set(
                [
                    {"id": "one", "patterns": ["раз"], "action": "one"},
                    {"id": "one", "patterns": ["два"], "action": "two"},
                ]
            )

    def test_unknown_slot_type_fails(self) -> None:
        with self.assertRaises(ValueError):
            make_set(
                [
                    {
                        "id": "bad",
                        "patterns": ["открой {site}"],
                        "slots": {"site": "unknown"},
                        "action": "bad",
                    }
                ]
            )

    def test_placeholder_without_slot_spec_fails(self) -> None:
        with self.assertRaises(ValueError):
            make_set(
                [
                    {
                        "id": "bad",
                        "patterns": ["открой {site}"],
                        "action": "bad",
                    }
                ]
            )

    def test_unknown_context_fails(self) -> None:
        with self.assertRaises(ValueError):
            make_set(
                [
                    {
                        "id": "bad",
                        "requires": "unknown",
                        "patterns": ["повтори"],
                        "action": "bad",
                    }
                ]
            )

    def test_missing_patterns_fails(self) -> None:
        with self.assertRaises(ValueError):
            make_set([{"id": "bad", "action": "bad"}])

    def test_unknown_version_fails(self) -> None:
        with self.assertRaises(ValueError):
            IntentSet.from_data({"version": 99, "intents": []})

    def test_missing_file_gives_empty_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            intents = IntentSet.load(Path(tmp) / "missing.json")
        self.assertEqual(intents.intents, ())
        self.assertIsNone(intents.match("открой ютуб"))

    def test_broken_json_gives_empty_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intents.json"
            path.write_text("{ not json", encoding="utf-8")
            intents = IntentSet.load(path)
        self.assertEqual(intents.intents, ())

    def test_saved_intents_load(self) -> None:
        data = {
            "version": 1,
            "intents": [
                {
                    "id": "open_site",
                    "patterns": ["открой {site}"],
                    "slots": {"site": "site"},
                    "action": "open_site",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intents.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            intents = IntentSet.load(path, site_names=["ютуб"])
        match = intents.match("открой ютуб")
        self.assertIsNotNone(match)
        self.assertEqual(match.intent.action, "open_site")


class ShippedIntentsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intents = IntentSet.load(
            site_names=["ютуб", "гугл", "саундклауд"],
            volume_parser=volume_percent,
        )

    def test_shipped_intents_exist(self) -> None:
        self.assertGreaterEqual(len(self.intents.intents), 20)

    def test_common_commands_match(self) -> None:
        cases = {
            "открой ютуб": "open_site",
            "найди на ютубе котиков": "youtube_search",
            "загугли погода москва": "google_search",
            "переключись на вкладку два": "switch_tab",
            "открой вторую ссылку": "open_result",
            "поставь на паузу": "pause",
            "стоп": "stop",
            "перемотай на 30 секунд": "seek",
            "перемотай назад на 10 секунд": "seek_back",
            "прокрути вниз": "scroll_down",
            "что играет": "status",
        }
        for phrase, intent_id in cases.items():
            with self.subTest(phrase=phrase):
                match = self.intents.match(phrase)
                self.assertIsNotNone(match)
                self.assertEqual(match.intent.id, intent_id)

    def test_youtube_beats_generic_search(self) -> None:
        match = self.intents.match("найди на ютубе котиков")
        self.assertEqual(match.intent.id, "youtube_search")

    def test_context_intents_need_context(self) -> None:
        context_only = {
            "еще один": "next_result",
            "повтори": "repeat_search",
            "сделай тише": "volume_down_context",
            "сделай громче": "volume_up_context",
        }
        for phrase, intent_id in context_only.items():
            with self.subTest(phrase=phrase):
                match = self.intents.match(phrase)
                self.assertTrue(match is None or match.intent.id != intent_id)

    def test_bare_volume_is_system_intent(self) -> None:
        self.assertEqual(self.intents.match("тише").intent.id, "system_volume_down")
        self.assertEqual(self.intents.match("громче").intent.id, "system_volume_up")


class WordOrderTest(unittest.TestCase):
    """Перестановка слов не должна менять интент."""

    def setUp(self) -> None:
        self.intents = IntentSet.load(
            site_names=["ютуб", "гугл", "саундклауд"],
            volume_parser=volume_percent,
        )

    def assert_match(self, phrase: str, intent_id: str, **slots) -> None:
        match = self.intents.match(phrase)
        self.assertIsNotNone(match, phrase)
        assert match is not None
        self.assertEqual(match.intent.id, intent_id)
        for name, value in slots.items():
            self.assertEqual(match.slots[name], value)

    def test_likes_keep_priority_over_generic_search(self) -> None:
        self.assert_match(
            "включи скучно в лайкнутых",
            "soundcloud_likes_play",
            query="скучно",
        )
        self.assert_match(
            "включи в лайкнутых скучно",
            "soundcloud_likes_play",
            query="скучно",
        )
        self.assert_match(
            "включи скучно в лайкнутых на саундклауде",
            "soundcloud_likes_play",
        )
        self.assert_match(
            "включи в лайкнутых на саундклауде скучно",
            "soundcloud_likes_play",
        )

    def test_likes_query_keeps_repeated_preposition(self) -> None:
        self.assert_match(
            "включи песню в лесу в лайкнутых",
            "soundcloud_likes_play",
            query="песню в лесу",
        )
        self.assert_match(
            "включи в лайкнутых песню в лесу",
            "soundcloud_likes_play",
            query="песню в лесу",
        )

    def test_youtube_search_in_any_order(self) -> None:
        self.assert_match(
            "на ютубе найди котиков",
            "youtube_search",
            query="котиков",
        )
        self.assert_match(
            "котиков найди на ютубе",
            "youtube_search",
            query="котиков",
        )
        self.assert_match(
            "найди на ютубе котиков",
            "youtube_search",
            query="котиков",
        )

    def test_soundcloud_search_in_any_order(self) -> None:
        self.assert_match(
            "на саундклауде включи скучно",
            "soundcloud_search",
            query="скучно",
        )
        self.assert_match(
            "включи скучно на саундклауде",
            "soundcloud_search",
            query="скучно",
        )

    def test_site_in_any_order(self) -> None:
        self.assert_match("ютуб открой", "open_site", site="ютуб")
        self.assert_match("открой ютуб", "open_site", site="ютуб")

    def test_volume_in_any_order(self) -> None:
        self.assert_match(
            "десять громкость саундклауда",
            "soundcloud_volume_set",
            level=10,
        )
        self.assert_match(
            "громкость саундклауда десять",
            "soundcloud_volume_set",
            level=10,
        )

    def test_result_and_tab_in_any_order(self) -> None:
        self.assert_match(
            "вторую открой ссылку",
            "open_result",
            index=2,
        )
        self.assert_match(
            "на вторую вкладку переключись",
            "switch_tab",
            tab=2,
        )
        self.assert_match(
            "на вторую переключись вкладку",
            "switch_tab",
            tab=2,
        )

    def test_stop_with_site_in_any_order(self) -> None:
        self.assert_match("стоп на ютубе", "stop", site="youtube")
        self.assert_match("на ютубе стоп", "stop", site="youtube")
        self.assert_match("останови трек на саундклауде", "stop", site="soundcloud")
        self.assert_match("стоп видео", "stop")

    def test_seek_seconds_and_direction(self) -> None:
        self.assert_match("перемотай на 30 секунд", "seek", seconds=30)
        self.assert_match("перемотай 30 секунд", "seek", seconds=30)
        self.assert_match("перемотай", "seek")
        self.assert_match("перемотай назад", "seek_back")
        self.assert_match("перемотай назад на 20 секунд", "seek_back", seconds=20)
        self.assert_match("перемотай на 20 секунд назад", "seek_back", seconds=20)
        self.assert_match("отмотай на 5 секунд", "seek_back", seconds=5)

    def test_seek_with_site_in_any_order(self) -> None:
        self.assert_match(
            "перемотай ютуб на 30 секунд",
            "seek",
            site="youtube",
            seconds=30,
        )
        self.assert_match(
            "перемотай на 30 секунд на ютубе",
            "seek",
            site="youtube",
            seconds=30,
        )
        self.assert_match(
            "перемотай видео на ютубе на 30 секунд",
            "seek",
            site="youtube",
            seconds=30,
        )
        self.assert_match(
            "перемотай назад на 10 секунд на саундклауде",
            "seek_back",
            site="soundcloud",
            seconds=10,
        )

    def test_play_with_site_in_any_order(self) -> None:
        self.assert_match("продолжи на ютубе", "play", site="youtube")
        self.assert_match("на ютубе продолжи", "play", site="youtube")
        self.assert_match("продолжи видео на ютубе", "play", site="youtube")

    def test_generic_search_does_not_steal_likes(self) -> None:
        self.assert_match(
            "включи на саундклауде скучно",
            "soundcloud_search",
            query="скучно",
        )

    def test_unrelated_phrases_still_do_not_match(self) -> None:
        self.assertIsNone(self.intents.match("включи музыку"))
        self.assertIsNone(self.intents.match("открой вкладку"))
        self.assertIsNone(self.intents.match("найди"))


if __name__ == "__main__":
    unittest.main()
