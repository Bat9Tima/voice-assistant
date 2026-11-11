from __future__ import annotations

import unittest

from app.actions import VOLUME_STEP, ActionRouter, parse_volume_percent


class FakeBrowser:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, action: str, timeout: float = 90, **args) -> dict:
        self.calls.append((action, args))
        return {"ok": True, "data": {}}


class FakeVolume:
    def __init__(self, level: float = 0.5, muted: bool = False) -> None:
        self.level = level
        self.muted = muted

    def GetMasterVolumeLevelScalar(self) -> float:
        return self.level

    def SetMasterVolumeLevelScalar(self, level: float, context=None) -> None:
        self.level = level

    def GetMute(self) -> int:
        return int(self.muted)

    def SetMute(self, mute: int, context=None) -> None:
        self.muted = bool(mute)


class VolumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.browser = FakeBrowser()
        self.router = ActionRouter({}, self.browser)
        self.volume = FakeVolume(0.5)
        self.router._volume = self.volume

    def test_parse_volume_percent(self) -> None:
        self.assertEqual(parse_volume_percent("5"), 50)
        self.assertEqual(parse_volume_percent("10"), 100)
        self.assertEqual(parse_volume_percent("2"), 20)
        self.assertEqual(parse_volume_percent("0"), 0)
        self.assertEqual(parse_volume_percent("пять"), 50)
        self.assertEqual(parse_volume_percent("десять"), 100)
        self.assertEqual(parse_volume_percent("50"), 50)
        self.assertEqual(parse_volume_percent("пятьдесят"), 50)
        self.assertEqual(parse_volume_percent("двадцать пять"), 25)
        self.assertEqual(parse_volume_percent("сто"), 100)
        self.assertEqual(parse_volume_percent("45 процентов"), 45)
        self.assertEqual(parse_volume_percent("5 процентов"), 5)
        self.assertEqual(parse_volume_percent("на максимум"), 100)
        self.assertIsNone(parse_volume_percent("громкость"))

    def test_absolute_volume_words(self) -> None:
        result = self.router.handle("громкость пятьдесят")
        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        self.assertTrue(result.ok)
        self.assertEqual(result.message, "Громкость: 50%")
        self.assertAlmostEqual(self.volume.level, 0.5)

        result = self.router.handle("громкость пять")
        self.assertIsNotNone(result)
        self.assertEqual(result.message, "Громкость: 50%")
        self.assertAlmostEqual(self.volume.level, 0.5)

        result = self.router.handle("громкость десять")
        self.assertIsNotNone(result)
        self.assertEqual(result.message, "Громкость: 100%")
        self.assertAlmostEqual(self.volume.level, 1.0)

    def test_absolute_volume_digits(self) -> None:
        result = self.router.handle("поставь громкость на 35")
        self.assertIsNotNone(result)
        self.assertEqual(result.message, "Громкость: 35%")
        self.assertAlmostEqual(self.volume.level, 0.35)

        result = self.router.handle("поставь громкость на 5")
        self.assertIsNotNone(result)
        self.assertEqual(result.message, "Громкость: 50%")
        self.assertAlmostEqual(self.volume.level, 0.5)

    def test_volume_up_uses_larger_step(self) -> None:
        result = self.router.handle("громче")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(self.volume.level, 0.5 + VOLUME_STEP)
        self.assertEqual(result.message, f"Громкость: {round((0.5 + VOLUME_STEP) * 100)}%")

    def test_volume_up_custom_amount(self) -> None:
        result = self.router.handle("громче на 20")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(self.volume.level, 0.7)
        self.assertEqual(result.message, "Громкость: 70%")

    def test_volume_up_custom_units(self) -> None:
        result = self.router.handle("громче на 2")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(self.volume.level, 0.7)
        self.assertEqual(result.message, "Громкость: 70%")

    def test_volume_is_clamped_at_max(self) -> None:
        self.volume.level = 0.95
        result = self.router.handle("громче")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(self.volume.level, 1.0)
        self.assertEqual(result.message, "Громкость: 100%")

        result = self.router.handle("громче")
        self.assertIsNotNone(result)
        self.assertEqual(result.message, "Громкость уже 100%")

    def test_volume_is_clamped_at_min(self) -> None:
        self.volume.level = 0.05
        result = self.router.handle("тише")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(self.volume.level, 0.0)
        self.assertEqual(result.message, "Громкость: 0%")

    def test_mute(self) -> None:
        result = self.router.handle("выключи звук")
        self.assertIsNotNone(result)
        self.assertTrue(self.volume.muted)
        self.assertEqual(result.message, "Звук выключен")


class SoundCloudTest(unittest.TestCase):
    def setUp(self) -> None:
        self.browser = FakeBrowser()
        self.router = ActionRouter({}, self.browser)

    def test_query_extraction(self) -> None:
        self.assertEqual(
            ActionRouter._soundcloud_query("включи на саундклауде never gonna give you up"),
            "never gonna give you up",
        )
        self.assertEqual(
            ActionRouter._soundcloud_query("включи трек в саунд клоуде daft punk"),
            "daft punk",
        )
        self.assertEqual(
            ActionRouter._soundcloud_query("поставь на саунд клауде lofi"),
            "lofi",
        )
        self.assertEqual(
            ActionRouter._soundcloud_query("включи платон влюбилась в саунклауде"),
            "платон влюбилась",
        )
        self.assertEqual(
            ActionRouter._soundcloud_query("включи трек в санклауде daft punk"),
            "daft punk",
        )
        self.assertEqual(
            ActionRouter._soundcloud_query("включи Bohemian Rhapsody на саунтлауди"),
            "Bohemian Rhapsody",
        )

    def test_misspelled_site_routes_to_soundcloud(self) -> None:
        result = self.router.handle("включи платон влюбилась в саунклауде")
        self.assertIsNotNone(result)
        action, args = self.browser.calls[0]
        self.assertEqual(action, "soundcloud")
        self.assertEqual(args["query"], "платон влюбилась")

        result = self.router.handle("громкость на саунклауде пятьдесят")
        self.assertIsNotNone(result)
        action, args = self.browser.calls[1]
        self.assertEqual(action, "soundcloud_volume")
        self.assertEqual(args["mode"], "set")
        self.assertEqual(args["value"], 50)

    def test_recognized_soundcloud_variants(self) -> None:
        for phrase in (
            "громкость сауклауда десять",
            "громкость соунклауда десять",
            "громкость саунглауда десять",
            "громкость сон клауда десять",
        ):
            self.browser.calls.clear()
            result = self.router.handle(phrase)
            self.assertIsNotNone(result, phrase)
            action, args = self.browser.calls[0]
            self.assertEqual(action, "soundcloud_volume", phrase)
            self.assertEqual(args["mode"], "set", phrase)
            self.assertEqual(args["value"], 100, phrase)

    def test_sauntlaudi_routes_to_soundcloud(self) -> None:
        result = self.router.handle("включи Bohemian Rhapsody на саунтлауди")
        self.assertIsNotNone(result)
        action, args = self.browser.calls[0]
        self.assertEqual(action, "soundcloud")
        self.assertEqual(args["query"], "bohemian rhapsody")

    def test_play_command_routes_to_soundcloud(self) -> None:
        result = self.router.handle("включи на саундклауде lofi hip hop")
        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        self.assertEqual(self.browser.calls[0][0], "soundcloud")
        self.assertEqual(self.browser.calls[0][1]["query"], "lofi hip hop")

    def test_open_command_opens_site(self) -> None:
        result = self.router.handle("открой саундклауд")
        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        self.assertEqual(self.browser.calls[0][0], "goto")
        self.assertEqual(self.browser.calls[0][1]["url"], "https://soundcloud.com")

    def test_media_commands_target_soundcloud(self) -> None:
        result = self.router.handle("следующий трек на саундклауде")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.calls[0][0], "next")

        result = self.router.handle("предыдущий трек в саундклауде")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.calls[1][0], "prev")

        result = self.router.handle("пауза на саундклауде")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.calls[2][0], "pause")

    def test_soundcloud_volume_set(self) -> None:
        result = self.router.handle("громкость саундклауда 5")
        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        action, args = self.browser.calls[0]
        self.assertEqual(action, "soundcloud_volume")
        self.assertEqual(args["mode"], "set")
        self.assertEqual(args["value"], 50)

        result = self.router.handle("громкость саундклауда 10")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.calls[1][1]["value"], 100)

        result = self.router.handle("громкость саундклауда 40")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.calls[2][1]["value"], 40)

    def test_soundcloud_volume_up_down(self) -> None:
        result = self.router.handle("громче в саундклауде")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.calls[0][0], "soundcloud_volume")
        self.assertEqual(self.browser.calls[0][1]["mode"], "up")

        result = self.router.handle("тише на 1 в саундклауде")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.calls[1][1]["mode"], "down")
        self.assertEqual(self.browser.calls[1][1]["value"], 10)

        result = self.router.handle("тише на 2 в саундклауде")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.calls[2][1]["value"], 20)

    def test_soundcloud_volume_mute(self) -> None:
        result = self.router.handle("выключи звук в саундклауде")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.calls[0][0], "soundcloud_volume")
        self.assertEqual(self.browser.calls[0][1]["mode"], "mute")

        result = self.router.handle("включи звук на саундклауде")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.calls[1][1]["mode"], "unmute")

    def test_likes_open(self) -> None:
        result = self.router.handle("открой лайкнутые треки на саундклауде")
        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        action, args = self.browser.calls[0]
        self.assertEqual(action, "soundcloud_likes")
        self.assertFalse(args["play"])

    def test_likes_specific_track(self) -> None:
        result = self.router.handle("включи лайкнутый трек eva eva на саундклауде")
        self.assertIsNotNone(result)
        action, args = self.browser.calls[0]
        self.assertEqual(action, "soundcloud_likes")
        self.assertTrue(args["play"])
        self.assertFalse(args["random"])
        self.assertEqual(args["query"], "eva eva")

    def test_likes_random_track(self) -> None:
        result = self.router.handle("включи случайный лайкнутый трек на саундклауде")
        self.assertIsNotNone(result)
        action, args = self.browser.calls[0]
        self.assertEqual(action, "soundcloud_likes")
        self.assertTrue(args["play"])
        self.assertTrue(args["random"])
        self.assertEqual(args["query"], "")

    def test_likes_random_track_stt_variant(self) -> None:
        # Реальная фраза из лога: «случайно» вместо «случайный» и без «на».
        result = self.router.handle("включи случайно лайкнутый трек санклауд")
        self.assertIsNotNone(result)
        action, args = self.browser.calls[0]
        self.assertEqual(action, "soundcloud_likes")
        self.assertTrue(args["play"])
        self.assertTrue(args["random"])
        self.assertEqual(args["query"], "")

    def test_likes_random_track_open_variant(self) -> None:
        result = self.router.handle("открой случайный лайкнутый трек на саундклауде")
        self.assertIsNotNone(result)
        action, args = self.browser.calls[0]
        self.assertEqual(action, "soundcloud_likes")
        self.assertTrue(args["play"])
        self.assertTrue(args["random"])

    def test_likes_open_without_site(self) -> None:
        for phrase in (
            "открой лайкнутые треки",
            "покажи лайкнутые треки",
            "открой избранное",
            "открой любимые треки",
        ):
            self.browser.calls.clear()
            result = self.router.handle(phrase)
            self.assertIsNotNone(result, phrase)
            action, args = self.browser.calls[0]
            self.assertEqual(action, "soundcloud_likes", phrase)
            self.assertFalse(args["play"], phrase)

    def test_likes_play_without_site(self) -> None:
        for phrase in (
            "включи лайкнутые треки",
            "включи избранное",
            "включи любимые треки",
        ):
            self.browser.calls.clear()
            result = self.router.handle(phrase)
            self.assertIsNotNone(result, phrase)
            action, args = self.browser.calls[0]
            self.assertEqual(action, "soundcloud_likes", phrase)
            self.assertTrue(args["play"], phrase)
            self.assertTrue(args["random"], phrase)

    def test_likes_play_named_track_without_site(self) -> None:
        result = self.router.handle("включи лайкнутый трек eva eva")
        self.assertIsNotNone(result)
        action, args = self.browser.calls[0]
        self.assertEqual(action, "soundcloud_likes")
        self.assertEqual(args["query"], "eva eva")
        self.assertFalse(args["random"])

        self.browser.calls.clear()
        result = self.router.handle("поставь лайкнутую песню eva eva")
        self.assertIsNotNone(result)
        action, args = self.browser.calls[0]
        self.assertEqual(action, "soundcloud_likes")
        self.assertEqual(args["query"], "eva eva")
        self.assertFalse(args["random"])

    def test_likes_site_variants_do_not_leak_prepositions(self) -> None:
        for phrase in (
            "включи лайкнутую песню на саундклауде",
            "включи любимый трек на саундклауде",
            "включи любимые треки в саундклауде",
            "поставь любимые треки на саундклауде",
            "включи избранное в саундклауде",
            "включи лайки на саундклауде",
        ):
            self.browser.calls.clear()
            result = self.router.handle(phrase)
            self.assertIsNotNone(result, phrase)
            action, args = self.browser.calls[0]
            self.assertEqual(action, "soundcloud_likes", phrase)
            self.assertEqual(args["query"], "", phrase)
            self.assertTrue(args["random"], phrase)

    def test_likes_prepositional_variants(self) -> None:
        # Реальные фразы из лога: «в лайкнутых треках» не совпадала с интентом
        # и уходила в обычный поиск SoundCloud.
        for phrase, query in (
            ("включи в лайкнутых треках отпусти меня", "отпусти меня"),
            ("включи лайкнутых треках отпусти меня", "отпусти меня"),
            (
                "включи лайкнутых треках на саундклауде и отпусти меня",
                "отпусти меня",
            ),
            ("включи в лайках eva eva", "eva eva"),
            ("включи из лайкнутых треков eva eva", "eva eva"),
        ):
            self.browser.calls.clear()
            result = self.router.handle(phrase)
            self.assertIsNotNone(result, phrase)
            action, args = self.browser.calls[0]
            self.assertEqual(action, "soundcloud_likes", phrase)
            self.assertTrue(args["play"], phrase)
            self.assertFalse(args["random"], phrase)
            self.assertEqual(args["query"], query, phrase)

    def test_likes_word_order_variants(self) -> None:
        # Перестановка слов не должна отправлять команду в общий поиск.
        for phrase in (
            "включи скучно в лайкнутых на саундклауде",
            "включи в лайкнутых на саундклауде скучно",
            "включи скучно в лайкнутых",
            "включи в лайкнутых скучно",
        ):
            self.browser.calls.clear()
            result = self.router.handle(phrase)
            self.assertIsNotNone(result, phrase)
            action, args = self.browser.calls[0]
            self.assertEqual(action, "soundcloud_likes", phrase)
            self.assertTrue(args["play"], phrase)
            self.assertFalse(args["random"], phrase)
            self.assertEqual(args["query"], "скучно", phrase)

    def test_likes_prepositional_without_query_plays_random(self) -> None:
        for phrase in (
            "включи в лайкнутых треках",
            "включи лайкнутых треках",
            "включи в лайках",
        ):
            self.browser.calls.clear()
            result = self.router.handle(phrase)
            self.assertIsNotNone(result, phrase)
            action, args = self.browser.calls[0]
            self.assertEqual(action, "soundcloud_likes", phrase)
            self.assertTrue(args["play"], phrase)
            self.assertTrue(args["random"], phrase)
            self.assertEqual(args["query"], "", phrase)

    def test_likes_prepositional_open(self) -> None:
        for phrase in ("открой в лайкнутых треках", "покажи в лайках"):
            self.browser.calls.clear()
            result = self.router.handle(phrase)
            self.assertIsNotNone(result, phrase)
            action, args = self.browser.calls[0]
            self.assertEqual(action, "soundcloud_likes", phrase)
            self.assertFalse(args["play"], phrase)

    def test_soundcloud_filler_without_query_opens_site(self) -> None:
        for phrase in (
            "включи трек на саундклауде",
            "поставь трек в саундклауде",
            "включи музыку саундклауд",
        ):
            self.browser.calls.clear()
            result = self.router.handle(phrase)
            self.assertIsNotNone(result, phrase)
            action, args = self.browser.calls[0]
            self.assertEqual(action, "goto", phrase)
            self.assertEqual(args["url"], "https://soundcloud.com", phrase)

    def test_open_soundcloud_and_play(self) -> None:
        result = self.router.handle("открой саундклауд и включи скучно")
        self.assertIsNotNone(result)
        action, args = self.browser.calls[0]
        self.assertEqual(action, "soundcloud")
        self.assertEqual(args["query"], "скучно")

    def test_pause_imperative(self) -> None:
        result = self.router.handle("остановись")
        self.assertIsNotNone(result)
        action, _ = self.browser.calls[0]
        self.assertEqual(action, "stop")


class MediaControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.browser = FakeBrowser()
        self.router = ActionRouter({}, self.browser)

    def handle(self, phrase: str) -> tuple[str, dict]:
        self.browser.calls.clear()
        result = self.router.handle(phrase)
        self.assertIsNotNone(result, phrase)
        self.assertTrue(result.handled, phrase)
        return self.browser.calls[0]

    def test_stop_does_not_require_site(self) -> None:
        for phrase in ("стоп", "стоп видео", "останови музыку"):
            action, args = self.handle(phrase)
            self.assertEqual(action, "stop", phrase)
            self.assertNotIn("site", args, phrase)

    def test_stop_targets_site(self) -> None:
        cases = {
            "стоп на ютубе": "youtube",
            "останови ютуб": "youtube",
            "стоп видео в саундклауде": "soundcloud",
            "останови трек на саундклауде": "soundcloud",
        }
        for phrase, site in cases.items():
            action, args = self.handle(phrase)
            self.assertEqual(action, "stop", phrase)
            self.assertEqual(args["site"], site, phrase)

    def test_seek_forward_seconds(self) -> None:
        for phrase, seconds in (
            ("перемотай на 30 секунд", 30),
            ("перемотай 30 секунд", 30),
            ("перемотай вперед на 15 секунд", 15),
            ("перемотай на 20 секунд вперед", 20),
        ):
            action, args = self.handle(phrase)
            self.assertEqual(action, "seek", phrase)
            self.assertEqual(args["seconds"], seconds, phrase)

    def test_seek_backward_seconds(self) -> None:
        for phrase, seconds in (
            ("перемотай назад на 20 секунд", -20),
            ("перемотай на 10 секунд назад", -10),
            ("отмотай на 5 секунд", -5),
            ("отмотай назад на 5 секунд", -5),
        ):
            action, args = self.handle(phrase)
            self.assertEqual(action, "seek", phrase)
            self.assertEqual(args["seconds"], seconds, phrase)

    def test_seek_default_is_ten_seconds(self) -> None:
        action, args = self.handle("перемотай")
        self.assertEqual(action, "seek")
        self.assertEqual(args["seconds"], 10)

        action, args = self.handle("перемотай назад")
        self.assertEqual(action, "seek")
        self.assertEqual(args["seconds"], -10)

    def test_seek_targets_site(self) -> None:
        cases = {
            "перемотай ютуб на 30 секунд": ("youtube", 30),
            "перемотай на ютубе на 30 секунд": ("youtube", 30),
            "перемотай видео на ютубе на 30 секунд": ("youtube", 30),
            "перемотай саундклауд назад на 10 секунд": ("soundcloud", -10),
            "перемотай назад на 10 секунд на саундклауде": ("soundcloud", -10),
        }
        for phrase, (site, seconds) in cases.items():
            action, args = self.handle(phrase)
            self.assertEqual(action, "seek", phrase)
            self.assertEqual(args["site"], site, phrase)
            self.assertEqual(args["seconds"], seconds, phrase)

    def test_seek_rejects_too_long(self) -> None:
        result = self.router.handle("перемотай на 4000 секунд")
        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        self.assertFalse(result.ok)
        self.assertEqual(self.browser.calls, [])

    def test_play_targets_site(self) -> None:
        cases = {
            "продолжи": None,
            "продолжи на ютубе": "youtube",
            "продолжи видео на ютубе": "youtube",
        }
        for phrase, site in cases.items():
            action, args = self.handle(phrase)
            self.assertEqual(action, "play", phrase)
            if site is None:
                self.assertNotIn("site", args, phrase)
            else:
                self.assertEqual(args["site"], site, phrase)

    def test_pause_targets_youtube(self) -> None:
        action, args = self.handle("пауза на ютубе")
        self.assertEqual(action, "pause")
        self.assertEqual(args["site"], "youtube")


class SoundCloudNoiseFallbackTest(unittest.TestCase):
    """GigaAM на фоне музыки теряет «саундклауда» — спасает контекст вкладки."""

    class TabsBrowser(FakeBrowser):
        def __init__(self, urls: list[str]) -> None:
            super().__init__()
            self.urls = urls

        def call(self, action: str, timeout: float = 90, **args) -> dict:
            self.calls.append((action, args))
            if action == "tabs":
                tabs = [
                    {"index": i + 1, "url": url, "title": ""}
                    for i, url in enumerate(self.urls)
                ]
                return {"ok": True, "data": {"tabs": tabs}}
            return {"ok": True, "data": {}}

    def make_router(self, urls: list[str]) -> tuple[ActionRouter, SoundCloudNoiseFallbackTest.TabsBrowser]:
        browser = self.TabsBrowser(urls)
        return ActionRouter({}, browser), browser

    def test_garbled_word_routes_to_soundcloud(self) -> None:
        router, browser = self.make_router(["https://soundcloud.com/discover"])
        for phrase in (
            "громкость на паузу пять",
            "громкость сомкала на пять",
            "пойдуна громко сама правда десять",
        ):
            browser.calls.clear()
            result = router.handle(phrase)
            self.assertIsNotNone(result, phrase)
            action, args = browser.calls[-1]
            self.assertEqual(action, "soundcloud_volume", phrase)
            self.assertEqual(args["mode"], "set", phrase)

    def test_sauna_klauha_is_soundcloud(self) -> None:
        router, browser = self.make_router(["https://soundcloud.com/discover"])
        result = router.handle("громкость сауна клауда десять")
        self.assertIsNotNone(result)
        action, args = browser.calls[-1]
        self.assertEqual(action, "soundcloud_volume")
        self.assertEqual(args["value"], 100)

    def test_leading_garble_still_volume(self) -> None:
        """Живой случай: «элонорд громкость саунклауда десять» открывал сайт."""
        router, browser = self.make_router(["https://soundcloud.com/discover"])
        result = router.handle("элонорд громкость саунклауда десять")
        self.assertIsNotNone(result)
        action, args = browser.calls[-1]
        self.assertEqual(action, "soundcloud_volume")
        self.assertEqual(args["mode"], "set")
        self.assertEqual(args["value"], 100)

    def test_relative_soundcloud_volume_not_system(self) -> None:
        router, browser = self.make_router(["https://soundcloud.com/discover"])
        result = router.handle("тише на один в саундклауде")
        self.assertIsNotNone(result)
        action, args = browser.calls[-1]
        self.assertEqual(action, "soundcloud_volume")
        self.assertEqual(args["mode"], "down")
        self.assertEqual(args["value"], 10)

    def test_without_soundcloud_tab_falls_through(self) -> None:
        router, _ = self.make_router(["https://www.youtube.com/"])
        self.assertIsNone(router.handle("громкость сомкала на пять"))

    def test_plain_volume_still_system(self) -> None:
        class Volume:
            level = 0.5
            muted = 0

            def GetMasterVolumeLevelScalar(self):
                return self.level

            def SetMasterVolumeLevelScalar(self, value, context=None):
                self.level = value

            def GetMute(self):
                return self.muted

            def SetMute(self, value, context=None):
                self.muted = value

        router, browser = self.make_router(["https://soundcloud.com/discover"])
        router._volume = Volume()
        result = router.handle("громкость пять")
        self.assertIsNotNone(result)
        self.assertEqual(router._volume.level, 0.5)
        self.assertFalse(
            any(action == "soundcloud_volume" for action, _ in browser.calls)
        )


class EnglishSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.browser = FakeBrowser()
        self.router = ActionRouter({}, self.browser)

    def test_google_search(self) -> None:
        for phrase, query in (
            ("Find Funny Cats", "funny cats"),
            ("search for best coffee near me", "best coffee near me"),
            ("find me funny cats", "funny cats"),
            ("look up python path", "python path"),
            ("google openai gpt 5", "openai gpt 5"),
            ("найди Funny Cats", "funny cats"),
        ):
            with self.subTest(phrase=phrase):
                browser = FakeBrowser()
                router = ActionRouter({}, browser)
                result = router.handle(phrase)
                self.assertIsNotNone(result)
                self.assertTrue(result.handled)
                action, args = browser.calls[0]
                self.assertEqual(action, "google")
                self.assertEqual(args["query"], query)

    def test_youtube_search(self) -> None:
        for phrase, query in (
            ("Find funny cats on YouTube", "funny cats"),
            ("search for lofi hip hop on youtube", "lofi hip hop"),
            ("найди на ютубе funny cats", "funny cats"),
        ):
            with self.subTest(phrase=phrase):
                browser = FakeBrowser()
                router = ActionRouter({}, browser)
                result = router.handle(phrase)
                self.assertIsNotNone(result)
                action, args = browser.calls[0]
                self.assertEqual(action, "youtube")
                self.assertEqual(args["query"], query)

    def test_soundcloud_search(self) -> None:
        for phrase, query in (
            ("Play Bohemian Rhapsody on SoundCloud", "bohemian rhapsody"),
            ("play music on soundcloud", "music"),
            ("включи Bohemian Rhapsody на саундклауде", "bohemian rhapsody"),
        ):
            with self.subTest(phrase=phrase):
                browser = FakeBrowser()
                router = ActionRouter({}, browser)
                result = router.handle(phrase)
                self.assertIsNotNone(result)
                action, args = browser.calls[0]
                self.assertEqual(action, "soundcloud")
                self.assertEqual(args["query"], query)

    def test_open_site(self) -> None:
        result = self.router.handle("open YouTube")
        self.assertIsNotNone(result)
        action, args = self.browser.calls[0]
        self.assertEqual(action, "goto")
        self.assertEqual(args["url"], "https://www.youtube.com")

        result = self.router.handle("open stack overflow")
        self.assertIsNotNone(result)
        action, args = self.browser.calls[1]
        self.assertEqual(action, "google")
        self.assertEqual(args["query"], "stack overflow")

    def test_truncated_youtube_opens_site(self) -> None:
        # GigaAM на быстрой речи слышит «открой ютуб» как «открой ту».
        result = self.router.handle("открой ту")
        self.assertIsNotNone(result)
        action, args = self.browser.calls[0]
        self.assertEqual(action, "goto")
        self.assertEqual(args["url"], "https://www.youtube.com")


class SearchResultTest(unittest.TestCase):
    def setUp(self) -> None:
        self.browser = FakeBrowser()
        self.router = ActionRouter({}, self.browser)

    def test_first_result_words(self) -> None:
        for phrase, index in (
            ("открой первую ссылку", 1),
            ("открой первый результат", 1),
            ("открой вторую ссылку", 2),
            ("перейди на третью ссылку", 3),
            ("нажми на четвертый результат", 4),
            ("открой пятую ссылку в гугле", 5),
        ):
            with self.subTest(phrase=phrase):
                browser = FakeBrowser()
                router = ActionRouter({}, browser)
                result = router.handle(phrase)
                self.assertIsNotNone(result)
                self.assertTrue(result.handled)
                action, args = browser.calls[0]
                self.assertEqual(action, "search_result")
                self.assertEqual(args["index"], index)

    def test_result_by_number(self) -> None:
        for phrase, index in (
            ("открой ссылку 2", 2),
            ("открой результат номер 3", 3),
            ("открой 1 ссылку", 1),
        ):
            with self.subTest(phrase=phrase):
                browser = FakeBrowser()
                router = ActionRouter({}, browser)
                result = router.handle(phrase)
                self.assertIsNotNone(result)
                action, args = browser.calls[0]
                self.assertEqual(action, "search_result")
                self.assertEqual(args["index"], index)

    def test_unknown_site_still_opens_google(self) -> None:
        result = self.router.handle("открой погоду")
        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        action, args = self.browser.calls[0]
        self.assertEqual(action, "google")
        self.assertEqual(args["query"], "погоду")

    def test_known_site_is_not_search_result(self) -> None:
        result = self.router.handle("открой саундклауд")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.calls[0][0], "goto")


if __name__ == "__main__":
    unittest.main()
