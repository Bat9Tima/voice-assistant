from __future__ import annotations

import unittest

from app.actions import ActionRouter
from app.context import CommandContext


class FakeBrowser:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, action: str, timeout: float = 90, **args) -> dict:
        self.calls.append((action, args))
        return {"ok": True, "data": {}}

    @property
    def last(self) -> tuple[str, dict]:
        return self.calls[-1]


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


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class DeclarativeRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.browser = FakeBrowser()
        self.router = ActionRouter({}, self.browser)

    def test_open_site_uses_intent_and_remembers_context(self) -> None:
        result = self.router.handle("открой яндекс музыку")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.last[0], "goto")
        self.assertEqual(self.browser.last[1]["url"], "https://music.yandex.ru")
        self.assertEqual(self.router.context.get("media").slots["site"], "яндекс музыка")

    def test_google_search_cleans_filler_words(self) -> None:
        result = self.router.handle("найди мне кота")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.last[0], "google")
        self.assertEqual(self.browser.last[1]["query"], "кота")
        self.assertEqual(self.router.context.get("search").action, "google_search")

    def test_youtube_site_phrase_beats_generic_search(self) -> None:
        result = self.router.handle("включи видео котики на ютубе")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.last[0], "youtube")
        self.assertEqual(self.browser.last[1]["query"], "котики")
        self.assertEqual(self.browser.last[1]["index"], 1)

    def test_intent_is_logged(self) -> None:
        with self.assertLogs("app.actions", level="INFO") as logs:
            self.router.handle("открой ютуб")
        self.assertTrue(any("Интент: open_site" in line for line in logs.output))


class SearchContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.browser = FakeBrowser()
        self.router = ActionRouter({}, self.browser)

    def test_next_youtube_result(self) -> None:
        self.router.handle("найди на ютубе котиков")
        result = self.router.handle("еще один")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.last[0], "youtube")
        self.assertEqual(self.browser.last[1]["query"], "котиков")
        self.assertEqual(self.browser.last[1]["index"], 2)

    def test_next_soundcloud_result(self) -> None:
        self.router.handle("включи на саундклауде lofi")
        result = self.router.handle("еще один")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.last[0], "soundcloud")
        self.assertEqual(self.browser.last[1]["query"], "lofi")
        self.assertEqual(self.browser.last[1]["index"], 2)

    def test_bare_ordinal_opens_result_from_context(self) -> None:
        self.router.handle("загугли погода москва")
        result = self.router.handle("вторую")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.last[0], "search_result")
        self.assertEqual(self.browser.last[1]["index"], 2)

    def test_next_google_result_uses_last_opened_index(self) -> None:
        self.router.handle("загугли погода москва")
        self.router.handle("вторую")
        self.router.handle("еще один")
        self.assertEqual(self.browser.last[0], "search_result")
        self.assertEqual(self.browser.last[1]["index"], 3)

    def test_repeat_google_search(self) -> None:
        self.router.handle("загугли погода москва")
        result = self.router.handle("повтори")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.last[0], "google")
        self.assertEqual(self.browser.last[1]["query"], "погода москва")

    def test_context_expires(self) -> None:
        clock = Clock()
        router = ActionRouter(
            {},
            self.browser,
            context=CommandContext(ttl_sec=5, clock=clock),
        )
        router.handle("найди на ютубе котиков")
        clock.advance(6)
        self.assertIsNone(router.handle("еще один"))


class VolumeContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.browser = FakeBrowser()
        self.router = ActionRouter({}, self.browser)
        self.volume = FakeVolume(0.5)
        self.router._volume = self.volume

    def test_explicit_quiet_after_soundcloud_volume_targets_player(self) -> None:
        self.router.handle("громкость саундклауда 5")
        result = self.router.handle("сделай тише")
        self.assertIsNotNone(result)
        self.assertEqual(self.browser.last[0], "soundcloud_volume")
        self.assertEqual(self.browser.last[1]["mode"], "down")
        self.assertEqual(self.browser.last[1]["value"], 10)
        self.assertAlmostEqual(self.volume.level, 0.5)

    def test_bare_quiet_stays_system_after_soundcloud_volume(self) -> None:
        self.router.handle("громкость саундклауда 5")
        before = len(self.browser.calls)
        result = self.router.handle("тише")
        self.assertIsNotNone(result)
        self.assertEqual(result.message, "Громкость: 40%")
        self.assertAlmostEqual(self.volume.level, 0.4)
        self.assertFalse(
            any(
                action == "soundcloud_volume"
                for action, _ in self.browser.calls[before:]
            )
        )

    def test_explicit_quiet_without_context_targets_system(self) -> None:
        result = self.router.handle("сделай тише")
        self.assertIsNotNone(result)
        self.assertEqual(result.message, "Громкость: 40%")
        self.assertAlmostEqual(self.volume.level, 0.4)

    def test_quiet_after_system_volume_targets_system(self) -> None:
        self.router.handle("громкость пять")
        result = self.router.handle("тише")
        self.assertIsNotNone(result)
        self.assertEqual(result.message, "Громкость: 40%")
        self.assertAlmostEqual(self.volume.level, 0.4)
        self.assertFalse(
            any(action == "soundcloud_volume" for action, _ in self.browser.calls)
        )


if __name__ == "__main__":
    unittest.main()
