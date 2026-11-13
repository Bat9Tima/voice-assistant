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
        self.assertIsNone(BrowserController._find_soundcloud_like(items, "зима холода"))

    def test_filter_queries_include_prefix_fallback(self) -> None:
        queries = BrowserController._soundcloud_filter_queries("скучна")
        self.assertIn("скучна", queries)
        self.assertIn("скуч", queries)


class FakePage:
    def __init__(self, url: str) -> None:
        self.url = url
        self.scripts: list[str] = []

    def evaluate(self, script: str, arg=None):
        self.scripts.append(script)
        return {"position": 42.0, "duration": 100.0}

    def bring_to_front(self) -> None:
        pass

    def is_closed(self) -> bool:
        return False


class FakeCdp:
    def __init__(self) -> None:
        self.enabled: list[bool] = []
        self.detached = False

    def send(self, method: str, params: dict) -> None:
        if method == "Emulation.setFocusEmulationEnabled":
            self.enabled.append(bool(params["enabled"]))

    def detach(self) -> None:
        self.detached = True


class FakeContext:
    def __init__(self) -> None:
        self.sessions: list[FakeCdp] = []

    def new_cdp_session(self, page) -> FakeCdp:
        session = FakeCdp()
        self.sessions.append(session)
        return session


class FocusPage(FakePage):
    def __init__(self, url: str, focused: bool) -> None:
        super().__init__(url)
        self.focused = focused

    def evaluate(self, script: str, arg=None):
        if "hasFocus" in script:
            return self.focused
        return super().evaluate(script, arg)


class CurrentPageTest(unittest.TestCase):
    def make_controller(self, pages: list[FakePage]) -> BrowserController:
        controller = BrowserController.__new__(BrowserController)
        controller._pages = lambda: pages
        controller._context = FakeContext()
        controller._assistant_page = None
        return controller

    def test_returns_focused_tab_and_restores_emulation(self) -> None:
        soundcloud = FocusPage("https://soundcloud.com/x", False)
        youtube = FocusPage("https://www.youtube.com/watch?v=1", True)
        controller = self.make_controller([soundcloud, youtube])
        self.assertIs(controller._current_page(), youtube)
        self.assertIs(controller._assistant_page, youtube)
        for session in controller._context.sessions:
            self.assertEqual(session.enabled, [False, True])
            self.assertTrue(session.detached)

    def test_falls_back_to_assistant_tab(self) -> None:
        soundcloud = FocusPage("https://soundcloud.com/x", False)
        youtube = FocusPage("https://www.youtube.com/watch?v=1", False)
        controller = self.make_controller([soundcloud, youtube])
        controller._assistant_page = soundcloud
        self.assertIs(controller._current_page(), soundcloud)

    def test_returns_none_without_focus_or_assistant_tab(self) -> None:
        page = FocusPage("https://example.com", False)
        controller = self.make_controller([page])
        self.assertIsNone(controller._current_page())


class MediaTargetTest(unittest.TestCase):
    """Без site команда должна попадать в текущую вкладку."""

    def setUp(self) -> None:
        self.current = FakePage("https://www.youtube.com/watch?v=1")
        self.playing = FakePage("https://soundcloud.com/track")
        self.youtube = FakePage("https://www.youtube.com/watch?v=2")
        self.soundcloud = FakePage("https://soundcloud.com/other")
        self.media = FakePage("https://example.com/media")
        self.controller = BrowserController.__new__(BrowserController)
        self.controller._current_page = lambda: self.current
        self.controller._page_has_media = lambda page: page is self.current
        self.controller._playing_page = lambda: self.playing
        self.controller._youtube_page = lambda: self.youtube
        self.controller._soundcloud_page = lambda: self.soundcloud
        self.controller._media_page = lambda: self.media

    def test_current_tab_wins(self) -> None:
        self.assertIs(self.controller._media_target(), self.current)

    def test_falls_back_to_playing_tab(self) -> None:
        self.controller._page_has_media = lambda page: False
        self.assertIs(self.controller._media_target(), self.playing)

    def test_explicit_site_wins_over_current(self) -> None:
        self.assertIs(self.controller._media_target("youtube"), self.youtube)
        self.assertIs(self.controller._media_target("soundcloud"), self.soundcloud)

    def test_falls_back_to_soundcloud_then_media(self) -> None:
        self.controller._page_has_media = lambda page: False
        self.controller._playing_page = lambda: None
        self.assertIs(self.controller._media_target(), self.soundcloud)
        self.controller._soundcloud_page = lambda: None
        self.assertIs(self.controller._media_target(), self.media)

    def test_stop_keeps_position(self) -> None:
        self.controller._media_target = lambda site=None: self.current
        result = self.controller._act_stop()
        self.assertEqual(result["state"], "stopped")
        self.assertEqual(result["position"], 42.0)
        self.assertNotIn("currentTime =", self.current.scripts[0])
