from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from app.actions import ActionRouter
from app.local_llm import (
    LocalLLM,
    LocalLLMError,
    LocalPlan,
    _build_plan,
    _extract_json,
)


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


class FakeResponse:
    def __init__(self, body: bytes = b"", status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args) -> bool:
        return False

    def read(self) -> bytes:
        return self.body


def make_llm(enabled: bool = True) -> LocalLLM:
    return LocalLLM(
        {
            "llm": {
                "enabled": enabled,
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "qwen2.5:3b-instruct",
                "timeout_sec": 2,
            }
        }
    )


class JsonExtractTest(unittest.TestCase):
    def test_plain_json(self) -> None:
        self.assertEqual(
            _extract_json('{"action": "pause"}'),
            {"action": "pause"},
        )

    def test_code_fence_and_text(self) -> None:
        raw = 'Вот ответ:\n```json\n{"action": "volume_up"}\n```\n'
        self.assertEqual(_extract_json(raw), {"action": "volume_up"})

    def test_no_json(self) -> None:
        self.assertIsNone(_extract_json("просто текст"))
        self.assertIsNone(_extract_json(""))


class PlanValidationTest(unittest.TestCase):
    def test_youtube(self) -> None:
        plan = _build_plan(
            {"action": "youtube", "query": "  lo-fi  ", "index": 2}
        )
        self.assertEqual(plan, LocalPlan("youtube", {"query": "lo-fi", "index": 2}))

    def test_index_is_clamped(self) -> None:
        plan = _build_plan({"action": "youtube", "query": "x", "index": 0})
        self.assertIsNotNone(plan)
        self.assertEqual(plan.args["index"], 1)
        plan = _build_plan({"action": "youtube", "query": "x", "index": 100})
        self.assertIsNotNone(plan)
        self.assertEqual(plan.args["index"], 20)

    def test_none_action(self) -> None:
        self.assertIsNone(_build_plan({"action": "none"}))

    def test_unknown_action(self) -> None:
        self.assertIsNone(_build_plan({"action": "launch_missiles"}))

    def test_missing_required_argument(self) -> None:
        self.assertIsNone(_build_plan({"action": "youtube"}))
        self.assertIsNone(_build_plan({"action": "open_site"}))

    def test_unknown_site(self) -> None:
        self.assertIsNone(
            _build_plan({"action": "open_site", "site": "неизвестный сайт"})
        )

    def test_open_site_known(self) -> None:
        plan = _build_plan({"action": "open_site", "site": "ютуб"})
        self.assertEqual(plan, LocalPlan("open_site", {"site": "ютуб"}))

    def test_goto_rejects_unsafe_schemes(self) -> None:
        self.assertIsNone(
            _build_plan({"action": "goto", "url": "javascript:alert(1)"})
        )
        self.assertIsNone(_build_plan({"action": "goto", "url": "file:///etc/passwd"}))
        self.assertIsNone(_build_plan({"action": "goto", "url": "https://exa mple.com"}))

    def test_goto_accepts_https(self) -> None:
        plan = _build_plan({"action": "goto", "url": "https://example.com/page"})
        self.assertEqual(
            plan, LocalPlan("goto", {"url": "https://example.com/page"})
        )

    def test_volume_range(self) -> None:
        self.assertEqual(
            _build_plan({"action": "volume", "level": 30}),
            LocalPlan("volume", {"level": 30}),
        )
        self.assertIsNone(_build_plan({"action": "volume", "level": 101}))
        self.assertIsNone(_build_plan({"action": "volume", "level": -1}))

    def test_scroll_direction(self) -> None:
        self.assertEqual(
            _build_plan({"action": "scroll", "direction": "up"}),
            LocalPlan("scroll", {"direction": "up"}),
        )
        self.assertIsNone(
            _build_plan({"action": "scroll", "direction": "sideways"})
        )

    def test_soundcloud_likes(self) -> None:
        plan = _build_plan(
            {"action": "soundcloud_likes", "query": "  отпусти меня  "}
        )
        self.assertEqual(
            plan,
            LocalPlan(
                "soundcloud_likes",
                {"query": "отпусти меня", "play": True},
            ),
        )

    def test_soundcloud_likes_defaults(self) -> None:
        self.assertEqual(
            _build_plan({"action": "soundcloud_likes"}),
            LocalPlan("soundcloud_likes", {"query": "", "play": True}),
        )
        self.assertEqual(
            _build_plan({"action": "soundcloud_likes", "play": False}),
            LocalPlan("soundcloud_likes", {"query": "", "play": False}),
        )
        self.assertEqual(
            _build_plan({"action": "soundcloud_likes", "play": "false"}),
            LocalPlan("soundcloud_likes", {"query": "", "play": False}),
        )

    def test_soundcloud_likes_rejects_long_query(self) -> None:
        self.assertIsNone(
            _build_plan({"action": "soundcloud_likes", "query": "x" * 201})
        )

    def test_stop(self) -> None:
        self.assertEqual(_build_plan({"action": "stop"}), LocalPlan("stop", {}))
        self.assertEqual(
            _build_plan({"action": "stop", "site": "YouTube"}),
            LocalPlan("stop", {"site": "youtube"}),
        )
        self.assertIsNone(_build_plan({"action": "stop", "site": "vk"}))

    def test_seek(self) -> None:
        self.assertEqual(
            _build_plan({"action": "seek", "seconds": 30}),
            LocalPlan("seek", {"seconds": 30, "direction": "forward"}),
        )
        self.assertEqual(
            _build_plan(
                {"action": "seek", "seconds": "10", "direction": "backward"}
            ),
            LocalPlan("seek", {"seconds": 10, "direction": "backward"}),
        )
        self.assertEqual(
            _build_plan({"action": "seek", "seconds": 5, "site": "SoundCloud"}),
            LocalPlan(
                "seek",
                {"seconds": 5, "direction": "forward", "site": "soundcloud"},
            ),
        )

    def test_seek_rejects_bad_values(self) -> None:
        self.assertIsNone(_build_plan({"action": "seek", "seconds": 0}))
        self.assertIsNone(_build_plan({"action": "seek", "seconds": 3601}))
        self.assertIsNone(
            _build_plan({"action": "seek", "direction": "sideways"})
        )

    def test_play_with_site(self) -> None:
        self.assertEqual(
            _build_plan({"action": "play", "site": "youtube"}),
            LocalPlan("play", {"site": "youtube"}),
        )
        self.assertIsNone(_build_plan({"action": "play", "site": "vk"}))


class LocalLLMPlanTest(unittest.TestCase):
    def test_disabled_returns_none(self) -> None:
        self.assertIsNone(make_llm(enabled=False).plan("включи музыку"))

    def test_plan_parses_model_answer(self) -> None:
        llm = make_llm()
        with patch.object(llm, "available", return_value=True), patch.object(
            llm, "_chat", return_value='{"action": "pause"}'
        ):
            self.assertEqual(llm.plan("пауза"), LocalPlan("pause"))

    def test_invalid_json_returns_none(self) -> None:
        llm = make_llm()
        with patch.object(llm, "available", return_value=True), patch.object(
            llm, "_chat", return_value="не знаю"
        ):
            self.assertIsNone(llm.plan("что-то"))

    def test_unavailable_returns_none(self) -> None:
        llm = make_llm()
        with patch.object(llm, "available", return_value=False):
            self.assertIsNone(llm.plan("включи музыку"))


class LocalLLMHttpTest(unittest.TestCase):
    def test_available_true_on_200(self) -> None:
        llm = make_llm()
        with patch(
            "app.local_llm.urllib.request.urlopen",
            return_value=FakeResponse(status=200),
        ):
            self.assertTrue(llm.available())

    def test_available_false_when_server_down(self) -> None:
        llm = make_llm()
        with patch(
            "app.local_llm.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            self.assertFalse(llm.available())

    def test_chat_parses_openai_response(self) -> None:
        body = json.dumps(
            {"choices": [{"message": {"content": '{"action": "play"}'}}]}
        ).encode("utf-8")
        llm = make_llm()
        with patch(
            "app.local_llm.urllib.request.urlopen",
            return_value=FakeResponse(body=body),
        ):
            self.assertEqual(llm._chat("включи"), '{"action": "play"}')

    def test_chat_sends_keep_alive(self) -> None:
        body = json.dumps(
            {"choices": [{"message": {"content": '{"action": "play"}'}}]}
        ).encode("utf-8")
        captured: dict = {}

        def fake_urlopen(request, timeout=None):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(body=body)

        llm = make_llm()
        with patch(
            "app.local_llm.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            llm._chat("включи")
        self.assertEqual(captured["payload"]["keep_alive"], "30m")

    def test_chat_omits_empty_keep_alive(self) -> None:
        body = json.dumps(
            {"choices": [{"message": {"content": '{"action": "play"}'}}]}
        ).encode("utf-8")
        captured: dict = {}

        def fake_urlopen(request, timeout=None):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(body=body)

        llm = make_llm()
        llm.keep_alive = ""
        with patch(
            "app.local_llm.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            llm._chat("включи")
        self.assertNotIn("keep_alive", captured["payload"])

    def test_http_error_is_reported(self) -> None:
        error = urllib.error.HTTPError(
            "http://127.0.0.1:11434/v1/chat/completions",
            500,
            "Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b"model is not loaded"),
        )
        llm = make_llm()
        with patch(
            "app.local_llm.urllib.request.urlopen",
            side_effect=error,
        ):
            with self.assertRaises(LocalLLMError) as ctx:
                llm._chat("включи")
        self.assertIn("model is not loaded", str(ctx.exception))


class ExecutePlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.browser = FakeBrowser()
        self.router = ActionRouter({}, self.browser)
        self.router._volume = FakeVolume(0.5)

    def test_youtube(self) -> None:
        result = self.router.execute_plan(
            "youtube", {"query": "lo-fi", "index": 1}
        )
        self.assertTrue(result.ok)
        self.assertEqual(self.browser.calls[-1][0], "youtube")
        self.assertEqual(self.browser.calls[-1][1]["query"], "lo-fi")

    def test_open_site(self) -> None:
        result = self.router.execute_plan("open_site", {"site": "ютуб"})
        self.assertTrue(result.ok)
        self.assertEqual(self.browser.calls[-1][0], "goto")
        self.assertIn("youtube.com", self.browser.calls[-1][1]["url"])

    def test_goto_rejects_bad_url(self) -> None:
        result = self.router.execute_plan("goto", {"url": "javascript:alert(1)"})
        self.assertFalse(result.ok)
        self.assertEqual(self.browser.calls, [])

    def test_volume(self) -> None:
        result = self.router.execute_plan("volume", {"level": 30})
        self.assertTrue(result.ok)
        self.assertEqual(result.message, "Громкость: 30%")
        self.assertAlmostEqual(self.router._volume.level, 0.3)

    def test_mute(self) -> None:
        result = self.router.execute_plan("mute", {})
        self.assertTrue(result.ok)
        self.assertTrue(self.router._volume.muted)

    def test_tab(self) -> None:
        result = self.router.execute_plan("tab", {"index": 2})
        self.assertTrue(result.ok)
        self.assertEqual(self.browser.calls[-1][0], "tab")
        self.assertEqual(self.browser.calls[-1][1]["index"], 2)

    def test_unknown_action(self) -> None:
        result = self.router.execute_plan("explode", {})
        self.assertFalse(result.ok)
        self.assertEqual(self.browser.calls, [])

    def test_soundcloud_likes(self) -> None:
        result = self.router.execute_plan(
            "soundcloud_likes", {"query": "отпусти меня", "play": True}
        )
        self.assertTrue(result.ok)
        action, args = self.browser.calls[-1]
        self.assertEqual(action, "soundcloud_likes")
        self.assertEqual(args["query"], "отпусти меня")
        self.assertFalse(args["random"])
        self.assertTrue(args["play"])

    def test_soundcloud_likes_cleans_query(self) -> None:
        self.router.execute_plan(
            "soundcloud_likes",
            {"query": "в лайкнутых треках отпусти меня", "play": True},
        )
        action, args = self.browser.calls[-1]
        self.assertEqual(action, "soundcloud_likes")
        self.assertEqual(args["query"], "отпусти меня")
        self.assertFalse(args["random"])

    def test_soundcloud_likes_random_without_query(self) -> None:
        result = self.router.execute_plan("soundcloud_likes", {"play": True})
        self.assertTrue(result.ok)
        action, args = self.browser.calls[-1]
        self.assertEqual(action, "soundcloud_likes")
        self.assertEqual(args["query"], "")
        self.assertTrue(args["random"])
        self.assertTrue(args["play"])

    def test_soundcloud_likes_open_only(self) -> None:
        result = self.router.execute_plan("soundcloud_likes", {"play": False})
        self.assertTrue(result.ok)
        action, args = self.browser.calls[-1]
        self.assertEqual(action, "soundcloud_likes")
        self.assertFalse(args["play"])

    def test_stop(self) -> None:
        result = self.router.execute_plan("stop", {})
        self.assertTrue(result.ok)
        self.assertEqual(self.browser.calls[-1], ("stop", {}))

    def test_stop_with_site(self) -> None:
        result = self.router.execute_plan("stop", {"site": "youtube"})
        self.assertTrue(result.ok)
        self.assertEqual(self.browser.calls[-1], ("stop", {"site": "youtube"}))
        self.assertFalse(
            self.router.execute_plan("stop", {"site": "vk"}).ok
        )

    def test_seek_forward(self) -> None:
        result = self.router.execute_plan(
            "seek", {"seconds": 30, "direction": "forward"}
        )
        self.assertTrue(result.ok)
        self.assertEqual(self.browser.calls[-1], ("seek", {"seconds": 30}))
        self.assertEqual(result.message, "Перематываю вперед на 30 с")

    def test_seek_backward(self) -> None:
        result = self.router.execute_plan(
            "seek", {"seconds": 10, "direction": "backward"}
        )
        self.assertTrue(result.ok)
        self.assertEqual(self.browser.calls[-1], ("seek", {"seconds": -10}))
        self.assertEqual(result.message, "Перематываю назад на 10 с")

    def test_seek_with_site(self) -> None:
        result = self.router.execute_plan(
            "seek", {"seconds": 15, "site": "soundcloud"}
        )
        self.assertTrue(result.ok)
        self.assertEqual(
            self.browser.calls[-1],
            ("seek", {"seconds": 15, "site": "soundcloud"}),
        )

    def test_seek_rejects_bad_values(self) -> None:
        for args in (
            {"seconds": 0},
            {"seconds": 3601},
            {"seconds": 10, "direction": "sideways"},
            {"seconds": 10, "site": "vk"},
        ):
            with self.subTest(args=args):
                result = self.router.execute_plan("seek", args)
                self.assertFalse(result.ok)
        self.assertEqual(self.browser.calls, [])

    def test_play_with_site(self) -> None:
        result = self.router.execute_plan("play", {"site": "youtube"})
        self.assertTrue(result.ok)
        self.assertEqual(self.browser.calls[-1], ("play", {"site": "youtube"}))


if __name__ == "__main__":
    unittest.main()
