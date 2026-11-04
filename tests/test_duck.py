from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.duck import AppVolumeDucker


class FakeAudio:
    def __init__(self, volume: float = 0.7, muted: bool = False) -> None:
        self.volume = volume
        self.muted = muted
        self.fail = False

    def GetMasterVolume(self) -> float:
        return self.volume

    def SetMasterVolume(self, value: float, context=None) -> None:
        if self.fail:
            raise RuntimeError("session closed")
        self.volume = value

    def GetMute(self) -> int:
        return int(self.muted)

    def SetMute(self, value: int, context=None) -> None:
        if self.fail:
            raise RuntimeError("session closed")
        self.muted = bool(value)


class FakeProcess:
    def __init__(self, name: str, pid: int | None = None) -> None:
        self._name = name
        self.pid = pid

    def name(self) -> str:
        return self._name


class FakeSession:
    def __init__(
        self,
        process: str,
        volume: float = 0.7,
        muted: bool = False,
        identifier: str | None = None,
    ) -> None:
        self.Process = FakeProcess(process)
        self.SimpleAudioVolume = FakeAudio(volume, muted)
        if identifier is not None:
            self.InstanceIdentifier = identifier


class BrokenSession:
    Process = None
    SimpleAudioVolume = FakeAudio()


class AppVolumeDuckerTest(unittest.TestCase):
    def make(self, sessions, level=0.15, names=("chrome.exe",)):
        return AppVolumeDucker(
            process_names=names,
            level=level,
            session_source=lambda: list(sessions),
        )

    def test_ducks_only_target_process(self) -> None:
        chrome = FakeSession("chrome.exe", 0.8)
        spotify = FakeSession("spotify.exe", 0.6)
        ducker = self.make([chrome, spotify])

        self.assertEqual(ducker.duck(), 1)
        self.assertAlmostEqual(chrome.SimpleAudioVolume.volume, 0.15)
        self.assertAlmostEqual(spotify.SimpleAudioVolume.volume, 0.6)

        ducker.restore()
        self.assertAlmostEqual(chrome.SimpleAudioVolume.volume, 0.8)

    def test_restores_mute_state(self) -> None:
        chrome = FakeSession("chrome.exe", 0.5, muted=True)
        ducker = self.make([chrome])

        ducker.duck()
        self.assertTrue(chrome.SimpleAudioVolume.muted)
        ducker.restore()
        self.assertAlmostEqual(chrome.SimpleAudioVolume.volume, 0.5)
        self.assertTrue(chrome.SimpleAudioVolume.muted)

    def test_handles_multiple_sessions(self) -> None:
        first = FakeSession("chrome.exe", 0.9)
        second = FakeSession("chrome.exe", 0.3)
        ducker = self.make([first, second])

        self.assertEqual(ducker.duck(), 2)
        self.assertAlmostEqual(first.SimpleAudioVolume.volume, 0.15)
        self.assertAlmostEqual(second.SimpleAudioVolume.volume, 0.15)

        ducker.restore()
        self.assertAlmostEqual(first.SimpleAudioVolume.volume, 0.9)
        self.assertAlmostEqual(second.SimpleAudioVolume.volume, 0.3)

    def test_context_manager_restores_on_error(self) -> None:
        chrome = FakeSession("chrome.exe", 0.8)
        ducker = self.make([chrome])

        with self.assertRaises(RuntimeError):
            with ducker:
                self.assertAlmostEqual(chrome.SimpleAudioVolume.volume, 0.15)
                raise RuntimeError("boom")

        self.assertAlmostEqual(chrome.SimpleAudioVolume.volume, 0.8)

    def test_skips_sessions_without_process(self) -> None:
        chrome = FakeSession("chrome.exe", 0.8)
        ducker = self.make([BrokenSession(), chrome])
        self.assertEqual(ducker.duck(), 1)
        ducker.restore()

    def test_broken_session_does_not_block_others(self) -> None:
        broken = FakeSession("chrome.exe", 0.8)
        broken.SimpleAudioVolume.fail = True
        working = FakeSession("chrome.exe", 0.4)
        ducker = self.make([broken, working])

        self.assertEqual(ducker.duck(), 1)
        self.assertAlmostEqual(working.SimpleAudioVolume.volume, 0.15)
        ducker.restore()
        self.assertAlmostEqual(working.SimpleAudioVolume.volume, 0.4)

    def test_duck_is_idempotent(self) -> None:
        chrome = FakeSession("chrome.exe", 0.8)
        ducker = self.make([chrome])

        ducker.duck()
        ducker.duck()
        ducker.restore()
        self.assertAlmostEqual(chrome.SimpleAudioVolume.volume, 0.8)

    def test_level_is_clamped(self) -> None:
        chrome = FakeSession("chrome.exe", 0.8)
        ducker = self.make([chrome], level=2.0)
        self.assertEqual(ducker.duck(), 0)
        self.assertAlmostEqual(chrome.SimpleAudioVolume.volume, 0.8)

        ducker = self.make([chrome], level=-1.0)
        ducker.duck()
        self.assertAlmostEqual(chrome.SimpleAudioVolume.volume, 0.0)
        ducker.restore()

    def test_restore_without_duck_is_safe(self) -> None:
        self.make([]).restore()

    def test_process_names_are_case_insensitive(self) -> None:
        chrome = FakeSession("Chrome.EXE", 0.8)
        ducker = self.make([chrome], names=("CHROME.exe",))
        self.assertEqual(ducker.duck(), 1)


class DuckStateTest(unittest.TestCase):
    """Громкость не должна «застревать» на 15% после падения ассистента."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name) / "duck-state.json"

    def make(self, sessions, level=0.15, state=True):
        return AppVolumeDucker(
            process_names=("chrome.exe",),
            level=level,
            session_source=lambda: list(sessions),
            state_path=self.state if state else None,
        )

    def test_state_is_written_before_duck_and_removed_after_restore(self) -> None:
        chrome = FakeSession("chrome.exe", 0.8, identifier="chrome-1")
        ducker = self.make([chrome])

        ducker.duck()
        self.assertTrue(self.state.exists())
        payload = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual(payload["sessions"][0]["key"], "chrome-1")
        self.assertAlmostEqual(payload["sessions"][0]["volume"], 0.8)

        ducker.restore()
        self.assertFalse(self.state.exists())
        self.assertAlmostEqual(chrome.SimpleAudioVolume.volume, 0.8)

    def test_recover_after_crash_returns_original_volume(self) -> None:
        chrome = FakeSession("chrome.exe", 0.8, identifier="chrome-1")
        sessions = [chrome]
        first = self.make(sessions)
        first.duck()
        self.assertAlmostEqual(chrome.SimpleAudioVolume.volume, 0.15)

        # Ассистент упал/был перезапущен, restore() не вызван.
        second = self.make(sessions)
        self.assertEqual(second.recover(), 1)
        self.assertAlmostEqual(chrome.SimpleAudioVolume.volume, 0.8)
        self.assertFalse(self.state.exists())

    def test_duck_after_crash_saves_true_original_not_ducked_level(self) -> None:
        chrome = FakeSession("chrome.exe", 0.8, identifier="chrome-1")
        sessions = [chrome]
        first = self.make(sessions)
        first.duck()

        second = self.make(sessions)
        self.assertEqual(second.duck(), 1)
        self.assertAlmostEqual(chrome.SimpleAudioVolume.volume, 0.15)

        second.restore()
        self.assertAlmostEqual(chrome.SimpleAudioVolume.volume, 0.8)

    def test_recover_does_not_override_user_volume_change(self) -> None:
        chrome = FakeSession("chrome.exe", 0.8, identifier="chrome-1")
        sessions = [chrome]
        first = self.make(sessions)
        first.duck()
        chrome.SimpleAudioVolume.volume = 0.45

        second = self.make(sessions)
        self.assertEqual(second.recover(), 0)
        self.assertAlmostEqual(chrome.SimpleAudioVolume.volume, 0.45)
        self.assertFalse(self.state.exists())

    def test_restore_finds_session_recreated_with_same_identifier(self) -> None:
        old = FakeSession("chrome.exe", 0.8, identifier="chrome-1")
        sessions = [old]
        ducker = self.make(sessions)
        ducker.duck()

        new = FakeSession("chrome.exe", 0.15, identifier="chrome-1")
        sessions[:] = [new]
        old.SimpleAudioVolume.fail = True

        ducker.restore()
        self.assertAlmostEqual(new.SimpleAudioVolume.volume, 0.8)

    def test_restore_matches_recreated_session_by_process(self) -> None:
        old = FakeSession("chrome.exe", 0.8, identifier="old")
        sessions = [old]
        ducker = self.make(sessions)
        ducker.duck()

        new = FakeSession("chrome.exe", 0.15, identifier="new")
        sessions[:] = [new]
        old.SimpleAudioVolume.fail = True

        ducker.restore()
        self.assertAlmostEqual(new.SimpleAudioVolume.volume, 0.8)

    def test_failed_restore_keeps_state_for_next_launch(self) -> None:
        chrome = FakeSession("chrome.exe", 0.8, identifier="chrome-1")
        sessions = [chrome]
        ducker = self.make(sessions)
        ducker.duck()

        chrome.SimpleAudioVolume.fail = True
        sessions.clear()
        with self.assertLogs("app.duck", level="WARNING"):
            ducker.restore()
        self.assertTrue(self.state.exists())

        fresh = FakeSession("chrome.exe", 0.15, identifier="chrome-1")
        sessions.append(fresh)
        second = self.make(sessions)
        self.assertEqual(second.recover(), 1)
        self.assertAlmostEqual(fresh.SimpleAudioVolume.volume, 0.8)
        self.assertFalse(self.state.exists())

    def test_restore_keeps_pending_entry_for_missing_session(self) -> None:
        first = FakeSession("chrome.exe", 0.9, identifier="chrome-1")
        second = FakeSession("chrome.exe", 0.3, identifier="chrome-2")
        sessions = [first, second]
        ducker = self.make(sessions)
        ducker.duck()

        sessions[:] = [first]
        second.SimpleAudioVolume.fail = True
        ducker.restore()
        self.assertAlmostEqual(first.SimpleAudioVolume.volume, 0.9)

        payload = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual([item["key"] for item in payload["sessions"]], ["chrome-2"])

    def test_recover_waits_for_missing_session(self) -> None:
        chrome = FakeSession("chrome.exe", 0.8, identifier="chrome-1")
        sessions = [chrome]
        self.make(sessions).duck()

        sessions.clear()
        self.assertEqual(self.make(sessions).recover(), 0)
        self.assertTrue(self.state.exists())

        fresh = FakeSession("chrome.exe", 0.15, identifier="chrome-1")
        sessions.append(fresh)
        self.assertEqual(self.make(sessions).recover(), 1)
        self.assertAlmostEqual(fresh.SimpleAudioVolume.volume, 0.8)
        self.assertFalse(self.state.exists())

    def test_duck_without_session_keeps_pending_original(self) -> None:
        chrome = FakeSession("chrome.exe", 0.8, identifier="chrome-1")
        sessions = [chrome]
        self.make(sessions).duck()

        sessions.clear()
        ducker = self.make(sessions)
        self.assertEqual(ducker.duck(), 0)
        self.assertTrue(self.state.exists())

        fresh = FakeSession("chrome.exe", 0.15, identifier="chrome-1")
        sessions.append(fresh)
        second = self.make(sessions)
        second.duck()  # recover() вернёт 0.8, затем снова приглушит
        second.restore()
        self.assertAlmostEqual(fresh.SimpleAudioVolume.volume, 0.8)

    def test_recover_restores_multiple_sessions(self) -> None:
        first = FakeSession("chrome.exe", 0.9, identifier="chrome-1")
        second = FakeSession("chrome.exe", 0.3, identifier="chrome-2")
        sessions = [first, second]
        self.make(sessions).duck()

        self.assertEqual(self.make(sessions).recover(), 2)
        self.assertAlmostEqual(first.SimpleAudioVolume.volume, 0.9)
        self.assertAlmostEqual(second.SimpleAudioVolume.volume, 0.3)

    def test_without_state_path_restore_still_works(self) -> None:
        chrome = FakeSession("chrome.exe", 0.8)
        ducker = self.make([chrome], state=False)
        ducker.duck()
        ducker.restore()
        self.assertAlmostEqual(chrome.SimpleAudioVolume.volume, 0.8)


if __name__ == "__main__":
    unittest.main()
