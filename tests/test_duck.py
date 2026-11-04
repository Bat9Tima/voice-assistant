from __future__ import annotations

import unittest

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
    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class FakeSession:
    def __init__(self, process: str, volume: float = 0.7, muted: bool = False) -> None:
        self.Process = FakeProcess(process)
        self.SimpleAudioVolume = FakeAudio(volume, muted)


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


if __name__ == "__main__":
    unittest.main()
