from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from typing import Any

log = logging.getLogger(__name__)


def _windows_sessions() -> Iterable[Any]:
    """Все аудиосессии Windows; на не-Windows или без pycaw — пусто."""
    try:
        from pycaw.pycaw import AudioUtilities
    except Exception:  # noqa: BLE001
        log.debug("pycaw недоступен, приглушение выключено", exc_info=True)
        return []
    try:
        return list(AudioUtilities.GetAllSessions())
    except Exception:  # noqa: BLE001
        log.debug("Не удалось получить аудиосессии", exc_info=True)
        return []


class AppVolumeDucker(AbstractContextManager["AppVolumeDucker"]):
    """Временно приглушает выбранные приложения на время записи команды.

    Музыка из браузера попадает в микрофон и портит распознавание. Громкость
    меняется на уровне аудиосессии приложения, поэтому системная громкость,
    громкость плеера и mute не затрагиваются. Исходные значения возвращаются
    в `restore()` даже при ошибке записи.
    """

    def __init__(
        self,
        process_names: Iterable[str] = ("chrome.exe",),
        level: float = 0.15,
        session_source: Callable[[], Iterable[Any]] | None = None,
    ) -> None:
        self.process_names = {
            str(name).strip().lower() for name in process_names if str(name).strip()
        }
        self.level = max(0.0, min(1.0, float(level)))
        self._session_source = session_source or _windows_sessions
        self._saved: list[tuple[Any, float, bool]] = []
        self._active = False

    def __enter__(self) -> "AppVolumeDucker":
        self.duck()
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.restore()
        return False

    def _matching_sessions(self) -> Iterable[Any]:
        for session in self._session_source():
            process = getattr(session, "Process", None)
            if process is None:
                continue
            try:
                name = str(process.name()).lower()
            except Exception:  # noqa: BLE001
                continue
            if name in self.process_names:
                yield session

    def duck(self) -> int:
        """Приглушает подходящие сессии и возвращает их число."""
        if self._active or self.level >= 1.0:
            return len(self._saved)
        self._saved.clear()
        for session in self._matching_sessions():
            try:
                volume = session.SimpleAudioVolume
                current = float(volume.GetMasterVolume())
                muted = bool(volume.GetMute())
                volume.SetMasterVolume(self.level, None)
                self._saved.append((volume, current, muted))
            except Exception:  # noqa: BLE001
                log.debug("Не удалось приглушить аудиосессию", exc_info=True)
        self._active = True
        if self._saved:
            log.info(
                "Приглушил звук приложений на время команды: %d сесс., %.0f%%",
                len(self._saved),
                self.level * 100,
            )
        return len(self._saved)

    def restore(self) -> None:
        """Возвращает громкость и mute, снятые в последнем `duck()`."""
        for volume, current, muted in reversed(self._saved):
            try:
                volume.SetMasterVolume(current, None)
                volume.SetMute(int(muted), None)
            except Exception:  # noqa: BLE001
                log.debug("Не удалось вернуть громкость сессии", exc_info=True)
        if self._saved:
            log.info("Вернул звук приложений после команды")
        self._saved.clear()
        self._active = False
