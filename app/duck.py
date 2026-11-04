from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_STATE_VERSION = 1


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


@dataclass
class _Target:
    """Найденная аудиосессия приложения."""

    audio: Any
    key: str
    process: str


@dataclass
class _SavedVolume:
    """Громкость, которую нужно вернуть после команды."""

    audio: Any
    key: str
    process: str
    volume: float
    muted: bool


class AppVolumeDucker(AbstractContextManager["AppVolumeDucker"]):
    """Временно приглушает выбранные приложения на время записи команды.

    Музыка из браузера попадает в микрофон и портит распознавание. Громкость
    меняется на уровне аудиосессии приложения, поэтому системная громкость,
    громкость плеера и mute не затрагиваются. Исходные значения возвращаются
    в `restore()` даже при ошибке записи.

    Если ассистент был перезапущен или упал во время приглушения, исходная
    громкость не должна потеряться: иначе при следующем запуске 15% будут
    сохранены как «исходные» и звук останется тихим навсегда. Поэтому перед
    изменением громкости исходные значения атомарно пишутся в `state_path`,
    а `recover()` возвращает их при следующем запуске.
    """

    def __init__(
        self,
        process_names: Iterable[str] = ("chrome.exe",),
        level: float = 0.15,
        session_source: Callable[[], Iterable[Any]] | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        self.process_names = {
            str(name).strip().lower() for name in process_names if str(name).strip()
        }
        self.level = max(0.0, min(1.0, float(level)))
        self._session_source = session_source or _windows_sessions
        self.state_path = Path(state_path) if state_path is not None else None
        self._saved: list[_SavedVolume] = []
        self._active = False
        self._lock = threading.RLock()

    def __enter__(self) -> "AppVolumeDucker":
        self.duck()
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.restore()
        return False

    # ---------------------------------------------------------------- sessions
    def _targets(self) -> list[_Target]:
        result: list[_Target] = []
        for session in self._session_source():
            process = getattr(session, "Process", None)
            if process is None:
                continue
            try:
                name = str(process.name()).lower()
            except Exception:  # noqa: BLE001
                continue
            if name not in self.process_names:
                continue
            audio = getattr(session, "SimpleAudioVolume", None)
            if audio is None:
                continue

            key = ""
            identifier = getattr(session, "InstanceIdentifier", None)
            if identifier:
                key = str(identifier)
            else:
                pid = getattr(process, "pid", None)
                if pid is not None:
                    key = f"{name}:{pid}"
            result.append(_Target(audio=audio, key=key, process=name))
        return result

    @staticmethod
    def _pair(
        originals: list[_SavedVolume], targets: list[_Target]
    ) -> list[tuple[_SavedVolume, _Target | None]]:
        """Сопоставляет сохранённые значения с текущими сессиями.

        Сначала по стабильному идентификатору. Если сессия была пересоздана
        (например, Chrome перезапустил аудиопоток), идентификатор меняется —
        тогда сопоставляем по процессу, но только когда это однозначно.
        """
        remaining = list(targets)
        pairs: list[tuple[_SavedVolume, _Target | None]] = []
        for original in originals:
            match = None
            if original.key:
                match = next(
                    (target for target in remaining if target.key == original.key),
                    None,
                )
            if match is not None:
                remaining.remove(match)
            pairs.append((original, match))

        for index, (original, match) in enumerate(pairs):
            if match is not None or not original.process:
                continue
            candidates = [
                target for target in remaining if target.process == original.process
            ]
            same_process = [
                item
                for item in pairs
                if item[1] is None and item[0].process == original.process
            ]
            if len(candidates) == 1 and len(same_process) == 1:
                chosen = candidates[0]
                remaining.remove(chosen)
                pairs[index] = (original, chosen)
        return pairs

    # ------------------------------------------------------------------- state
    def _load_state(self) -> list[_SavedVolume]:
        if self.state_path is None or not self.state_path.exists():
            return []
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            log.warning(
                "Не удалось прочитать состояние приглушения: %s",
                self.state_path,
                exc_info=True,
            )
            return []
        if not isinstance(data, dict):
            return []
        try:
            version = int(data.get("version", 0))
        except (TypeError, ValueError):
            version = 0
        if version != _STATE_VERSION:
            return []

        result: list[_SavedVolume] = []
        for item in data.get("sessions") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "")
            if not key:
                continue
            try:
                volume = float(item["volume"])
                muted = bool(item.get("muted", False))
            except (KeyError, TypeError, ValueError):
                continue
            result.append(
                _SavedVolume(
                    audio=None,
                    key=key,
                    process=str(item.get("process") or "").lower(),
                    volume=volume,
                    muted=muted,
                )
            )
        return result

    def _write_state(self, saved: Iterable[_SavedVolume]) -> None:
        if self.state_path is None:
            return
        entries = [
            {
                "key": item.key,
                "process": item.process,
                "volume": item.volume,
                "muted": item.muted,
            }
            for item in saved
            if item.key
        ]
        if not entries:
            self._clear_state()
            return
        payload = {
            "version": _STATE_VERSION,
            "level": self.level,
            "updated": time.time(),
            "sessions": entries,
        }
        temporary = self.state_path.with_name(self.state_path.name + ".tmp")
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, self.state_path)
        except Exception:  # noqa: BLE001
            log.warning(
                "Не удалось сохранить состояние приглушения: %s",
                self.state_path,
                exc_info=True,
            )

    def _clear_state(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            log.debug("Не удалось удалить состояние приглушения", exc_info=True)

    # ------------------------------------------------------------------ public
    def recover(self) -> int:
        """Возвращает громкость, оставшуюся от прерванного приглушения.

        Вызывается при запуске ассистента и перед новым приглушением. Если
        пользователь успел сам поменять громкость, его выбор не перебивается.
        """
        with self._lock:
            if self._active:
                return 0
            saved = self._load_state()
            if not saved:
                self._clear_state()
                return 0

            restored = 0
            remaining: list[_SavedVolume] = []
            for original, target in self._pair(saved, self._targets()):
                if target is None:
                    # Chrome закрыт или ещё не создал аудиосессию (например,
                    # после перезагрузки). Оставляем запись: вернём громкость,
                    # когда сессия появится.
                    remaining.append(original)
                    continue
                try:
                    current = float(target.audio.GetMasterVolume())
                except Exception:  # noqa: BLE001
                    remaining.append(original)
                    continue
                if abs(current - self.level) > 0.02:
                    # Громкость уже не наша — не трогаем выбор пользователя.
                    continue
                try:
                    target.audio.SetMasterVolume(original.volume, None)
                    target.audio.SetMute(int(original.muted), None)
                    restored += 1
                except Exception:  # noqa: BLE001
                    log.debug("Не удалось восстановить аудиосессию", exc_info=True)
                    remaining.append(original)

            if remaining:
                self._write_state(remaining)
            else:
                self._clear_state()
            if restored:
                log.info(
                    "Восстановил звук приложений после прежнего запуска: %d сесс.",
                    restored,
                )
            return restored

    def duck(self) -> int:
        """Приглушает подходящие сессии и возвращает их число."""
        with self._lock:
            if self._active or self.level >= 1.0:
                return len(self._saved)

            # Хвост прошлого запуска: если ассистент упал во время команды,
            # текущие 15% — это не исходная громкость.
            self.recover()
            leftovers = {item.key: item for item in self._load_state() if item.key}

            saved: list[_SavedVolume] = []
            for target in self._targets():
                try:
                    current = float(target.audio.GetMasterVolume())
                    muted = bool(target.audio.GetMute())
                except Exception:  # noqa: BLE001
                    log.debug("Не удалось прочитать громкость сессии", exc_info=True)
                    continue
                previous = leftovers.get(target.key) if target.key else None
                saved.append(
                    _SavedVolume(
                        audio=target.audio,
                        key=target.key,
                        process=target.process,
                        volume=previous.volume if previous is not None else current,
                        muted=previous.muted if previous is not None else muted,
                    )
                )

            # Фиксируем исходные значения ДО изменения громкости: если процесс
            # убьют сразу после SetMasterVolume, recover() всё равно вернёт их.
            # Незавершённые записи для отсутствующих сессий сохраняем.
            current_keys = {item.key for item in saved if item.key}
            pending = [
                item
                for item in self._load_state()
                if item.key and item.key not in current_keys
            ]
            self._write_state([*pending, *saved])
            self._saved = []
            for item in saved:
                try:
                    item.audio.SetMasterVolume(self.level, None)
                    self._saved.append(item)
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
        with self._lock:
            if not self._saved:
                self._active = False
                return

            restored = 0
            pending = {
                item.key: item for item in self._load_state() if item.key
            }
            for original, target in self._pair(self._saved, self._targets()):
                # Сессия могла быть пересоздана: тогда пишем в актуальный
                # объект, а не в протухший COM-интерфейс.
                audio = target.audio if target is not None else original.audio
                try:
                    audio.SetMasterVolume(original.volume, None)
                    audio.SetMute(int(original.muted), None)
                    restored += 1
                    pending.pop(original.key, None)
                except Exception:  # noqa: BLE001
                    log.debug("Не удалось вернуть громкость сессии", exc_info=True)
                    if original.key:
                        pending[original.key] = original

            if pending:
                self._write_state(pending.values())
            else:
                self._clear_state()
            if restored:
                log.info("Вернул звук приложений после команды")
            elif self._saved:
                log.warning(
                    "Не удалось вернуть звук приложений: %d сесс.", len(self._saved)
                )
            self._saved.clear()
            self._active = False
