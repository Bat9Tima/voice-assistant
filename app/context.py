from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

DEFAULT_TTL_SEC = 180.0


@dataclass(frozen=True)
class ContextEntry:
    """Одно явно запомненное продолжение разговора."""

    action: str
    slots: Mapping[str, Any] = field(default_factory=dict)
    at: float = 0.0


class CommandContext:
    """Короткий контекст команд: поиск, медиа, выдача, громкость.

    Контекст намеренно разделён по видам, а не хранит «последнюю команду».
    Иначе системная громкость затирала бы последний поиск, и «ещё один»
    перестало бы работать. Каждый вид живёт отдельно и истекает по TTL.
    """

    KINDS = {"search", "media", "results", "volume"}

    def __init__(
        self,
        ttl_sec: float = DEFAULT_TTL_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_sec = max(1.0, float(ttl_sec))
        self._clock = clock
        self._entries: dict[str, ContextEntry] = {}

    def remember(self, kind: str, action: str, **slots: Any) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"неизвестный вид контекста: {kind}")
        self._entries[kind] = ContextEntry(action, dict(slots), self._clock())

    def get(self, kind: str) -> ContextEntry | None:
        entry = self._entries.get(kind)
        if entry is None:
            return None
        if self._clock() - entry.at > self.ttl_sec:
            self._entries.pop(kind, None)
            return None
        return entry

    def matches(self, kind: str) -> bool:
        return self.get(kind) is not None

    def clear(self) -> None:
        self._entries.clear()
