"""Лента событий наблюдения за экраном (v4.0).

Слой поверх ``ScreenWatcher``: превращает шаги наблюдения в короткую
историю — старт, смена окна, изменение экрана, долгий простой. Модели не
вызываются, кадры не сохраняются, на диск ничего не пишется. Лента живёт
в памяти, пока режим наблюдения включён, и забывается при паузе или
остановке.

Это фундамент v4.0: по нему можно ответить «что происходило» без модели,
а следующие шаги добавят к событиям ключевые кадры и уместные комментарии.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

from .human_time import format_duration, plural
from .screen_watch import WatchEvent
from .text import short_title

KIND_START = "start"
KIND_WINDOW = "window"
KIND_CHANGE = "change"
KIND_IDLE = "idle"

DEFAULT_MAX_EVENTS = 40
DEFAULT_IDLE_AFTER_SEC = 120.0
DEFAULT_COALESCE_SEC = 10.0


@dataclass(frozen=True)
class VisionEvent:
    """Одно событие ленты. Кадр не хранится: только факт и окно."""

    kind: str
    at: float
    window_title: str | None = None
    previous_title: str | None = None
    count: int = 1
    detail: str = ""


class EventTimeline:
    """Потокобезопасная лента событий с ограниченной длиной."""

    def __init__(
        self,
        *,
        max_events: int = DEFAULT_MAX_EVENTS,
        idle_after_sec: float = DEFAULT_IDLE_AFTER_SEC,
        coalesce_sec: float = DEFAULT_COALESCE_SEC,
    ) -> None:
        self.max_events = max(1, int(max_events))
        self.idle_after_sec = max(0.0, float(idle_after_sec))
        self.coalesce_sec = max(0.0, float(coalesce_sec))
        self._lock = threading.RLock()
        self._events: list[VisionEvent] = []
        self._window_title: str | None = None
        self._last_activity_at: float | None = None
        self._idle_reported = False

    def reset(self) -> None:
        """Забывает историю: новый старт наблюдения начинается с чистого листа."""
        with self._lock:
            self._events.clear()
            self._window_title = None
            self._last_activity_at = None
            self._idle_reported = False

    def record(self, event: WatchEvent, *, now: float | None = None) -> VisionEvent | None:
        """Добавляет событие по шагу наблюдения.

        Возвращает записанное событие или ``None``, если шаг ничего
        заметного не дал. Смена окна важнее изменения картинки; частые
        изменения в одном окне склеиваются, чтобы лента не состояла из
        одинаковых строк.
        """
        current = float(event.at if now is None else now)
        with self._lock:
            if event.result is None:
                self._events.clear()
                self._window_title = event.window_title
                self._last_activity_at = current
                self._idle_reported = False
                return self._append(
                    VisionEvent(
                        KIND_START,
                        current,
                        window_title=event.window_title,
                    )
                )

            previous = self._window_title
            if event.window_changed:
                kind = KIND_WINDOW
            elif event.changed:
                kind = KIND_CHANGE
            else:
                kind = ""

            if kind:
                self._window_title = event.window_title
                self._last_activity_at = current
                self._idle_reported = False
                if kind == KIND_CHANGE and self._coalesce(current, event.window_title):
                    return self._events[-1]
                return self._append(
                    VisionEvent(
                        kind,
                        current,
                        window_title=event.window_title,
                        previous_title=previous if event.window_changed else None,
                    )
                )

            self._window_title = event.window_title
            if self._should_report_idle(current):
                self._idle_reported = True
                idle_for = current - (self._last_activity_at or current)
                return self._append(
                    VisionEvent(
                        KIND_IDLE,
                        current,
                        window_title=event.window_title,
                        detail=format_duration(idle_for),
                    )
                )
            return None

    def events(self, *, limit: int | None = None) -> tuple[VisionEvent, ...]:
        """События по порядку; ``limit`` оставляет только последние."""
        with self._lock:
            items = tuple(self._events)
        if limit is None:
            return items
        if limit <= 0:
            return ()
        return items[-limit:]

    def describe(self, *, now: float | None = None, limit: int = 6) -> str:
        """Короткий рассказ о событиях без модели."""
        with self._lock:
            total = len(self._events)
            items = tuple(self._events[-max(1, int(limit)) :])
        if not items:
            return "Событий пока нет."
        current = time.monotonic() if now is None else float(now)
        parts = [f"{_when(current, event.at)} — {_phrase(event)}" for event in items]
        if total > len(items):
            count = len(items)
            word = plural(count, "событие", "события", "событий")
            prefix = f"Последние {count} {word}: "
        else:
            prefix = "Что происходило: "
        return prefix + "; ".join(parts) + "."

    # ------------------------------------------------------------- internals
    def _append(self, event: VisionEvent) -> VisionEvent:
        self._events.append(event)
        if len(self._events) > self.max_events:
            del self._events[: len(self._events) - self.max_events]
        return event

    def _coalesce(self, current: float, title: str | None) -> bool:
        if self.coalesce_sec <= 0 or not self._events:
            return False
        last = self._events[-1]
        if last.kind != KIND_CHANGE or last.window_title != title:
            return False
        if current - last.at > self.coalesce_sec:
            return False
        self._events[-1] = replace(last, at=current, count=last.count + 1)
        return True

    def _should_report_idle(self, current: float) -> bool:
        if self.idle_after_sec <= 0 or self._idle_reported:
            return False
        if self._last_activity_at is None:
            return False
        return current - self._last_activity_at >= self.idle_after_sec


def _when(now: float, at: float) -> str:
    age = max(0.0, now - at)
    if age < 5:
        return "только что"
    return f"{format_duration(age)} назад"


def _phrase(event: VisionEvent) -> str:
    title = short_title(event.window_title) if event.window_title else ""
    if event.kind == KIND_START:
        return f"наблюдение началось, окно «{title}»" if title else "наблюдение началось"
    if event.kind == KIND_WINDOW:
        return f"активное окно — «{title}»" if title else "активное окно сменилось"
    if event.kind == KIND_CHANGE:
        if event.count > 1:
            times = plural(event.count, "раз", "раза", "раз")
            base = f"экран изменился {event.count} {times}"
        else:
            base = "экран изменился"
        return f"{base}, окно «{title}»" if title else base
    if event.kind == KIND_IDLE:
        return f"экран не менялся {event.detail}" if event.detail else "экран долго не менялся"
    return event.kind


__all__ = [
    "DEFAULT_COALESCE_SEC",
    "DEFAULT_IDLE_AFTER_SEC",
    "DEFAULT_MAX_EVENTS",
    "KIND_CHANGE",
    "KIND_IDLE",
    "KIND_START",
    "KIND_WINDOW",
    "EventTimeline",
    "VisionEvent",
]
