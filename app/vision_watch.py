"""Режим наблюдения за экраном для ассистента (v2.0).

Поток раз в ``interval_sec`` снимает кадр, сравнивает его с предыдущим и
держит последний кадр в памяти. Модели не вызываются, на диск ничего не
пишется, в облако ничего не уходит.

Режим включается только явно (командой «включи зрение»), выключается
командой, паузой микрофона или по таймеру ``auto_off_sec``. С v4.0 сюда
же передаётся ``EventTimeline``: наблюдатель записывает в неё события,
а пауза и остановка ленту забывают.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .human_time import format_duration, plural
from .screen_capture import Frame
from .screen_watch import ScreenWatcher, WatchEvent
from .vision_events import EventTimeline

log = logging.getLogger(__name__)

DEFAULT_INTERVAL_SEC = 2.0
DEFAULT_AUTO_OFF_SEC = 600.0
MIN_INTERVAL_SEC = 0.25
JOIN_TIMEOUT_SEC = 2.0

StopCallback = Callable[[str], None]


@dataclass(frozen=True)
class WatchStatus:
    """Снимок состояния режима наблюдения."""

    active: bool
    paused: bool
    started_at: float | None
    last_poll_at: float | None
    last_change_at: float | None
    polls: int
    changes: int
    window_title: str | None
    error: str | None
    interval_sec: float
    auto_off_sec: float


class VisionObserver:
    """Периодически снимает экран, пока режим включён явно."""

    def __init__(
        self,
        watcher: ScreenWatcher,
        *,
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        auto_off_sec: float = DEFAULT_AUTO_OFF_SEC,
        clock: Callable[[], float] = time.monotonic,
        on_stop: StopCallback | None = None,
        timeline: EventTimeline | None = None,
    ) -> None:
        self.watcher = watcher
        self.timeline = timeline
        self.interval_sec = max(MIN_INTERVAL_SEC, float(interval_sec))
        self.auto_off_sec = max(0.0, float(auto_off_sec))
        self._clock = clock
        self._on_stop = on_stop
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._paused = False
        self._started_at: float | None = None
        self._last_poll_at: float | None = None
        self._last_change_at: float | None = None
        self._polls = 0
        self._changes = 0
        self._window_title: str | None = None
        self._error: str | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def last_frame(self) -> Frame | None:
        """Последний снятый кадр; после остановки — ``None``."""
        with self._lock:
            return self.watcher.last_frame

    def start(self, *, background: bool = True) -> bool:
        """Включает режим. ``False`` — уже был включён.

        ``background=False`` не запускает поток: шаги делает вызывающий
        через ``tick()``. Нужно для детерминированных тестов.
        """
        with self._lock:
            if self._active:
                return False
            self.watcher.reset()
            if self.timeline is not None:
                self.timeline.reset()
            self._active = True
            self._paused = False
            self._started_at = self._clock()
            self._last_poll_at = None
            self._last_change_at = None
            self._polls = 0
            self._changes = 0
            self._window_title = None
            self._error = None
            self._stop_event.clear()
            thread = None
            if background:
                thread = threading.Thread(
                    target=self._run,
                    name="vision-watch",
                    daemon=True,
                )
                self._thread = thread
        if thread is not None:
            thread.start()
        log.info(
            "Наблюдение включено: интервал %.1f с, автоотключение %.0f с",
            self.interval_sec,
            self.auto_off_sec,
        )
        return True

    def stop(self) -> bool:
        """Выключает режим и забывает кадры. ``False`` — уже был выключен."""
        with self._lock:
            was_active = self._active
            self._stop_locked()
            thread = self._thread
            self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=JOIN_TIMEOUT_SEC)
        if was_active:
            log.info("Наблюдение выключено")
        return was_active

    def pause(self) -> bool:
        """Приостанавливает захват, не выключая режим."""
        with self._lock:
            if not self._active or self._paused:
                return False
            self._paused = True
            self.watcher.reset()
            if self.timeline is not None:
                self.timeline.reset()
            self._last_poll_at = None
            self._window_title = None
            log.info("Наблюдение на паузе")
            return True

    def resume(self) -> bool:
        """Возвращает захват после паузы. Следующий кадр — новый старт."""
        with self._lock:
            if not self._active or not self._paused:
                return False
            self._paused = False
            self._last_poll_at = None
            log.info("Наблюдение продолжено")
            return True

    def tick(self, now: float | None = None) -> WatchEvent | None:
        """Один шаг: снять кадр, если подошёл интервал.

        Возвращает событие только когда кадр действительно снят. Проверка
        автоотключения идёт раньше интервала, чтобы режим не жил лишний шаг.
        """
        timeout = False
        event: WatchEvent | None = None
        with self._lock:
            if not self._active:
                return None
            current = self._clock() if now is None else float(now)
            started = self._started_at if self._started_at is not None else current
            if self.auto_off_sec > 0 and current - started >= self.auto_off_sec:
                self._stop_locked()
                timeout = True
            elif self._paused:
                # Пауза не должна продлевать жизнь режима: таймер идёт
                # от включения, даже если захват приостановлен.
                return None
            elif (
                self._last_poll_at is not None and current - self._last_poll_at < self.interval_sec
            ):
                return None
            else:
                try:
                    event = self.watcher.poll()
                except Exception as exc:  # noqa: BLE001
                    error = str(exc) or exc.__class__.__name__
                    if error != self._error:
                        # Одна и та же ошибка каждые двадцать секунд не должна
                        # забивать лог: пишем её при появлении и смене.
                        log.warning("Наблюдение: захват не удался: %s", error)
                    self._error = error
                    self._last_poll_at = current
                    return None
                self._error = None
                self._polls += 1
                self._last_poll_at = current
                self._window_title = event.window_title
                if event.changed:
                    self._changes += 1
                    self._last_change_at = current
                if self.timeline is not None:
                    self.timeline.record(
                        event,
                        now=current,
                        frame=self.watcher.last_frame,
                    )
        if timeout:
            log.info("Наблюдение выключено: время вышло")
            self._report_stop("timeout")
            return None
        return event

    def status(self) -> WatchStatus:
        with self._lock:
            return WatchStatus(
                active=self._active,
                paused=self._paused,
                started_at=self._started_at,
                last_poll_at=self._last_poll_at,
                last_change_at=self._last_change_at,
                polls=self._polls,
                changes=self._changes,
                window_title=self._window_title,
                error=self._error,
                interval_sec=self.interval_sec,
                auto_off_sec=self.auto_off_sec,
            )

    # ------------------------------------------------------------- internals
    def _stop_locked(self) -> None:
        self._active = False
        self._paused = False
        self._stop_event.set()
        self.watcher.reset()
        if self.timeline is not None:
            self.timeline.reset()
        self._window_title = None

    def _report_stop(self, reason: str) -> None:
        callback = self._on_stop
        if callback is None:
            return
        try:
            callback(reason)
        except Exception:  # noqa: BLE001
            log.exception("Наблюдение: обработчик остановки упал")

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self.tick()
                if self._stop_event.wait(self.interval_sec):
                    break
        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None


def _short_title(title: str, limit: int = 60) -> str:
    cleaned = " ".join(title.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def describe_status(status: WatchStatus, *, now: float | None = None) -> str:
    """Короткий отчёт о режиме для голоса и уведомления."""
    if not status.active:
        return "Наблюдение выключено."
    if status.paused:
        return "Наблюдение на паузе."
    current = time.monotonic() if now is None else float(now)
    started = status.started_at if status.started_at is not None else current
    parts = [f"Наблюдение включено, работает {format_duration(current - started)}."]
    if status.polls <= 1:
        parts.append("Экран ещё не сравнивался.")
    elif status.changes == 0:
        parts.append("Экран пока не менялся.")
    else:
        times = plural(status.changes, "раз", "раза", "раз")
        parts.append(f"Экран менялся {status.changes} {times}.")
        if status.last_change_at is not None:
            age = max(0.0, current - status.last_change_at)
            parts.append(f"Последнее изменение {format_duration(age)} назад.")
    if status.window_title:
        parts.append(f"Активное окно: {_short_title(status.window_title)}.")
    if status.error:
        parts.append(f"Последняя ошибка захвата: {status.error}.")
    return " ".join(parts)


__all__ = [
    "DEFAULT_AUTO_OFF_SEC",
    "DEFAULT_INTERVAL_SEC",
    "MIN_INTERVAL_SEC",
    "VisionObserver",
    "WatchStatus",
    "describe_status",
    "format_duration",
]
