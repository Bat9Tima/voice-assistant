"""Проактивные комментарии наблюдения (v4.0, шаг 3).

Отдельный, явно включаемый режим поверх ``EventTimeline``. Комментарии
готовятся в фоновом потоке, не чаще ``min_interval_sec`` и не больше
``max_per_hour`` проверок в час. В облако уходит текст события и, если
разрешено, один кадр; на диск ничего не пишется.

Модуль не знает о моделях и ассистенте: подготовку текста и озвучку
передают снаружи. Так политику частоты и паузы можно тестировать без
сети.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from .human_time import format_duration, plural
from .screen_capture import Frame
from .vision_events import KIND_CHANGE, KIND_IDLE, KIND_START, KIND_WINDOW, VisionEvent

log = logging.getLogger(__name__)

DEFAULT_MIN_INTERVAL_SEC = 90.0
DEFAULT_MAX_PER_HOUR = 10
HOUR_SEC = 3600.0
MAX_COMMENT_CHARS = 300
JOIN_TIMEOUT_SEC = 2.0

# Комментируем только смену окна и долгий простой. Частые изменения
# картинки в одном окне слишком шумные: для них остаётся лента и команда
# «сравни кадры».
COMMENT_KINDS = frozenset({KIND_WINDOW, KIND_IDLE})

Compose = Callable[[VisionEvent, Frame | None], str | None]
OnComment = Callable[[str], None]
IsBusy = Callable[[], bool]
Clock = Callable[[], float]


@dataclass(frozen=True)
class CommentStatus:
    """Снимок состояния режима комментариев."""

    available: bool
    active: bool
    paused: bool
    use_frame: bool
    min_interval_sec: float
    max_per_hour: int
    attempts: int
    comments: int
    last_attempt_at: float | None
    last_comment_at: float | None
    next_allowed_at: float | None
    last_error: str | None = None


class ProactiveCommenter:
    """Очередь событий и политика уместных комментариев.

    ``compose`` вызывается в фоновом потоке и возвращает готовый текст или
    ``None``/``SKIP``. ``on_comment`` получает только принятый комментарий;
    его вызывает тот же поток, поэтому озвучка не блокирует наблюдение.
    """

    def __init__(
        self,
        *,
        compose: Compose,
        on_comment: OnComment,
        is_busy: IsBusy | None = None,
        clock: Clock = time.monotonic,
        available: bool = True,
        use_frame: bool = True,
        min_interval_sec: float = DEFAULT_MIN_INTERVAL_SEC,
        max_per_hour: int = DEFAULT_MAX_PER_HOUR,
        queue_size: int = 1,
    ) -> None:
        self._compose = compose
        self._on_comment = on_comment
        self._is_busy = is_busy or (lambda: False)
        self._clock = clock
        self.available = bool(available)
        self.use_frame = bool(use_frame)
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self.max_per_hour = max(0, int(max_per_hour))
        self._lock = threading.RLock()
        self._queue: queue.Queue[tuple[VisionEvent, Frame | None]] = queue.Queue(
            maxsize=max(1, int(queue_size))
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._active = False
        self._paused = False
        self._attempts: deque[float] = deque()
        self._comments = 0
        self._last_attempt_at: float | None = None
        self._last_comment_at: float | None = None
        self._last_error: str | None = None

    # ------------------------------------------------------------------ state
    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def start(self) -> bool:
        """Включает режим. ``False`` — уже включён или запрещён настройкой."""
        with self._lock:
            if not self.available or self._active:
                return False
            self._active = True
            self._paused = False
            self._generation += 1
            generation = self._generation
            stop = threading.Event()
            self._stop = stop
            self._drain_locked()
            thread = self._thread
            if thread is None or not thread.is_alive():
                thread = threading.Thread(
                    target=self._run,
                    args=(generation, stop),
                    name="vision-comments",
                    daemon=True,
                )
                self._thread = thread
        thread.start()
        log.info(
            "Комментарии включены: не чаще %.0f с, до %d проверок в час",
            self.min_interval_sec,
            self.max_per_hour,
        )
        return True

    def stop(self) -> bool:
        """Выключает режим и выбрасывает очередь. ``False`` — уже выключен."""
        with self._lock:
            was_active = self._active
            self._active = False
            self._paused = False
            self._generation += 1
            self._stop.set()
            self._drain_locked()
            thread = self._thread
            self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=JOIN_TIMEOUT_SEC)
        if was_active:
            log.info("Комментарии выключены")
        return was_active

    def pause(self) -> bool:
        """Приостанавливает комментарии, не забывая лимиты часа."""
        with self._lock:
            if not self._active or self._paused:
                return False
            self._paused = True
            self._drain_locked()
            return True

    def resume(self) -> bool:
        with self._lock:
            if not self._active or not self._paused:
                return False
            self._paused = False
            return True

    def enqueue(
        self,
        event: VisionEvent,
        frame: Frame | None = None,
        *,
        now: float | None = None,
    ) -> bool:
        """Ставит событие в очередь. ``False`` — событие не подходит.

        Очередь короткая: если пришло новое событие, старое выбрасывается,
        чтобы комментарий не отставал от экрана.
        """
        current = self._clock() if now is None else float(now)
        with self._lock:
            if not self.available or not self._active or self._paused:
                return False
            if event.kind not in COMMENT_KINDS:
                return False
            if self._busy_locked():
                return False
            if not self._can_attempt_locked(current):
                return False
            self._drain_locked()
            try:
                self._queue.put_nowait((event, frame))
            except queue.Full:
                return False
            return True

    def status(self, *, now: float | None = None) -> CommentStatus:
        with self._lock:
            current = self._clock() if now is None else float(now)
            self._prune_locked(current)
            return CommentStatus(
                available=self.available,
                active=self._active,
                paused=self._paused,
                use_frame=self.use_frame,
                min_interval_sec=self.min_interval_sec,
                max_per_hour=self.max_per_hour,
                attempts=len(self._attempts),
                comments=self._comments,
                last_attempt_at=self._last_attempt_at,
                last_comment_at=self._last_comment_at,
                next_allowed_at=self._next_allowed_locked(current),
                last_error=self._last_error,
            )

    def describe(self, *, now: float | None = None) -> str:
        """Короткий отчёт для голоса и статуса."""
        status = self.status(now=now)
        current = self._clock() if now is None else float(now)
        if not status.available:
            return "Комментарии выключены в настройках зрения."
        if not status.active:
            return "Комментарии выключены."
        if status.paused:
            return "Комментарии на паузе."
        parts = [
            f"Комментарии включены: {status.comments} "
            f"{plural(status.comments, 'комментарий', 'комментария', 'комментариев')}, "
            f"{status.attempts} из {status.max_per_hour} проверок за час."
        ]
        if status.last_comment_at is not None:
            age = max(0.0, current - status.last_comment_at)
            parts.append(f"Последний комментарий {format_duration(age)} назад.")
        if status.next_allowed_at is not None and status.next_allowed_at > current:
            wait = status.next_allowed_at - current
            parts.append(f"Следующий не раньше чем через {format_duration(wait)}.")
        if status.use_frame:
            parts.append("В облако уходит текст события и не больше одного кадра на комментарий.")
        else:
            parts.append("В облако уходит только текст события, без кадров.")
        return " ".join(parts)

    # -------------------------------------------------------------- internals
    def _run(self, generation: int, stop: threading.Event) -> None:
        while not stop.wait(0.2):
            try:
                event, frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._lock:
                if generation != self._generation:
                    return
                if not self._active or self._paused:
                    continue
                if self._busy_locked():
                    continue
                now = self._clock()
                if not self._can_attempt_locked(now):
                    continue
                self._attempts.append(now)
                self._last_attempt_at = now
            comment: str | None = None
            try:
                comment = self._compose(event, frame if self.use_frame else None)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._last_error = exc.__class__.__name__
                log.warning("Комментарии: подготовка не удалась: %s", exc.__class__.__name__)
                continue
            text = clean_comment(comment)
            if text is None:
                with self._lock:
                    self._last_error = None
                continue
            with self._lock:
                if generation != self._generation:
                    return
                if not self._active or self._paused or self._busy_locked():
                    continue
                self._comments += 1
                self._last_comment_at = self._clock()
                self._last_error = None
            try:
                self._on_comment(text)
            except Exception:  # noqa: BLE001
                log.exception("Комментарии: обработчик озвучки упал")

    def _drain_locked(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _busy_locked(self) -> bool:
        try:
            return bool(self._is_busy())
        except Exception:  # noqa: BLE001
            log.debug("Комментарии: проверка занятости упала", exc_info=True)
            return True

    def _prune_locked(self, now: float) -> None:
        while self._attempts and now - self._attempts[0] >= HOUR_SEC:
            self._attempts.popleft()

    def _can_attempt_locked(self, now: float) -> bool:
        if self.max_per_hour <= 0:
            return False
        self._prune_locked(now)
        if len(self._attempts) >= self.max_per_hour:
            return False
        if (
            self._last_attempt_at is not None
            and now - self._last_attempt_at < self.min_interval_sec
        ):
            return False
        return True

    def _next_allowed_locked(self, now: float) -> float | None:
        if not self._active or self.max_per_hour <= 0:
            return None
        self._prune_locked(now)
        next_at = 0.0
        if self._last_attempt_at is not None:
            next_at = max(next_at, self._last_attempt_at + self.min_interval_sec)
        if len(self._attempts) >= self.max_per_hour and self._attempts:
            next_at = max(next_at, self._attempts[0] + HOUR_SEC)
        return next_at or None


def clean_comment(text: str | None) -> str | None:
    """Приводит ответ модели к одной короткой реплике или ``None``."""
    if not text:
        return None
    cleaned = " ".join(str(text).split()).strip()
    if not cleaned:
        return None
    if cleaned.upper().startswith("SKIP"):
        return None
    for prefix in ("Комментарий:", "Комментарий —", "Ответ:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
            break
    cleaned = cleaned.strip(" \"'«»")
    if not cleaned:
        return None
    if len(cleaned) > MAX_COMMENT_CHARS:
        cleaned = cleaned[: MAX_COMMENT_CHARS - 1].rstrip() + "…"
    return cleaned


def describe_event(event: VisionEvent) -> str:
    """Текст события для промпта, без содержимого экрана."""
    if event.kind == KIND_WINDOW:
        current = event.window_title or "без названия"
        if event.previous_title:
            return f"активное окно сменилось с «{event.previous_title}» на «{current}»"
        return f"активное окно: «{current}»"
    if event.kind == KIND_IDLE:
        return f"экран не менялся {event.detail or 'долго'}"
    if event.kind == KIND_CHANGE:
        if event.count > 1:
            return f"экран изменился ({event.count} раз)"
        return "экран изменился"
    if event.kind == KIND_START:
        return "наблюдение началось"
    return "событие наблюдения"


def build_text_prompt(event: VisionEvent, context: str = "") -> str:
    """Промпт для комментария без картинки: только событие и его история."""
    context = " ".join((context or "").split())
    prompt = (
        "Ты — голосовой ассистент Луна. Ниже краткая история событий на экране. "
        "Если уместен один короткий комментарий или полезная подсказка — ответь "
        "одним предложением по-русски, не длиннее 15 слов. Если комментировать "
        "нечего, это отвлечёт или ты не уверена — ответь ровно SKIP. "
        "Не выдумывай содержимое экрана и не предлагай опасных действий. "
        f"Текущее событие: {describe_event(event)}."
    )
    if context:
        prompt += f" История: {context}"
    return prompt


__all__ = [
    "COMMENT_KINDS",
    "CommentStatus",
    "DEFAULT_MAX_PER_HOUR",
    "DEFAULT_MIN_INTERVAL_SEC",
    "ProactiveCommenter",
    "build_text_prompt",
    "clean_comment",
    "describe_event",
]
