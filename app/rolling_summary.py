"""Скользящее резюме текущего разговора.

Окно истории конечно: когда реплики выпадают из него, модель перестаёт
их видеть. Чтобы разговор не рвался, выпавшие реплики сжимаются в
короткое содержание и подмешиваются в системный промпт следующим ходом.
Резюме обновляется в фоне: ответ пользователю его не ждёт.

Живёт резюме только в памяти процесса — это часть текущего разговора, а
не архив. Итоговое резюме сессии считается отдельно после выхода. Сбой
модели ничего не ломает: остаётся прошлое содержание, разговор идёт
дальше.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any

from .prompts import ROLLING_SYSTEM
from .session import DialogSession, Message

log = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 1200
DEFAULT_MAX_INPUT_CHARS = 4000
DEFAULT_MAX_TOKENS = 250
DEFAULT_TIMEOUT_SEC = 30.0
DEFAULT_MIN_MESSAGES = 4
DEFAULT_TEMPERATURE = 0.2
ROLLING_HEADER = "Краткое содержание начала этого разговора:"
ROLE_NAMES = {"user": "Пользователь", "assistant": "Ассистент"}


def _int(value: Any, default: int) -> int:
    """Целое из конфига: ``None`` — значение по умолчанию, ноль разрешён."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Batch:
    """Порция выпавших реплик и поколение разговора."""

    generation: int
    messages: tuple[Message, ...]


class RollingSummary:
    """Держит краткое содержание начала текущего разговора."""

    def __init__(
        self,
        llm: Any,
        *,
        enabled: bool = True,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        min_messages: int = DEFAULT_MIN_MESSAGES,
    ) -> None:
        self.llm = llm
        self.enabled = bool(enabled)
        self.max_chars = max(0, int(max_chars))
        self.max_input_chars = max(0, int(max_input_chars))
        self.max_tokens = max(1, int(max_tokens))
        self.temperature = float(temperature)
        self.timeout = max(1.0, float(timeout))
        self.min_messages = max(1, int(min_messages))
        self._text = ""
        # Сколько реплик уже отдано в резюме: всё до этого индекса
        # считается покрытым, даже если воркер ещё не ответил.
        self._covered = 0
        self._generation = 0
        self._lock = threading.Lock()
        self._queue: queue.Queue[Batch | None] = queue.Queue()
        self._worker: threading.Thread | None = None

    @classmethod
    def from_config(cls, llm: Any, section: dict[str, Any] | None = None) -> RollingSummary:
        """Собирает резюме из секции ``dialog.context.rolling_summary``."""
        section = section or {}
        return cls(
            llm,
            enabled=bool(section.get("enabled", True)),
            max_chars=_int(section.get("max_chars"), DEFAULT_MAX_CHARS),
            max_input_chars=_int(section.get("max_input_chars"), DEFAULT_MAX_INPUT_CHARS),
            max_tokens=_int(section.get("max_tokens"), DEFAULT_MAX_TOKENS),
            temperature=float(section.get("temperature", DEFAULT_TEMPERATURE)),
            timeout=float(section.get("timeout_sec") or DEFAULT_TIMEOUT_SEC),
            min_messages=_int(section.get("min_messages"), DEFAULT_MIN_MESSAGES),
        )

    # ------------------------------------------------------------------ public
    def reset(self) -> None:
        """Забывает содержание: начался или закончился разговор."""
        with self._lock:
            self._text = ""
            self._covered = 0
            # Ответы старого воркера после сброса уже не нужны.
            self._generation += 1

    def block(self) -> str:
        """Блок для системного промпта или пустая строка."""
        with self._lock:
            text = self._text
        return f"{ROLLING_HEADER}\n{text}" if text else ""

    def observe(self, session: DialogSession | None, window_start: int) -> None:
        """Отдаёт в резюме реплики, выпавшие из окна истории."""
        if not self.enabled or session is None:
            return
        with self._lock:
            start = min(max(0, int(window_start)), len(session.messages))
            if start <= self._covered:
                return
            messages = tuple(session.messages[self._covered : start])
            # Мелкие порции копим: на каждый ход отдельный запрос не нужен.
            if len(messages) < self.min_messages:
                return
            self._covered = start
            batch = Batch(self._generation, messages)
        self._queue.put(batch)
        self._ensure_worker()

    def close(self, timeout: float | None = None) -> None:
        """Даёт очереди доработать перед выходом программы."""
        worker = self._worker
        if worker is None:
            return
        limit = self.timeout if timeout is None else float(timeout)
        self._queue.put(None)
        worker.join(limit)
        if worker.is_alive():
            log.warning("Скользящее резюме не успело за %.0f с", limit)
        else:
            with self._lock:
                if self._worker is worker:
                    self._worker = None

    # ---------------------------------------------------------------- private
    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._work,
                name="dialog-rolling-summary",
                daemon=True,
            )
            self._worker.start()

    def _work(self) -> None:
        while True:
            batch = self._queue.get()
            try:
                if batch is None:
                    return
                self._process(batch)
            except Exception:  # noqa: BLE001
                log.exception("Не удалось обновить скользящее резюме разговора")
            finally:
                self._queue.task_done()

    def _process(self, batch: Batch) -> None:
        with self._lock:
            if batch.generation != self._generation:
                return
            base = self._text
        answer = self.llm.chat(
            [{"role": "user", "content": self._prompt(base, batch.messages)}],
            system=ROLLING_SYSTEM,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        summary = " ".join((answer or "").split())
        if not summary:
            raise ValueError("модель вернула пустое содержание")
        summary = self._clip(summary)
        with self._lock:
            if batch.generation != self._generation:
                return
            self._text = summary
        log.info("Скользящее резюме обновлено: %s символов", len(summary))

    def _prompt(self, base: str, messages: tuple[Message, ...]) -> str:
        transcript = self._transcript(messages)
        if base:
            return (
                f"Прошлое содержание:\n{base}\n\nНовые реплики:\n{transcript}\n\nОбнови содержание."
            )
        return f"Новые реплики:\n{transcript}\n\nСоставь содержание."

    def _transcript(self, messages: tuple[Message, ...]) -> str:
        lines = [
            f"{ROLE_NAMES.get(message.role, message.role)}: {message.content}"
            for message in messages
        ]
        text = "\n".join(lines)
        if self.max_input_chars and len(text) > self.max_input_chars:
            tail = text[-self.max_input_chars :]
            boundary = tail.find("\n")
            if boundary != -1:
                tail = tail[boundary + 1 :]
            text = "…\n" + tail
        return text

    def _clip(self, text: str) -> str:
        """Режет содержание до ``max_chars``, по возможности по концу фразы."""
        if not self.max_chars or len(text) <= self.max_chars:
            return text
        cut = text[: self.max_chars]
        boundary = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        if boundary >= self.max_chars // 2:
            return cut[: boundary + 1]
        return cut[: self.max_chars - 1].rstrip() + "…"
