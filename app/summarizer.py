"""Краткое резюме завершённого разговора.

Резюме — производные данные. Сначала ``SessionStore`` пишет сырую сессию,
потом отдельным запросом к модели делается сжатый пересказ. Если модель
недоступна или вернула мусор, сырьё остаётся на месте, а резюме можно
посчитать позже.

Пока резюме только сохраняется в таблицу ``summaries``. В новые
разговоры оно не подмешивается: это следующий этап.

Запрос к модели не должен задерживать выход из диалога, поэтому по
умолчанию резюме считается в фоновом потоке. ``close()`` дожидается
очереди при завершении программы.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

from .prompts import SUMMARY_PROMPT_VERSION, SUMMARY_SYSTEM
from .session import DialogSession, SessionMemory

log = logging.getLogger(__name__)

DEFAULT_MAX_INPUT_CHARS = 6000
DEFAULT_MAX_TOKENS = 300
DEFAULT_TIMEOUT_SEC = 45.0
DEFAULT_MIN_MESSAGES = 2
DEFAULT_TEMPERATURE = 0.2
ROLE_NAMES = {"user": "Пользователь", "assistant": "Ассистент"}


def _int(value: Any, default: int) -> int:
    """Целое из конфига: ``None`` — значение по умолчанию, ноль разрешён."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class Summarizer:
    """Сжимает завершённую сессию в короткое резюме и пишет его в базу."""

    def __init__(
        self,
        llm: Any,
        store: Any,
        *,
        enabled: bool = True,
        model: str = "",
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        min_messages: int = DEFAULT_MIN_MESSAGES,
        prompt_version: int = SUMMARY_PROMPT_VERSION,
        background: bool = True,
    ) -> None:
        self.llm = llm
        self.store = store
        self.enabled = bool(enabled)
        self.model = str(model or "").strip()
        self.max_input_chars = max(0, int(max_input_chars))
        self.max_tokens = max(1, int(max_tokens))
        self.temperature = float(temperature)
        self.timeout = max(1.0, float(timeout))
        self.min_messages = max(1, int(min_messages))
        self.prompt_version = max(1, int(prompt_version))
        self.background = bool(background)
        self._queue: queue.Queue[DialogSession | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()

    @classmethod
    def from_config(
        cls,
        llm: Any,
        store: Any,
        section: dict[str, Any] | None = None,
    ) -> Summarizer:
        """Собирает суммаризатор из вложенной секции ``dialog.summary``."""
        section = section or {}
        return cls(
            llm,
            store,
            enabled=bool(section.get("enabled", True)),
            model=str(section.get("model") or ""),
            max_input_chars=_int(section.get("max_input_chars"), DEFAULT_MAX_INPUT_CHARS),
            max_tokens=_int(section.get("max_tokens"), DEFAULT_MAX_TOKENS),
            temperature=float(section.get("temperature", DEFAULT_TEMPERATURE)),
            timeout=float(section.get("timeout_sec") or DEFAULT_TIMEOUT_SEC),
            min_messages=_int(section.get("min_messages"), DEFAULT_MIN_MESSAGES),
            prompt_version=_int(section.get("prompt_version"), SUMMARY_PROMPT_VERSION),
            background=bool(section.get("background", True)),
        )

    # ------------------------------------------------------------------ hooks
    def attach(self, sessions: SessionMemory) -> Summarizer:
        """Вешает суммаризацию на завершение сессий. Возвращает себя."""
        sessions.add_hook(self.handle)
        return self

    def handle(self, session: DialogSession) -> None:
        """Хук окончания сессии: очередь на резюме или синхронный вызов."""
        if not self.enabled:
            return
        if len(session.messages) < self.min_messages:
            log.info("Сессия %s короткая, резюме не нужно", session.id)
            return
        if self.background:
            self.enqueue(session)
            return
        self.summarize(session)

    # ---------------------------------------------------------------- summary
    def summarize(self, session: DialogSession) -> str | None:
        """Просит у модели резюме и сохраняет его. Возвращает текст."""
        transcript = self.transcript(session)
        if not transcript:
            return None
        answer = self.llm.chat(
            [{"role": "user", "content": transcript}],
            system=SUMMARY_SYSTEM,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
            model=self.model or None,
        )
        summary = (answer or "").strip()
        if not summary:
            raise ValueError("модель вернула пустое резюме")
        self.store.save_summary(
            session.id,
            summary,
            model=self.model_name(),
            prompt_version=self.prompt_version,
        )
        log.info("Резюме сессии %s сохранено: %s символов", session.id, len(summary))
        return summary

    def transcript(self, session: DialogSession) -> str:
        """Реплики одной строкой на каждую, в пределах лимита символов.

        Длинная сессия обрезается с начала: для резюме важнее, чем
        закончился разговор, а не с чего начался. Резка идёт по границе
        строки, чтобы не начинать с половины реплики.
        """
        lines = [
            f"{ROLE_NAMES.get(message.role, message.role)}: {message.content}"
            for message in session.messages
        ]
        text = "\n".join(lines)
        if self.max_input_chars and len(text) > self.max_input_chars:
            tail = text[-self.max_input_chars :]
            boundary = tail.find("\n")
            if boundary != -1:
                tail = tail[boundary + 1 :]
            text = "…\n" + tail
        return text

    def model_name(self) -> str:
        """Модель для записи в базу: своя из конфига или модель клиента."""
        return self.model or str(getattr(self.llm, "model", "") or "")

    # ------------------------------------------------------------- background
    def enqueue(self, session: DialogSession) -> None:
        """Ставит сессию в фоновую очередь и поднимает воркер."""
        self._queue.put(session)
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
            log.warning("Резюме не успело дописаться за %.0f с", limit)
        else:
            self._worker = None

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._work,
                name="dialog-summary",
                daemon=True,
            )
            self._worker.start()

    def _work(self) -> None:
        while True:
            session = self._queue.get()
            try:
                if session is None:
                    return
                self.summarize(session)
            except Exception:  # noqa: BLE001
                log.exception(
                    "Не удалось суммировать сессию %s",
                    getattr(session, "id", "?"),
                )
            finally:
                self._queue.task_done()
