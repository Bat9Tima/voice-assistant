"""Краткое резюме завершённого разговора.

Резюме — производные данные. Сначала ``SessionStore`` пишет сырую сессию,
потом отдельным запросом к модели делается сжатый пересказ. Если модель
недоступна или вернула мусор, сырьё остаётся на месте, а резюме можно
посчитать позже.

После резюме тем же фоновым воркером извлекаются факты Луны о себе и
дописываются в ``PersonaMemory``. Это отдельная память личности: она не
смешивается ни с профилем пользователя, ни с эпизодической памятью.

Запрос к модели не должен задерживать выход из диалога, поэтому по
умолчанию резюме считается в фоновом потоке. ``close()`` дожидается
очереди при завершении программы.
"""

from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime
from typing import Any

from .persona import PersonaMemory, parse_facts
from .prompts import SELF_FACTS_SYSTEM, SUMMARY_PROMPT_VERSION, SUMMARY_SYSTEM
from .session import DialogSession, SessionMemory
from .summary_schema import (
    StructuredSummary,
    SummaryFormatError,
    parse_summary,
    render_summary,
)

log = logging.getLogger(__name__)

DEFAULT_MAX_INPUT_CHARS = 6000
DEFAULT_MAX_TOKENS = 700
DEFAULT_TIMEOUT_SEC = 45.0
DEFAULT_MIN_MESSAGES = 2
DEFAULT_TEMPERATURE = 0.2
# Доля лимита, которая остаётся началу разговора. Хвост важнее, но и
# начало содержит факты о пользователе, которые иначе терялись.
HEAD_SHARE = 0.4
ROLE_NAMES = {"user": "Пользователь", "assistant": "Ассистент"}


def _int(value: Any, default: int) -> int:
    """Целое из конфига: ``None`` — значение по умолчанию, ноль разрешён."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _user_text(session: DialogSession) -> str:
    """Только реплики пользователя: по ним проверяются цитаты-подтверждения."""
    return "\n".join(message.content for message in session.messages if message.role == "user")


def _cut_head(text: str, size: int) -> str:
    """Начало транскрипта по границе строки или слова, не длиннее ``size``."""
    chunk = text[:size]
    for marker in ("\n", " "):
        position = chunk.rfind(marker)
        if position > 0:
            return chunk[:position]
    return chunk


def _cut_tail(text: str, size: int) -> str:
    """Хвост транскрипта по границе строки или слова, не длиннее ``size``."""
    chunk = text[-size:]
    for marker in ("\n", " "):
        position = chunk.find(marker)
        if position != -1:
            return chunk[position + 1 :]
    return chunk


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
        memory: Any | None = None,
        persona: PersonaMemory | None = None,
        manager: Any | None = None,
    ) -> None:
        self.llm = llm
        self.store = store
        # Необязательный индекс: если он упал, резюме всё равно сохранено.
        self.memory = memory
        # Факты Луны о себе: отдельный файл, отдельный запрос к модели.
        self.persona = persona
        # Менеджер фактов: после резюме обновляет отдельную таблицу памяти.
        self.manager = manager
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
        *,
        memory: Any | None = None,
        persona: PersonaMemory | None = None,
        manager: Any | None = None,
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
            memory=memory,
            persona=persona,
            manager=manager,
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

    # ---------------------------------------------------------------- recovery
    def recover(self) -> int:
        """Закрывает разговоры, оборванные прошлым сбоем, и суммирует их.

        Сначала база помечает такие сессии завершёнными, чтобы они не
        копились, потом каждая уходит в обычный путь резюме: очередь,
        память и факты о Луне. Уже посчитанные резюме не пересчитываются.
        """
        recover = getattr(self.store, "recover", None)
        if recover is None:
            return 0
        sessions = recover()
        for session in sessions:
            try:
                if self.store.summary(session.id) is not None:
                    continue
                self.handle(session)
            except Exception:  # noqa: BLE001
                log.exception("Оборванная сессия %s не суммирована", session.id)
        return len(sessions)

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
        raw = (answer or "").strip()
        if not raw:
            raise ValueError("модель вернула пустое резюме")
        data: dict[str, Any] = {}
        structured: StructuredSummary | None = None
        try:
            structured = parse_summary(raw, _user_text(session))
        except SummaryFormatError:
            # Модель ответила прозой вместо JSON: не теряем разговор,
            # но структуры и подтверждённых фактов у такой записи нет.
            log.warning("Резюме сессии %s не JSON, сохраняю текстом", session.id)
            summary = raw
        else:
            summary = render_summary(structured)
            data = structured.to_dict()
            if not summary:
                raise ValueError("модель вернула пустое резюме")
        self.store.save_summary(
            session.id,
            summary,
            model=self.model_name(),
            prompt_version=self.prompt_version,
            data=data,
        )
        log.info("Резюме сессии %s сохранено: %s символов", session.id, len(summary))
        if structured is not None:
            self._update_facts(session.id, structured)
        self._remember(session.id, summary, structured, session.started_at)
        self._remember_self(session, transcript)
        return summary

    def _update_facts(self, session_id: str, summary: StructuredSummary) -> None:
        """Передаёт подтверждённые пункты менеджеру памяти. Сбой не критичен."""
        if self.manager is None:
            return
        try:
            result = self.manager.ingest(session_id, summary)
        except Exception:  # noqa: BLE001
            log.exception("Факты сессии %s не обновлены", session_id)
            return
        log.info(
            "Факты сессии %s: +%s, объединено %s, обновлено %s, на проверку %s, закрыто %s",
            session_id,
            result.added,
            result.merged,
            result.superseded,
            result.review,
            result.resolved,
        )
        self._sync_fact_index()

    def _sync_fact_index(self) -> None:
        """Переносит изменения фактов в векторный индекс. Сбой не критичен."""
        if self.memory is None:
            return
        sync = getattr(self.memory, "sync_facts", None)
        if not callable(sync):
            return
        try:
            sync()
        except Exception:  # noqa: BLE001
            log.exception("Индекс фактов не обновлён")

    def _remember(
        self,
        session_id: str,
        summary: str,
        structured: StructuredSummary | None,
        started_at: datetime,
    ) -> None:
        """Индексирует темы разговора. Сбой не критичен."""
        if self.memory is None:
            return
        topics = structured.to_dict().get("topics") if structured is not None else None
        try:
            self.memory.remember(
                session_id,
                summary,
                topics=topics,
                created_at=started_at,
            )
        except Exception:  # noqa: BLE001
            log.exception("Резюме сессии %s не попало в память", session_id)

    def _remember_self(self, session: DialogSession, transcript: str) -> None:
        """Извлекает из разговора факты Луны о себе. Сбой не критичен."""
        if self.persona is None:
            return
        try:
            answer = self.llm.chat(
                [{"role": "user", "content": transcript}],
                system=SELF_FACTS_SYSTEM,
                max_tokens=self.max_tokens,
                temperature=0.0,
                timeout=self.timeout,
                model=self.model or None,
            )
            facts = parse_facts(answer or "")
            # Пустой список тоже полезен: merge заодно чистит старые повторы.
            added = self.persona.merge(facts)
        except Exception:  # noqa: BLE001
            log.exception("Факты о Луне не сохранены для сессии %s", session.id)
            return
        if added:
            log.info("Память личности: добавлено фактов %s", added)

    def transcript(self, session: DialogSession) -> str:
        """Реплики одной строкой на каждую, в пределах лимита символов.

        Длинная сессия обрезается с двух сторон: начало содержит факты,
        с которых разговор начался, хвост — чем он закончился. Середина
        при этом теряется. Резка идёт по границе строки, чтобы не
        начинать с половины реплики.
        """
        lines = [
            f"{ROLE_NAMES.get(message.role, message.role)}: {message.content}"
            for message in session.messages
        ]
        text = "\n".join(lines)
        if not self.max_input_chars or len(text) <= self.max_input_chars:
            return text
        if self.max_input_chars <= len("\n…\n"):
            return text[: self.max_input_chars]
        head_size = max(1, int(self.max_input_chars * HEAD_SHARE))
        tail_size = max(1, self.max_input_chars - head_size - len("\n…\n"))
        return f"{_cut_head(text, head_size)}\n…\n{_cut_tail(text, tail_size)}"

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
