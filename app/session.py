"""История текущего диалога в памяти процесса.

Модуль намеренно не знает про сеть, файлы и базу данных. Его задача —
держать сообщения одного разговора и отдавать модели урезанное окно.
Жизненный цикл отдаётся колбэками: ``on_start`` (сессия началась),
``on_message`` (появилась реплика) и ``on_finish`` + ``hooks`` (сессия
закрылась). В приложении к ним подключены ``SessionStore`` (запись в
SQLite по ходу разговора) и суммаризатор (резюме после выхода).
Сам модуль о SQLite и модели ничего не знает.

Границы: ``start`` открывает сессию, ``finish`` закрывает её и вызывает
колбэки. Сбой одного колбэка только пишется в лог: остальные колбэки
и голосовой цикл он не ломает.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_HISTORY_MESSAGES = 12
DEFAULT_HISTORY_CHARS = 4000
DEFAULT_KEEP_FINISHED = 10
ROLES = ("user", "assistant")

SessionHook = Callable[["DialogSession"], None]
MessageHook = Callable[["DialogSession", "Message", int], None]


def _now() -> datetime:
    """Локальное время с часовым поясом: так архив удобнее читать."""
    return datetime.now().astimezone()


@dataclass(frozen=True)
class Message:
    """Одна реплика разговора."""

    role: str
    content: str
    at: datetime

    def as_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
            "at": self.at.isoformat(),
        }


@dataclass
class DialogSession:
    """Один разговор: от «давай поговорим» до выхода."""

    started_at: datetime
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ended_at: datetime | None = None
    messages: list[Message] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.ended_at is not None

    def add(
        self,
        role: str,
        content: str,
        *,
        at: datetime | None = None,
    ) -> Message | None:
        """Добавляет реплику. Пустой текст и чужая роль отбрасываются."""
        role = (role or "").strip()
        content = " ".join((content or "").split())
        if role not in ROLES or not content:
            return None
        message = Message(role=role, content=content, at=at or _now())
        self.messages.append(message)
        return message

    def finish(self, *, at: datetime | None = None) -> None:
        """Ставит время окончания один раз: повторный вызов ничего не меняет."""
        if self.ended_at is None:
            self.ended_at = at or _now()

    def as_dict(self) -> dict[str, Any]:
        """Плоское представление для будущего сохранения."""
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at is not None else None,
            "messages": [message.as_dict() for message in self.messages],
        }


class SessionMemory:
    """Текущая сессия и короткий список завершённых.

    Полная история хранится целиком, но в модель уходит только окно:
    последние ``max_messages`` сообщений и не больше ``max_chars`` символов.
    Ноль отключает соответствующий лимит.
    """

    def __init__(
        self,
        max_messages: int = DEFAULT_HISTORY_MESSAGES,
        max_chars: int = DEFAULT_HISTORY_CHARS,
        *,
        keep_finished: int = DEFAULT_KEEP_FINISHED,
        on_finish: SessionHook | None = None,
        hooks: Iterable[SessionHook] | None = None,
        on_start: SessionHook | None = None,
        on_message: MessageHook | None = None,
    ) -> None:
        self.max_messages = max(0, int(max_messages))
        self.max_chars = max(0, int(max_chars))
        self.keep_finished = max(0, int(keep_finished))
        self.on_finish = on_finish
        self.hooks: list[SessionHook] = list(hooks or ())
        # Колбэки записи: сессия заводится в начале разговора, каждая
        # реплика дописывается сразу после появления. Так база не теряет
        # разговор, если программу закрыли посреди беседы.
        self.on_start = on_start
        self.on_message = on_message
        self._current: DialogSession | None = None
        self._finished: list[DialogSession] = []

    def add_hook(self, hook: SessionHook) -> SessionMemory:
        """Добавляет обработчик завершённых сессий и возвращает себя.

        Основной ``on_finish`` вызывается первым, затем хуки по порядку:
        так сырая сессия успевает лечь в базу до того, как её начнут
        сжимать в резюме.
        """
        self.hooks.append(hook)
        return self

    @property
    def current(self) -> DialogSession | None:
        return self._current

    @property
    def finished(self) -> list[DialogSession]:
        """Копия: вызывающий не должен править внутренний список."""
        return list(self._finished)

    # ------------------------------------------------------------- lifecycle
    def start(self) -> DialogSession:
        """Открывает новую сессию. Незакрытая предыдущая сохраняется."""
        if self._current is not None:
            self.finish()
        self._current = DialogSession(started_at=_now())
        self._notify_start(self._current)
        return self._current

    def ensure(self) -> DialogSession:
        """Текущая сессия или новая, если разговор ещё не начат."""
        return self._current or self.start()

    def finish(self) -> DialogSession | None:
        """Закрывает текущую сессию и отдаёт её в ``on_finish``.

        Пустая сессия не попадает в список завершённых, но колбэк получает
        её тоже: хранилище по нему удаляет пустую запись из базы.
        """
        session = self._current
        if session is None:
            return None
        self._current = None
        session.finish()
        if session.messages:
            self._remember(session)
        self._notify(session)
        return session

    def clear(self) -> None:
        """Забывает всё, не вызывая ``on_finish``."""
        self._current = None
        self._finished.clear()

    # -------------------------------------------------------------- messages
    def add(
        self,
        role: str,
        content: str,
        *,
        at: datetime | None = None,
    ) -> Message | None:
        """Добавляет реплику в текущую сессию, открывая её при необходимости.

        После добавления реплика сразу отдаётся в ``on_message``: так она
        попадает в базу, не дожидаясь конца разговора.
        """
        session = self.ensure()
        message = session.add(role, content, at=at)
        if message is None:
            return None
        self._notify_message(session, message, len(session.messages) - 1)
        return message

    def history(
        self,
        *,
        max_messages: int | None = None,
        max_chars: int | None = None,
    ) -> list[dict[str, str]]:
        """Окно истории в формате OpenAI: старые реплики отбрасываются."""
        session = self._current
        if session is None or not session.messages:
            return []
        start = self.window_start(max_messages=max_messages, max_chars=max_chars)
        return [{"role": item.role, "content": item.content} for item in session.messages[start:]]

    def window_start(
        self,
        *,
        max_messages: int | None = None,
        max_chars: int | None = None,
    ) -> int:
        """Индекс первой реплики, которая ещё попадает в окно истории.

        Всё до него в модель не уходит: именно эти реплики сжимает
        скользящее резюме текущего разговора.
        """
        session = self._current
        if session is None or not session.messages:
            return 0
        messages = session.messages
        limit = self.max_messages if max_messages is None else max(0, int(max_messages))
        chars = self.max_chars if max_chars is None else max(0, int(max_chars))
        start = max(0, len(messages) - limit) if limit else 0
        if chars:
            total = 0
            index = len(messages)
            while index > start:
                size = len(messages[index - 1].content)
                # Самая свежая реплика попадает в окно всегда.
                if index < len(messages) and total + size > chars:
                    break
                total += size
                index -= 1
            start = index
        return start

    # ---------------------------------------------------------------- private
    def _remember(self, session: DialogSession) -> None:
        self._finished.append(session)
        if self.keep_finished:
            self._finished = self._finished[-self.keep_finished :]
        else:
            self._finished.clear()

    def _notify(self, session: DialogSession) -> None:
        callbacks = ([self.on_finish] if self.on_finish is not None else []) + list(self.hooks)
        for callback in callbacks:
            try:
                callback(session)
            except Exception:  # noqa: BLE001
                log.exception("Обработчик завершённой сессии %s упал", session.id)

    def _notify_start(self, session: DialogSession) -> None:
        if self.on_start is None:
            return
        try:
            self.on_start(session)
        except Exception:  # noqa: BLE001
            log.exception("Не удалось открыть сессию %s", session.id)

    def _notify_message(self, session: DialogSession, message: Message, position: int) -> None:
        if self.on_message is None:
            return
        try:
            self.on_message(session, message, position)
        except Exception:  # noqa: BLE001
            log.exception(
                "Не удалось записать реплику %s сессии %s",
                position,
                session.id,
            )
