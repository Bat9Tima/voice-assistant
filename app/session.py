"""История текущего диалога в памяти процесса.

Модуль намеренно не знает про сеть, файлы и базу данных. Его задача —
держать сообщения одного разговора и отдавать модели урезанное окно.
Завершённая сессия уходит в колбэк ``on_finish``: на следующем этапе туда
встанет запись в SQLite, а диалог трогать уже не придётся.

Границы: ``start`` открывает сессию, ``finish`` закрывает её и вызывает
колбэк. Сбой колбэка только пишется в лог, голосовой цикл он не ломает.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_HISTORY_MESSAGES = 12
DEFAULT_HISTORY_CHARS = 4000
DEFAULT_KEEP_FINISHED = 10
ROLES = ("user", "assistant")

SessionHook = Callable[["DialogSession"], None]


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
    ) -> None:
        self.max_messages = max(0, int(max_messages))
        self.max_chars = max(0, int(max_chars))
        self.keep_finished = max(0, int(keep_finished))
        self.on_finish = on_finish
        self._current: DialogSession | None = None
        self._finished: list[DialogSession] = []

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
        return self._current

    def ensure(self) -> DialogSession:
        """Текущая сессия или новая, если разговор ещё не начат."""
        return self._current or self.start()

    def finish(self) -> DialogSession | None:
        """Закрывает текущую сессию и отдаёт её в ``on_finish``.

        Пустая сессия не сохраняется: разговор без реплик никому не нужен.
        """
        session = self._current
        if session is None:
            return None
        self._current = None
        session.finish()
        if not session.messages:
            return session
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
        """Добавляет реплику в текущую сессию, открывая её при необходимости."""
        return self.ensure().add(role, content, at=at)

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
        messages = session.messages
        limit = self.max_messages if max_messages is None else max(0, int(max_messages))
        chars = self.max_chars if max_chars is None else max(0, int(max_chars))
        if limit:
            messages = messages[-limit:]
        if chars:
            messages = _fit_chars(messages, chars)
        return [{"role": item.role, "content": item.content} for item in messages]

    # ---------------------------------------------------------------- private
    def _remember(self, session: DialogSession) -> None:
        self._finished.append(session)
        if self.keep_finished:
            self._finished = self._finished[-self.keep_finished :]
        else:
            self._finished.clear()

    def _notify(self, session: DialogSession) -> None:
        if self.on_finish is None:
            return
        try:
            self.on_finish(session)
        except Exception:  # noqa: BLE001
            log.exception("Не удалось сохранить сессию %s", session.id)


def _fit_chars(messages: list[Message], limit: int) -> list[Message]:
    """Оставляет самые свежие реплики в пределах бюджета символов.

    Самая свежая реплика попадает в окно всегда, даже если она длиннее
    лимита: иначе окно осталось бы пустым.
    """
    selected: list[Message] = []
    total = 0
    for message in reversed(messages):
        size = len(message.content)
        if selected and total + size > limit:
            break
        selected.append(message)
        total += size
    selected.reverse()
    return selected
