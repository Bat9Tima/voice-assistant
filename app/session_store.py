"""Сохранение завершённых сессий диалога в SQLite.

Одна сессия пишется одной транзакцией: либо сессия и все её реплики, либо
ничего. Соединение открывается на каждую операцию и сразу закрывается,
поэтому голосовой цикл и будущие читатели не делят одно подключение
между потоками.

Модуль не решает, когда сохранять: это делает ``SessionMemory.on_finish``.
Ошибки записи пробрасываются наверх — их ловит и логирует ``SessionMemory``,
чтобы сбой диска не ломал разговор.

Кроме сырых сессий и реплик есть таблица ``summaries``: одно резюме на
сессию. Пока это только хранилище — генерация и подмешивание в диалог
будут отдельным этапом.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .session import DialogSession, SessionMemory


def _now() -> datetime:
    """Локальное время с часовым поясом, как в сессиях."""
    return datetime.now().astimezone()


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    messages_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, position);

CREATE INDEX IF NOT EXISTS idx_sessions_started
    ON sessions(started_at);

CREATE TABLE IF NOT EXISTS summaries (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version INTEGER NOT NULL DEFAULT 1
);
"""


class SessionStore:
    """SQLite-хранилище разговоров. Путь создаётся при первой записи."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def attach(self, sessions: SessionMemory) -> SessionStore:
        """Вешает сохранение на память сессий и возвращает себя."""
        sessions.on_finish = self.save
        return self

    # ------------------------------------------------------------------ write
    def save(self, session: DialogSession) -> None:
        """Пишет сессию и все реплики. Повторное сохранение не плодит дубли."""
        if not session.messages:
            return
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, started_at, ended_at, messages_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    ended_at = excluded.ended_at,
                    messages_count = excluded.messages_count
                """,
                (
                    session.id,
                    session.started_at.isoformat(),
                    session.ended_at.isoformat() if session.ended_at is not None else None,
                    len(session.messages),
                ),
            )
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session.id,))
            conn.executemany(
                """
                INSERT INTO messages (session_id, position, role, content, at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        session.id,
                        position,
                        message.role,
                        message.content,
                        message.at.isoformat(),
                    )
                    for position, message in enumerate(session.messages)
                ],
            )

    def delete(self, session_id: str) -> bool:
        """Удаляет сессию вместе с репликами и резюме. ``True``, если запись была."""
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    def save_summary(
        self,
        session_id: str,
        summary: str,
        *,
        model: str = "",
        prompt_version: int = 1,
        created_at: datetime | None = None,
    ) -> None:
        """Сохраняет резюме сессии. Повторная запись заменяет прошлое."""
        text = (summary or "").strip()
        if not text:
            raise ValueError("пустое резюме")
        moment = created_at or _now()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO summaries (session_id, summary, created_at, model, prompt_version)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    created_at = excluded.created_at,
                    model = excluded.model,
                    prompt_version = excluded.prompt_version
                """,
                (
                    session_id,
                    text,
                    moment.isoformat(),
                    str(model or ""),
                    max(1, int(prompt_version)),
                ),
            )

    # ------------------------------------------------------------------- read
    def sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Последние сессии, новые сверху."""
        limit = max(1, int(limit))
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, started_at, ended_at, messages_count
                FROM sessions
                ORDER BY started_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def messages(self, session_id: str) -> list[dict[str, str]]:
        """Реплики сессии в порядке разговора."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT role, content, at
                FROM messages
                WHERE session_id = ?
                ORDER BY position
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def summary(self, session_id: str) -> dict[str, Any] | None:
        """Резюме одной сессии или ``None``."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT session_id, summary, created_at, model, prompt_version
                FROM summaries
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def summaries(self, limit: int = 20) -> list[dict[str, Any]]:
        """Последние резюме, новые сверху."""
        limit = max(1, int(limit))
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT session_id, summary, created_at, model, prompt_version
                FROM summaries
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        """Сколько сессий лежит в базе."""
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM sessions").fetchone()
        return int(row["total"])

    # --------------------------------------------------------------- private
    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(SCHEMA)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
