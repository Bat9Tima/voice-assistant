"""Сохранение разговоров диалога в SQLite.

Запись идёт по ходу разговора, а не одной пачкой в конце:

* ``create`` заводит сессию один раз, в момент начала разговора
  (``sessions.status = 'active'``);
* ``append`` дописывает каждую реплику сразу после её появления;
* ``finish`` ставит ``status = 'completed'`` и время окончания.

Так разговор переживает аварийное закрытие программы: в базе остаётся
всё, что успели записать. Перечитать его можно через ``sessions`` и
``messages``; незавершённые сессии видны в ``active``.

``save`` остаётся полной синхронизацией сессии: им чинят разговор,
если отдельные реплики не дошли до диска. Соединение открывается на
каждую операцию и сразу закрывается, поэтому голосовой цикл и фоновое
резюме не делят одно подключение между потоками.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .session import DialogSession, Message, SessionMemory

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    messages_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'completed'
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_messages_position
    ON messages(session_id, position);

CREATE INDEX IF NOT EXISTS idx_sessions_started
    ON sessions(started_at);

CREATE TABLE IF NOT EXISTS summaries (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version INTEGER NOT NULL DEFAULT 1,
    data TEXT NOT NULL DEFAULT '{}'
);
"""


class SessionStore:
    """SQLite-хранилище разговоров. Путь создаётся при первой записи."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._ready = False
        self._migrate_lock = threading.Lock()

    def attach(self, sessions: SessionMemory) -> SessionStore:
        """Вешает запись на память сессий и возвращает себя.

        Сессия заводится на старте, реплики пишутся по одной, финал
        закрывает сессию. Резюме подключается отдельным хуком.
        """
        sessions.on_start = self.create
        sessions.on_message = self.append
        sessions.on_finish = self.finish
        return self

    # ------------------------------------------------------------------ write
    def create(self, session: DialogSession) -> None:
        """Заводит сессию в начале разговора. Повторный вызов безопасен."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, started_at, ended_at, messages_count, status)
                VALUES (?, ?, NULL, 0, 'active')
                ON CONFLICT(id) DO NOTHING
                """,
                (session.id, session.started_at.isoformat()),
            )

    def append(self, session: DialogSession, message: Message, position: int) -> None:
        """Дописывает одну реплику. Повтор с той же позицией не плодит дубль."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO messages (session_id, position, role, content, at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    position,
                    message.role,
                    message.content,
                    message.at.isoformat(),
                ),
            )
            self._sync_count(conn, session.id)

    def finish(self, session: DialogSession) -> None:
        """Закрывает сессию и ставит время окончания.

        Метод самодостаточен: если у сессии не проставлен ``ended_at``,
        берётся текущее время. Пустая сессия удаляется — восстанавливать
        из неё нечего. Если в базе не хватает реплик (сбой при
        дописывании), сессия целиком перезаписывается из памяти.
        """
        with self._connection() as conn:
            if not session.messages:
                conn.execute("DELETE FROM sessions WHERE id = ?", (session.id,))
                return
            stored = conn.execute(
                "SELECT COUNT(*) AS total FROM messages WHERE session_id = ?",
                (session.id,),
            ).fetchone()["total"]
            if int(stored) != len(session.messages):
                self._write_all(conn, session, closed=True)
            else:
                self._upsert_session(conn, session, closed=True)

    def save(self, session: DialogSession) -> None:
        """Полная запись сессии одной транзакцией: чинит недостающие реплики."""
        if not session.messages:
            return
        with self._connection() as conn:
            self._write_all(conn, session)

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
        data: dict[str, Any] | None = None,
    ) -> None:
        """Сохраняет резюме сессии. Повторная запись заменяет прошлое.

        ``summary`` — обычный текст для поиска и чтения, ``data`` —
        структурированные поля промпта v2. Старые записи получают ``{}``.
        """
        text = (summary or "").strip()
        if not text:
            raise ValueError("пустое резюме")
        moment = created_at or _now()
        payload = json.dumps(data or {}, ensure_ascii=False, sort_keys=True)
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO summaries
                    (session_id, summary, created_at, model, prompt_version, data)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    created_at = excluded.created_at,
                    model = excluded.model,
                    prompt_version = excluded.prompt_version,
                    data = excluded.data
                """,
                (
                    session_id,
                    text,
                    moment.isoformat(),
                    str(model or ""),
                    max(1, int(prompt_version)),
                    payload,
                ),
            )

    # ------------------------------------------------------------------- read
    def sessions(self, limit: int = 20, *, status: str = "") -> list[dict[str, Any]]:
        """Последние сессии, новые сверху. ``status`` фильтрует active/completed."""
        limit = max(1, int(limit))
        query = """
            SELECT id, started_at, ended_at, messages_count, status
            FROM sessions
        """
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY started_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def active(self, limit: int = 20) -> list[dict[str, Any]]:
        """Незавершённые сессии: разговор идёт сейчас или оборвался сбоем."""
        return self.sessions(limit, status="active")

    def recover(self) -> list[DialogSession]:
        """Закрывает сессии, оборванные сбоем, и отдаёт их для резюме.

        Сессия, оставшаяся ``active`` после падения прошлого запуска, сама
        уже не закроется. Временем окончания считаем последнюю реплику:
        именно тогда разговор фактически прервался. Пустые сессии
        удаляются, как и при обычном ``finish``.
        """
        recovered: list[DialogSession] = []
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, started_at FROM sessions
                WHERE status = 'active'
                ORDER BY started_at
                """
            ).fetchall()
            for row in rows:
                messages = conn.execute(
                    """
                    SELECT role, content, at FROM messages
                    WHERE session_id = ?
                    ORDER BY position
                    """,
                    (row["id"],),
                ).fetchall()
                if not messages:
                    conn.execute("DELETE FROM sessions WHERE id = ?", (row["id"],))
                    continue
                ended_at = _parse_time(messages[-1]["at"])
                conn.execute(
                    """
                    UPDATE sessions SET status = 'completed', ended_at = ?
                    WHERE id = ?
                    """,
                    (ended_at.isoformat(), row["id"]),
                )
                session = DialogSession(
                    id=row["id"],
                    started_at=_parse_time(row["started_at"]),
                    ended_at=ended_at,
                )
                session.messages = [
                    Message(
                        role=item["role"],
                        content=item["content"],
                        at=_parse_time(item["at"]),
                    )
                    for item in messages
                ]
                recovered.append(session)
        if recovered:
            log.info("SQLite: закрыто оборванных сессий: %s", len(recovered))
        return recovered

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
                SELECT session_id, summary, created_at, model, prompt_version, data
                FROM summaries
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return _summary_row(row) if row is not None else None

    def summaries(self, limit: int | None = 20) -> list[dict[str, Any]]:
        """Последние резюме, новые сверху. ``None`` — все резюме."""
        query = """
            SELECT session_id, summary, created_at, model, prompt_version, data
            FROM summaries
            ORDER BY created_at DESC, rowid DESC
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (max(1, int(limit)),)
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_summary_row(row) for row in rows]

    def summaries_with_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Резюме вместе с временем и размером сессии: для дневного дайджеста."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    m.session_id, m.summary, m.created_at, m.model,
                    m.prompt_version, m.data,
                    s.started_at, s.ended_at, s.messages_count, s.status
                FROM summaries m
                JOIN sessions s ON s.id = m.session_id
                ORDER BY s.started_at DESC, m.rowid DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [_summary_row(row) for row in rows]

    def count(self) -> int:
        """Сколько сессий лежит в базе."""
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM sessions").fetchone()
        return int(row["total"])

    # --------------------------------------------------------------- private
    def _write_all(
        self,
        conn: sqlite3.Connection,
        session: DialogSession,
        *,
        closed: bool = False,
    ) -> None:
        """Перезаписывает сессию и все её реплики."""
        self._upsert_session(conn, session, closed=closed)
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
        self._sync_count(conn, session.id)

    @staticmethod
    def _upsert_session(
        conn: sqlite3.Connection,
        session: DialogSession,
        *,
        closed: bool = False,
    ) -> None:
        ended_at = session.ended_at
        if ended_at is None and closed:
            ended_at = _now()
        conn.execute(
            """
            INSERT INTO sessions (id, started_at, ended_at, messages_count, status)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                started_at = excluded.started_at,
                ended_at = excluded.ended_at,
                messages_count = excluded.messages_count,
                status = excluded.status
            """,
            (
                session.id,
                session.started_at.isoformat(),
                ended_at.isoformat() if ended_at is not None else None,
                len(session.messages),
                "completed" if closed or session.finished else "active",
            ),
        )

    @staticmethod
    def _sync_count(conn: sqlite3.Connection, session_id: str) -> None:
        conn.execute(
            """
            UPDATE sessions
            SET messages_count = (
                SELECT COUNT(*) FROM messages WHERE session_id = sessions.id
            )
            WHERE id = ?
            """,
            (session_id,),
        )

    def _migrate_once(self, conn: sqlite3.Connection) -> None:
        """Миграция схемы один раз на процесс: на каждой записи она лишняя."""
        if self._ready:
            return
        with self._migrate_lock:
            if self._ready:
                return
            _migrate(conn)
            self._ready = True

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(SCHEMA)
            self._migrate_once(conn)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Доводит старую базу до текущей схемы: колонка status и уникальность."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
    if "status" not in columns:
        # Сессии, записанные до этого изменения, все завершены.
        conn.execute("ALTER TABLE sessions ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'")
    summary_columns = {row["name"] for row in conn.execute("PRAGMA table_info(summaries)")}
    if summary_columns and "data" not in summary_columns:
        # Резюме до промпта v2 хранят только текст; структуры у них нет.
        conn.execute("ALTER TABLE summaries ADD COLUMN data TEXT NOT NULL DEFAULT '{}'")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_messages_position
            ON messages(session_id, position)
        """
    )
    conn.execute("DROP INDEX IF EXISTS idx_messages_session")


def _summary_row(row: sqlite3.Row) -> dict[str, Any]:
    """Строка резюме с разобранным JSON в ``data``. Битый JSON — пустой словарь."""
    result = dict(row)
    result["data"] = _parse_data(result.get("data"))
    return result


def _parse_data(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _now() -> datetime:
    """Локальное время с часовым поясом, как в сессиях."""
    return datetime.now().astimezone()


def _parse_time(value: str) -> datetime:
    """Время из базы. Старые записи могли лечь без пояса — считаем локальными."""
    moment = datetime.fromisoformat(str(value))
    if moment.tzinfo is None:
        return moment.astimezone()
    return moment
