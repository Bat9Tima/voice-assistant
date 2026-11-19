"""Состояние диалога в SQLite: сессии, резюме, факты и манера общения.

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

from .durability import PERMANENT, TRAIT
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

CREATE TABLE IF NOT EXISTS memory_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'trait',
    durability TEXT NOT NULL DEFAULT 'permanent',
    event_at TEXT NOT NULL DEFAULT '',
    fact_key TEXT NOT NULL,
    text TEXT NOT NULL,
    evidence TEXT NOT NULL,
    importance TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    replaced_by INTEGER,
    conflict_with INTEGER,
    resolved_by INTEGER
);

CREATE INDEX IF NOT EXISTS idx_memory_facts_status
    ON memory_facts(status, updated_at);

CREATE INDEX IF NOT EXISTS idx_memory_facts_session
    ON memory_facts(session_id);

CREATE TABLE IF NOT EXISTS character_versions (
    version INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    quote TEXT NOT NULL,
    style TEXT NOT NULL DEFAULT '{}',
    moment TEXT NOT NULL DEFAULT '{}',
    moment_date TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT ''
);
"""

FACT_FIELDS = (
    "kind",
    "scope",
    "durability",
    "event_at",
    "fact_key",
    "text",
    "evidence",
    "importance",
    "confidence",
    "status",
    "updated_at",
    "replaced_by",
    "conflict_with",
    "resolved_by",
)


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

    # --------------------------------------------------------------- facts
    def add_memory_fact(
        self,
        *,
        session_id: str,
        kind: str,
        scope: str = TRAIT,
        durability: str = PERMANENT,
        event_at: str = "",
        fact_key: str,
        text: str,
        evidence: str,
        importance: str,
        confidence: float,
        status: str = "active",
        created_at: str,
        updated_at: str,
        replaced_by: int | None = None,
        conflict_with: int | None = None,
        resolved_by: int | None = None,
    ) -> int:
        """Добавляет факт памяти и возвращает его id.

        Факты живут отдельно от резюме: один факт может быть обновлён,
        вытеснен или закрыт, не переписывая исходный разговор.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memory_facts
                    (session_id, kind, scope, durability, event_at, fact_key,
                     text, evidence, importance, confidence, status, created_at,
                     updated_at, replaced_by, conflict_with, resolved_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    kind,
                    scope,
                    durability,
                    event_at,
                    fact_key,
                    text,
                    evidence,
                    importance,
                    float(confidence),
                    status,
                    created_at,
                    updated_at,
                    replaced_by,
                    conflict_with,
                    resolved_by,
                ),
            )
        return int(cursor.lastrowid or 0)

    def update_memory_fact(self, fact_id: int, **fields: Any) -> bool:
        """Меняет поля факта. Неизвестные поля игнорируются.

        ``True``, если запись нашлась. Так менеджер помечает объединение,
        конфликт и закрытие задачи, не удаляя историю.
        """
        updates = {key: value for key, value in fields.items() if key in FACT_FIELDS}
        if not updates:
            return False
        columns = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values())
        values.append(int(fact_id))
        with self._connection() as conn:
            cursor = conn.execute(
                f"UPDATE memory_facts SET {columns} WHERE id = ?",
                values,
            )
        return cursor.rowcount > 0

    def delete_memory_facts(self, session_id: str) -> int:
        """Удаляет все факты сессии: пересчёт резюме не должен плодить дубли."""
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM memory_facts WHERE session_id = ?", (session_id,))
        return cursor.rowcount

    # ------------------------------------------------------------- character
    def add_character_version(
        self,
        *,
        created_at: str,
        source: str,
        quote: str,
        style: dict[str, Any],
        moment: dict[str, Any],
        moment_date: str,
        summary: str,
    ) -> int:
        """Добавляет версию манеры общения и возвращает её номер.

        Каждая явная просьба пользователя — новая версия со снимком обоих
        слоёв. История не переписывается, поэтому откат всегда возможен.
        """
        payload_style = json.dumps(style or {}, ensure_ascii=False, sort_keys=True)
        payload_moment = json.dumps(moment or {}, ensure_ascii=False, sort_keys=True)
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO character_versions
                    (created_at, source, quote, style, moment, moment_date, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    str(source or ""),
                    str(quote or ""),
                    payload_style,
                    payload_moment,
                    str(moment_date or ""),
                    str(summary or ""),
                ),
            )
        return int(cursor.lastrowid or 0)

    def latest_character(self) -> dict[str, Any] | None:
        """Последняя версия манеры или ``None``, если просьб ещё не было."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM character_versions ORDER BY version DESC LIMIT 1"
            ).fetchone()
        return _character_row(row) if row is not None else None

    def character_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Версии манеры, новые сверху."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM character_versions ORDER BY version DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [_character_row(row) for row in rows]

    # ------------------------------------------------------------------- read
    def memory_facts(self, *, status: str = "", limit: int = 100) -> list[dict[str, Any]]:
        """Факты памяти, свежие сверху. ``status`` фильтрует active/superseded."""
        query = """
            SELECT id, session_id, kind, scope, durability, event_at, fact_key,
                   text, evidence, importance, confidence, status, created_at,
                   updated_at, replaced_by, conflict_with, resolved_by
            FROM memory_facts
        """
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def memory_fact(self, fact_id: int) -> dict[str, Any] | None:
        """Один факт по id или ``None``. Нужен для пары конфликта."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT id, session_id, kind, scope, durability, event_at, fact_key,
                       text, evidence, importance, confidence, status, created_at,
                       updated_at, replaced_by, conflict_with, resolved_by
                FROM memory_facts
                WHERE id = ?
                """,
                (int(fact_id),),
            ).fetchone()
        return dict(row) if row is not None else None

    def memory_fact_count(self, *, status: str = "active") -> int:
        """Сколько фактов в выбранном состоянии."""
        with self._connection() as conn:
            if status:
                row = conn.execute(
                    "SELECT COUNT(*) AS total FROM memory_facts WHERE status = ?",
                    (status,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS total FROM memory_facts").fetchone()
        return int(row["total"])

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
        """Резюме одной сессии с временем её начала или ``None``."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT
                    m.session_id, m.summary, m.created_at, m.model,
                    m.prompt_version, m.data, s.started_at
                FROM summaries m
                LEFT JOIN sessions s ON s.id = m.session_id
                WHERE m.session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return _summary_row(row) if row is not None else None

    def summaries(self, limit: int | None = 20) -> list[dict[str, Any]]:
        """Последние резюме, новые сверху. ``None`` — все резюме."""
        query = """
            SELECT
                m.session_id, m.summary, m.created_at, m.model,
                m.prompt_version, m.data, s.started_at
            FROM summaries m
            LEFT JOIN sessions s ON s.id = m.session_id
            ORDER BY m.created_at DESC, m.rowid DESC
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
    fact_columns = {row["name"] for row in conn.execute("PRAGMA table_info(memory_facts)")}
    if fact_columns and "durability" not in fact_columns:
        # Старые факты не классифицированы. Постоянные безопаснее:
        # конфликт уйдёт на проверку, а не заменит запись молча.
        conn.execute(
            "ALTER TABLE memory_facts ADD COLUMN durability TEXT NOT NULL DEFAULT 'permanent'"
        )
    if fact_columns and "scope" not in fact_columns:
        # Старые записи дозаполнит MemoryManager.maintain() по тексту.
        conn.execute("ALTER TABLE memory_facts ADD COLUMN scope TEXT NOT NULL DEFAULT 'trait'")
    if fact_columns and "event_at" not in fact_columns:
        # Пустая дата означает «не событие»; для событий её ставит менеджер.
        conn.execute("ALTER TABLE memory_facts ADD COLUMN event_at TEXT NOT NULL DEFAULT ''")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_messages_position
            ON messages(session_id, position)
        """
    )
    conn.execute("DROP INDEX IF EXISTS idx_messages_session")


def _character_row(row: sqlite3.Row) -> dict[str, Any]:
    """Версия манеры с разобранными JSON-слоями."""
    result = dict(row)
    result["style"] = _parse_data(result.get("style"))
    result["moment"] = _parse_data(result.get("moment"))
    return result


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
