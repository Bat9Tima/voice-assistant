"""Семантическая память по резюме прошлых разговоров.

Разделение источников:

* SQLite (``SessionStore``) — источник правды: сырые сессии, реплики и
  тексты резюме.
* ChromaDB — пересобираемый индекс: векторы резюме и минимум метаданных.
* ``profile.txt`` — постоянный профиль пользователя. Он не смешивается с
  эпизодической памятью: в промпте это отдельные блоки, профиль главнее.

Ошибки Voyage или Chroma не ломают разговор: ``remember`` просто
пропускает обновление индекса, ``recall`` возвращает пустой список.
Вектор — производные данные, его всегда можно пересчитать из резюме:
``restore`` достраивает индекс по SQLite после смены модели или версии.

У каждого вектора в метаданных лежат ``embedding_model`` и
``index_version``. Поиск видит только векторы текущей пары: старые не
попадают в выдачу, пока ``restore`` их не пересчитает. Это защищает от
тихого смешивания векторов разных моделей.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .embeddings import EmbeddingError, VoyageEmbedder
from .memory_manager import MemoryManager
from .session_store import SessionStore

log = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.35
DEFAULT_MIN_MARGIN = 0.05
DEFAULT_MAX_CHARS = 2400
DEFAULT_COLLECTION = "summaries"
# Дневной дайджест: сегодняшние содержательные сессии добавляются к
# семантическим находкам, даже если поиск их не выбрал.
DEFAULT_DIGEST_MIN_MESSAGES = 6
DEFAULT_DIGEST_LIMIT = 3
DEFAULT_DIGEST_ITEM_CHARS = 320
# Версия раскладки индекса. Меняйте, если изменился способ подготовки
# текста или модель: старые векторы перестанут участвовать в поиске,
# а restore пересчитает их из резюме в SQLite.
INDEX_VERSION = 1
RESTORE_BATCH = 16
# Насколько ниже порога пускаем лучший результат, если он оторвался
# от второго. Так находятся перефразировки, а «погода» не цепляет мусор.
MARGIN_WINDOW = 0.1
MEMORY_HEADER = (
    "Эпизодическая память: краткие резюме прошлых разговоров, не текущего.\n"
    "Это не факты из профиля. Если воспоминание противоречит профилю, "
    "верь профилю.\n"
    "Используй воспоминания, только если они помогают ответить. Не "
    "пересказывай извинения Луны и её слова о том, что она чего-то не "
    "помнит: это не факт о разговоре."
)


@dataclass(frozen=True)
class VectorHit:
    """Найденный вектор: id сессии, близость и метаданные."""

    session_id: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MemoryHit:
    """Воспоминание: резюме из SQLite и его близость к запросу."""

    session_id: str
    summary: str
    score: float
    created_at: str = ""


class VectorStore(Protocol):
    """Минимальный интерфейс хранилища векторов.

    Chroma — основная реализация, но сервис памяти её не требует:
    в тестах подставляется простая реализация в памяти.
    """

    def upsert(
        self,
        session_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None: ...

    def search(
        self,
        embedding: list[float],
        limit: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]: ...

    def indexed(self) -> dict[str, dict[str, Any]]: ...

    def delete(self, session_id: str) -> None: ...

    def count(self) -> int: ...

    def close(self) -> None: ...


class ChromaVectorStore:
    """Векторы в ChromaDB. База лежит в ``state/chroma`` и пересобираема.

    Импорт ``chromadb`` ленивый: если пакет не установлен, приложение
    работает без памяти, а не падает на старте.
    """

    def __init__(
        self,
        path: Path | str,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        import chromadb
        from chromadb.config import Settings

        self.path = Path(path)
        self.collection_name = collection or DEFAULT_COLLECTION
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False),
        )
        # Косинус: score = 1 - distance, порог задаётся в привычных долях.
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        session_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        clean = {
            key: value
            for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool))
        }
        self._collection.upsert(
            ids=[session_id],
            embeddings=[embedding],
            metadatas=[clean],
        )

    def search(
        self,
        embedding: list[float],
        limit: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        limit = max(1, int(limit))
        request: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": limit,
            "include": ["metadatas", "distances"],
        }
        if where:
            request["where"] = where
        result = self._collection.query(**request)
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        hits: list[VectorHit] = []
        for position, session_id in enumerate(ids):
            distance = distances[position] if position < len(distances) else 1.0
            metadata = metadatas[position] if position < len(metadatas) else {}
            hits.append(
                VectorHit(
                    session_id=str(session_id),
                    score=1.0 - float(distance),
                    metadata=dict(metadata or {}),
                )
            )
        return hits

    def indexed(self) -> dict[str, dict[str, Any]]:
        """Метаданные всех векторов: id сессии → словарь без эмбеддинга."""
        result = self._collection.get(include=["metadatas"])
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        found: dict[str, dict[str, Any]] = {}
        for position, session_id in enumerate(ids):
            metadata = metadatas[position] if position < len(metadatas) else {}
            found[str(session_id)] = dict(metadata or {})
        return found

    def delete(self, session_id: str) -> None:
        self._collection.delete(ids=[session_id])

    def count(self) -> int:
        return int(self._collection.count())

    def close(self) -> None:
        """Освобождает файлы Chroma: на Windows иначе не удалить папку."""
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass


class MemoryService:
    """Связка «эмбеддер + векторы + SQLite» с безопасными ошибками."""

    def __init__(
        self,
        embedder: VoyageEmbedder,
        store: SessionStore,
        vectors: VectorStore,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        min_margin: float = DEFAULT_MIN_MARGIN,
        max_chars: int = DEFAULT_MAX_CHARS,
        digest_min_messages: int = DEFAULT_DIGEST_MIN_MESSAGES,
        digest_limit: int = DEFAULT_DIGEST_LIMIT,
        digest_item_chars: int = DEFAULT_DIGEST_ITEM_CHARS,
        index_version: int = INDEX_VERSION,
        manager: MemoryManager | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.vectors = vectors
        # Факты менеджера идут первыми: это уже проверенные утверждения,
        # а не пересказ целой сессии.
        self.manager = manager
        self.top_k = max(1, int(top_k))
        self.min_score = float(min_score)
        self.min_margin = max(0.0, float(min_margin))
        self.max_chars = max(0, int(max_chars))
        self.digest_min_messages = max(1, int(digest_min_messages))
        self.digest_limit = max(0, int(digest_limit))
        self.digest_item_chars = max(0, int(digest_item_chars))
        self.index_version = max(1, int(index_version))

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        store: SessionStore,
        *,
        manager: MemoryManager | None = None,
    ) -> MemoryService | None:
        """Собирает память из секции ``memory``. ``None`` — память выключена."""
        section = (cfg or {}).get("memory") or {}
        if not bool(section.get("enabled", False)):
            return None
        embedder = VoyageEmbedder.from_config(cfg)
        if not embedder.available():
            log.warning("Память включена, но ключ или модель эмбеддингов не заданы")
            return None
        try:
            vectors = ChromaVectorStore(
                str(section.get("path") or "state/chroma"),
                str(section.get("collection") or DEFAULT_COLLECTION),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Chroma недоступна, память выключена: %s", exc)
            return None
        return cls(
            embedder,
            store,
            vectors,
            top_k=int(section.get("top_k") or DEFAULT_TOP_K),
            min_score=float(section.get("min_score", DEFAULT_MIN_SCORE)),
            min_margin=float(section.get("min_margin", DEFAULT_MIN_MARGIN)),
            max_chars=int(section.get("max_chars") or DEFAULT_MAX_CHARS),
            digest_min_messages=_int_setting(
                section.get("digest_min_messages"), DEFAULT_DIGEST_MIN_MESSAGES
            ),
            digest_limit=_int_setting(
                section.get("digest_limit"), DEFAULT_DIGEST_LIMIT
            ),
            digest_item_chars=_int_setting(
                section.get("digest_item_chars"), DEFAULT_DIGEST_ITEM_CHARS
            ),
            index_version=int(section.get("index_version") or INDEX_VERSION),
            manager=manager,
        )

    # ------------------------------------------------------------------ write
    def remember(self, session_id: str, summary: str) -> bool:
        """Индексирует резюме. Сбой только логируется: диалог важнее."""
        text = (summary or "").strip()
        if not text or not session_id:
            return False
        try:
            vector = self.embedder.embed([text], input_type="document")[0]
            self.vectors.upsert(
                session_id,
                vector,
                self._metadata(vector),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Память не обновлена для сессии %s: %s", session_id, exc)
            return False
        log.info("Память: резюме сессии %s проиндексировано", session_id)
        return True

    def restore(self) -> int:
        """Достраивает индекс по резюме из SQLite. Возвращает число векторов.

        Пересчитывает только те резюме, чей вектор отсутствует или сделан
        другой моделью/версией. Устаревшие векторы без резюме удаляются.
        Сбой эмбеддера не роняет приложение: что успели — то пересчитали.
        """
        try:
            indexed = self.vectors.indexed()
        except Exception as exc:  # noqa: BLE001
            log.warning("Память: не удалось прочитать индекс: %s", exc)
            return 0
        try:
            rows = self.store.summaries(limit=None)
        except Exception as exc:  # noqa: BLE001
            log.warning("Память: не удалось прочитать резюме: %s", exc)
            return 0

        known = {str(row["session_id"]) for row in rows}
        for orphan_id in sorted(set(indexed) - known):
            self.forget(orphan_id)

        stale = [row for row in rows if not self._is_current(indexed.get(str(row["session_id"])))]
        if not stale:
            log.info("Память: индекс актуален, резюме %s", len(rows))
            return 0

        restored = 0
        for start in range(0, len(stale), RESTORE_BATCH):
            batch = stale[start : start + RESTORE_BATCH]
            texts = [str(row["summary"]) for row in batch]
            try:
                vectors = self.embedder.embed(texts, input_type="document")
            except Exception as exc:  # noqa: BLE001
                log.warning("Память: векторы не пересчитаны: %s", exc)
                break
            for row, vector in zip(batch, vectors, strict=False):
                session_id = str(row["session_id"])
                try:
                    self.vectors.upsert(session_id, vector, self._metadata(vector))
                except Exception as exc:  # noqa: BLE001
                    log.warning("Память: вектор сессии %s не сохранён: %s", session_id, exc)
                    continue
                restored += 1
        log.info("Память: пересчитано резюме %s из %s", restored, len(rows))
        return restored

    def _metadata(self, vector: list[float]) -> dict[str, Any]:
        """Пометки вектора: по ним поиск отсекает чужую модель и версию."""
        return {
            "embedding_model": self.embedder.model,
            "index_version": self.index_version,
            "dim": len(vector),
            "created_at": _now_iso(),
        }

    def _is_current(self, metadata: dict[str, Any] | None) -> bool:
        if not metadata:
            return False
        if str(metadata.get("embedding_model") or "") != self.embedder.model:
            return False
        try:
            version = int(metadata.get("index_version") or 0)
        except (TypeError, ValueError):
            return False
        return version == self.index_version

    def _index_filter(self) -> dict[str, Any]:
        """Фильтр Chroma: только векторы текущей модели и версии."""
        return {
            "$and": [
                {"embedding_model": self.embedder.model},
                {"index_version": self.index_version},
            ]
        }

    def forget(self, session_id: str) -> bool:
        """Убирает вектор удалённой сессии. Сбой только логируется."""
        try:
            self.vectors.delete(session_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("Вектор сессии %s не удалён: %s", session_id, exc)
            return False
        return True

    # ------------------------------------------------------------------- read
    def recall(self, query: str) -> list[MemoryHit]:
        """Ищет прошлые резюме по смыслу. Любой сбой — пустой список."""
        text = " ".join((query or "").split())
        if not text:
            return []
        try:
            vector = self.embedder.embed([text], input_type="query")[0]
            hits = self.vectors.search(vector, self.top_k * 2, self._index_filter())
        except EmbeddingError as exc:
            log.warning("Поиск по памяти пропущен: %s", exc)
            return []
        except Exception as exc:  # noqa: BLE001
            log.warning("Chroma не ответила, память пропущена: %s", exc)
            return []

        found: list[MemoryHit] = []
        ranked = sorted(hits, key=lambda item: item.score, reverse=True)
        for index, hit in enumerate(ranked):
            if not self._relevant(hit.score, ranked, index):
                continue
            saved = self.store.summary(hit.session_id)
            if saved is None:
                continue
            found.append(
                MemoryHit(
                    session_id=hit.session_id,
                    summary=str(saved["summary"]),
                    score=hit.score,
                    created_at=str(saved.get("created_at") or ""),
                )
            )
            if len(found) >= self.top_k:
                break
        return found

    def _relevant(self, score: float, ranked: list[VectorHit], index: int) -> bool:
        """Проходит ли результат по порогу или по отрыву от второго места.

        Перефразировка часто даёт чуть меньший score, чем прямой вопрос.
        Если лучший результат заметно выше второго, ему можно доверять
        и ниже порога. Остальные места требуют обычного порога.
        """
        if score >= self.min_score:
            return True
        if index != 0 or self.min_margin <= 0:
            return False
        second = ranked[1].score if len(ranked) > 1 else 0.0
        gap = score - second
        return score >= self.min_score - MARGIN_WINDOW and gap >= self.min_margin

    def block(self, query: str, *, now: datetime | None = None) -> str:
        """Готовый блок для системного промпта или пустая строка.

        Сначала идут отдельные факты менеджера: это проверенные факты,
        уже очищенные от повторов и противоречий. Потом — сегодняшние
        содержательные сессии, затем семантические находки без повторов.
        """
        facts = self.facts_block(query)
        hits = self.recall(query)
        digest = self.digest(now=now)
        if not facts and not hits and not digest:
            return ""
        lines: list[str] = []
        total = 0
        if facts:
            lines.append(facts)
            total = len(facts) + 1
        if hits or digest:
            lines.append(MEMORY_HEADER)
            total += len(MEMORY_HEADER) + 1
        seen: set[str] = set()
        for hit in digest:
            line = f"- Сегодня, {_clock(hit.created_at)}: {hit.summary}"
            if self.max_chars and total + len(line) + 1 > self.max_chars:
                break
            lines.append(line)
            total += len(line) + 1
            seen.add(hit.session_id)
        for hit in hits:
            if hit.session_id in seen:
                continue
            moment = _short_date(hit.created_at)
            line = f"- {moment}{hit.summary}" if moment else f"- {hit.summary}"
            if self.max_chars and total + len(line) + 1 > self.max_chars:
                break
            lines.append(line)
            total += len(line) + 1
            seen.add(hit.session_id)
        return "\n".join(lines) if len(lines) > 1 or facts else ""

    def facts_block(self, query: str = "") -> str:
        """Факты менеджера. Сбой чтения не должен ломать системный промпт."""
        if self.manager is None:
            return ""
        try:
            return self.manager.block(query)
        except Exception as exc:  # noqa: BLE001
            log.warning("Память: факты недоступны: %s", exc)
            return ""

    def digest(self, *, now: datetime | None = None) -> list[MemoryHit]:
        """Сегодняшние завершённые сессии с достаточным числом реплик.

        Нужен как страховка от чистого semantic search: мета-вопрос
        «что мы обсуждали» плохо совпадает с длинным резюме, зато свежая
        сессия важна сама по себе. Ошибка чтения базы не ломает диалог.
        """
        if self.digest_limit <= 0:
            return []
        try:
            rows = self.store.summaries_with_sessions(
                limit=max(20, self.digest_limit * 5)
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Память: дневной дайджест недоступен: %s", exc)
            return []
        today = _as_local(now).date()
        found: list[MemoryHit] = []
        for row in rows:
            if str(row.get("status") or "") != "completed":
                continue
            if int(row.get("messages_count") or 0) < self.digest_min_messages:
                continue
            started = _parse_moment(str(row.get("started_at") or ""))
            if started is None or started.date() != today:
                continue
            text = str(row.get("summary") or "").strip()
            if not text:
                continue
            found.append(
                MemoryHit(
                    session_id=str(row["session_id"]),
                    summary=_clip(text, self.digest_item_chars),
                    score=0.0,
                    created_at=str(row.get("started_at") or ""),
                )
            )
            if len(found) >= self.digest_limit:
                break
        return found

    def close(self) -> None:
        """Освобождает ресурсы индекса при выходе программы."""
        close = getattr(self.vectors, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception:  # noqa: BLE001
            log.warning("Не удалось закрыть хранилище памяти")


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _int_setting(value: Any, default: int) -> int:
    """Целое из конфига: ``None`` — значение по умолчанию, ноль разрешён."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_local(moment: datetime | None) -> datetime:
    if moment is None:
        return datetime.now().astimezone()
    if moment.tzinfo is None:
        return moment.astimezone()
    return moment


def _parse_moment(value: str) -> datetime | None:
    try:
        return _as_local(datetime.fromisoformat(value))
    except (TypeError, ValueError):
        return None


def _clock(value: str) -> str:
    """«17:18» из ISO-строки; мусор не роняет строку памяти."""
    moment = _parse_moment(value)
    return moment.strftime("%H:%M") if moment is not None else "--:--"


def _clip(text: str, limit: int) -> str:
    """Сжимает резюме до лимита по границе предложения или слова."""
    flat = " ".join((text or "").split())
    if not limit or len(flat) <= limit:
        return flat
    cut = flat[:limit]
    floor = limit // 2
    for marker in (". ", "! ", "? ", "; ", ", "):
        position = cut.rfind(marker)
        if position >= floor:
            return cut[: position + len(marker)].rstrip() + "…"
    position = cut.rfind(" ")
    if position >= floor:
        cut = cut[:position]
    return cut.rstrip(" ,;:") + "…"


def _short_date(value: str) -> str:
    """«16.11.2026 01:45» из ISO-строки; мусор даёт пустую строку."""
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return ""
    return moment.strftime("%d.%m.%Y %H:%M: ")
