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

import hashlib
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Protocol

from .embeddings import EmbeddingError, VoyageEmbedder
from .memory_manager import HIGH, MAX_ACTIVE_FACTS, MEDIUM, MemoryManager
from .session_store import SessionStore

log = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.35
DEFAULT_MIN_MARGIN = 0.05
DEFAULT_MAX_CHARS = 2400
DEFAULT_COLLECTION = "summaries"
# Факты лежат отдельной коллекцией: у них короткий текст и другая шкала
# близости, поэтому смешивать их с резюме в одном поиске нельзя.
DEFAULT_FACTS_COLLECTION = "facts"
DEFAULT_FACTS_TOP_K = 4
DEFAULT_FACTS_MIN_SCORE = 0.42
FACT_VECTOR_PREFIX = "fact:"
# Общие вопросы о памяти не несут темы: смысловой поиск по ним вернул бы
# случайные факты. Для них остаётся обычный список свежих фактов.
_GENERAL_QUERY_WORDS = frozenset(
    {
        "а",
        "было",
        "был",
        "была",
        "были",
        "вспомни",
        "вспоминаешь",
        "говорили",
        "знаешь",
        "и",
        "какая",
        "какие",
        "какой",
        "кто",
        "мы",
        "мне",
        "обо",
        "обсуждали",
        "память",
        "помни",
        "помнишь",
        "помню",
        "расскажи",
        "сказал",
        "сказала",
        "ты",
        "что",
        "я",
    }
)
# Дневной дайджест: сегодняшние содержательные сессии добавляются к
# семантическим находкам, даже если поиск их не выбрал.
DEFAULT_DIGEST_MIN_MESSAGES = 6
DEFAULT_DIGEST_LIMIT = 3
DEFAULT_DIGEST_ITEM_CHARS = 320
# Последние разговоры по времени, а не по смыслу: это страховка от
# того, что свежая тема проиграет более заметному старому резюме.
DEFAULT_RECENT_LIMIT = 2
DEFAULT_RECENT_ITEM_CHARS = 320
# Свежесть как небольшой бонус к смысловой близости: она решает только
# близкие случаи, а явно более похожее старое резюме всё равно остаётся
# первым. Переиндексация не нужна: дата берётся из SQLite.
RECENCY_BONUS = 0.05
RECENCY_DAYS = 7.0
# Версия раскладки индекса. Меняйте, если изменился способ подготовки
# текста или модель: старые векторы перестанут участвовать в поиске,
# а restore пересчитает их из резюме в SQLite.
INDEX_VERSION = 2
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
    """Найденный вектор: id, близость и метаданные."""

    vector_id: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MemoryHit:
    """Воспоминание: тема из SQLite и её близость к запросу.

    ``topic_no`` — номер темы внутри сессии, ``-1`` — вся сессия целиком
    (дайджест или последний разговор без разбивки на темы).
    """

    session_id: str
    summary: str
    score: float
    created_at: str = ""
    topic_no: int = 0
    topic_title: str = ""


class VectorStore(Protocol):
    """Минимальный интерфейс хранилища векторов.

    Chroma — основная реализация, но сервис памяти её не требует:
    в тестах подставляется простая реализация в памяти.
    """

    def upsert(
        self,
        vector_id: str,
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

    def delete_session(self, session_id: str) -> None: ...

    def delete_id(self, vector_id: str) -> None: ...

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
        vector_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        clean = {
            key: value
            for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool))
        }
        self._collection.upsert(
            ids=[vector_id],
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
        for position, vector_id in enumerate(ids):
            distance = distances[position] if position < len(distances) else 1.0
            metadata = metadatas[position] if position < len(metadatas) else {}
            hits.append(
                VectorHit(
                    vector_id=str(vector_id),
                    score=1.0 - float(distance),
                    metadata=dict(metadata or {}),
                )
            )
        return hits

    def indexed(self) -> dict[str, dict[str, Any]]:
        """Метаданные всех векторов: id вектора → словарь без эмбеддинга."""
        result = self._collection.get(include=["metadatas"])
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        found: dict[str, dict[str, Any]] = {}
        for position, vector_id in enumerate(ids):
            metadata = metadatas[position] if position < len(metadatas) else {}
            found[str(vector_id)] = dict(metadata or {})
        return found

    def delete_session(self, session_id: str) -> None:
        """Удаляет все векторы сессии: у темы свой id, но сессия общая."""
        self._collection.delete(where={"session_id": session_id})

    def delete_id(self, vector_id: str) -> None:
        self._collection.delete(ids=[vector_id])

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
        recent_limit: int = DEFAULT_RECENT_LIMIT,
        recent_item_chars: int = DEFAULT_RECENT_ITEM_CHARS,
        index_version: int = INDEX_VERSION,
        facts_vectors: VectorStore | None = None,
        facts_top_k: int = DEFAULT_FACTS_TOP_K,
        facts_min_score: float = DEFAULT_FACTS_MIN_SCORE,
        manager: MemoryManager | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.vectors = vectors
        # Факты менеджера идут первыми: это уже проверенные утверждения,
        # а не пересказ целой сессии.
        self.manager = manager
        # Отдельный индекс фактов. Он необязателен: без него факты
        # работают по словам, как раньше.
        self.facts_vectors = facts_vectors
        self.facts_top_k = max(0, int(facts_top_k))
        self.facts_min_score = float(facts_min_score)
        self.top_k = max(1, int(top_k))
        self.min_score = float(min_score)
        self.min_margin = max(0.0, float(min_margin))
        self.max_chars = max(0, int(max_chars))
        self.digest_min_messages = max(1, int(digest_min_messages))
        self.digest_limit = max(0, int(digest_limit))
        self.digest_item_chars = max(0, int(digest_item_chars))
        self.recent_limit = max(0, int(recent_limit))
        self.recent_item_chars = max(0, int(recent_item_chars))
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
        facts_vectors: ChromaVectorStore | None = None
        try:
            facts_vectors = ChromaVectorStore(
                str(section.get("path") or "state/chroma"),
                str(section.get("facts_collection") or DEFAULT_FACTS_COLLECTION),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Chroma фактов недоступна, поиск по смыслу выключен: %s", exc)
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
            digest_limit=_int_setting(section.get("digest_limit"), DEFAULT_DIGEST_LIMIT),
            digest_item_chars=_int_setting(
                section.get("digest_item_chars"), DEFAULT_DIGEST_ITEM_CHARS
            ),
            recent_limit=_int_setting(section.get("recent_limit"), DEFAULT_RECENT_LIMIT),
            recent_item_chars=_int_setting(
                section.get("recent_item_chars"), DEFAULT_RECENT_ITEM_CHARS
            ),
            index_version=int(section.get("index_version") or INDEX_VERSION),
            facts_vectors=facts_vectors,
            facts_top_k=_int_setting(section.get("facts_top_k"), DEFAULT_FACTS_TOP_K),
            facts_min_score=float(section.get("facts_min_score", DEFAULT_FACTS_MIN_SCORE)),
            manager=manager,
        )

    # ------------------------------------------------------------------ write
    def remember(
        self,
        session_id: str,
        summary: str,
        *,
        topics: list[dict[str, Any]] | None = None,
        created_at: datetime | None = None,
    ) -> bool:
        """Индексирует темы резюме, а без них — резюме целиком.

        Каждая тема становится отдельным вектором: так у неё есть дата и
        номер внутри разговора. Сбой только логируется: диалог важнее.
        """
        if not session_id:
            return False
        items = _summary_items({"summary": summary, "data": {"topics": topics or []}})
        if not items:
            return False
        try:
            vectors = self.embedder.embed(
                [_embedding_text(item) for item in items], input_type="document"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Память не обновлена для сессии %s: %s", session_id, exc)
            return False
        stamp = created_at or _now()
        saved = 0
        for item, vector in zip(items, vectors, strict=False):
            vector_id = _vector_id(session_id, item["topic_no"])
            try:
                self.vectors.upsert(
                    vector_id,
                    vector,
                    self._metadata(vector, session_id=session_id, item=item, created_at=stamp),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Память: тема %s не сохранена: %s", vector_id, exc)
                continue
            saved += 1
        if saved:
            log.info("Память: сессия %s проиндексирована, тем %s", session_id, saved)
        return saved > 0

    def sync_facts(self) -> int:
        """Синхронизирует векторный индекс фактов с SQLite.

        SQLite — источник правды. Индексируются только активные важные
        факты; вытесненные, удалённые и потерявшие важность векторы
        удаляются. Пересчитываются только изменившиеся тексты: обычное
        резюме добавляет или меняет пару фактов, а не всю память.
        """
        if self.facts_vectors is None or self.manager is None or self.facts_top_k <= 0:
            return 0
        try:
            indexed = self.facts_vectors.indexed()
        except Exception as exc:  # noqa: BLE001
            log.warning("Память: индекс фактов недоступен: %s", exc)
            return 0
        try:
            rows = self.store.memory_facts(status="active", limit=MAX_ACTIVE_FACTS)
        except Exception as exc:  # noqa: BLE001
            log.warning("Память: факты для индекса недоступны: %s", exc)
            return 0
        expected = {
            _fact_vector_id(row["id"]): row for row in rows if row["importance"] in (HIGH, MEDIUM)
        }
        for orphan in sorted(set(indexed) - set(expected)):
            try:
                self.facts_vectors.delete_id(orphan)
            except Exception as exc:  # noqa: BLE001
                log.warning("Память: устаревший вектор факта %s не удалён: %s", orphan, exc)
        stale = [
            vector_id
            for vector_id, row in expected.items()
            if not self._fact_current(indexed.get(vector_id), row)
        ]
        if not stale:
            return 0
        saved = 0
        for start in range(0, len(stale), RESTORE_BATCH):
            batch = stale[start : start + RESTORE_BATCH]
            texts = [str(expected[vector_id]["text"]) for vector_id in batch]
            try:
                vectors = self.embedder.embed(texts, input_type="document")
            except Exception as exc:  # noqa: BLE001
                log.warning("Память: векторы фактов не пересчитаны: %s", exc)
                break
            for vector_id, vector in zip(batch, vectors, strict=False):
                try:
                    self.facts_vectors.upsert(
                        vector_id,
                        vector,
                        self._fact_metadata(vector, expected[vector_id]),
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("Память: вектор факта %s не сохранён: %s", vector_id, exc)
                    continue
                saved += 1
        if saved:
            log.info("Память: фактов проиндексировано %s", saved)
        return saved

    def restore(self) -> int:
        """Достраивает оба индекса по SQLite. Возвращает число векторов.

        Темы и факты восстанавливаются независимо: сбой одного индекса
        не мешает другому.
        """
        return self._restore_topics() + self.sync_facts()

    def _restore_topics(self) -> int:
        """Достраивает индекс тем по резюме из SQLite. Возвращает число тем.

        Пересчитывает только темы, чей вектор отсутствует или сделан
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

        expected: dict[str, tuple[str, dict[str, Any], datetime | None]] = {}
        for row in rows:
            session_id = str(row["session_id"])
            started = _parse_moment(str(row.get("started_at") or row.get("created_at") or ""))
            for item in _summary_items(row):
                expected[_vector_id(session_id, item["topic_no"])] = (session_id, item, started)

        for orphan in sorted(set(indexed) - set(expected)):
            try:
                self.vectors.delete_id(orphan)
            except Exception as exc:  # noqa: BLE001
                log.warning("Память: устаревший вектор %s не удалён: %s", orphan, exc)

        stale = [
            vector_id for vector_id in expected if not self._is_current(indexed.get(vector_id))
        ]
        if not stale:
            log.info("Память: индекс актуален, тем %s", len(expected))
            return 0

        restored = 0
        for start in range(0, len(stale), RESTORE_BATCH):
            batch = stale[start : start + RESTORE_BATCH]
            texts = [_embedding_text(expected[vector_id][1]) for vector_id in batch]
            try:
                vectors = self.embedder.embed(texts, input_type="document")
            except Exception as exc:  # noqa: BLE001
                log.warning("Память: векторы не пересчитаны: %s", exc)
                break
            for vector_id, vector in zip(batch, vectors, strict=False):
                session_id, item, started = expected[vector_id]
                try:
                    self.vectors.upsert(
                        vector_id,
                        vector,
                        self._metadata(
                            vector, session_id=session_id, item=item, created_at=started
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("Память: вектор %s не сохранён: %s", vector_id, exc)
                    continue
                restored += 1
        log.info("Память: пересчитано тем %s (резюме %s)", restored, len(rows))
        return restored

    def _metadata(
        self,
        vector: list[float],
        *,
        session_id: str = "",
        item: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Пометки вектора: модель, версия, сессия, тема и время разговора."""
        item = item or {"topic_no": 0, "title": ""}
        stamp = created_at or _now()
        return {
            "embedding_model": self.embedder.model,
            "index_version": self.index_version,
            "dim": len(vector),
            "indexed_at": _now_iso(),
            "session_id": session_id,
            "topic_no": int(item.get("topic_no") or 0),
            "topic_title": str(item.get("title") or ""),
            "created_ts": int(stamp.timestamp()),
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

    def _fact_metadata(self, vector: list[float], row: dict[str, Any]) -> dict[str, Any]:
        """Пометки вектора факта: модель, версия, id, важность и хеш текста."""
        return {
            "embedding_model": self.embedder.model,
            "index_version": self.index_version,
            "dim": len(vector),
            "indexed_at": _now_iso(),
            "fact_id": int(row["id"]),
            "fact_importance": str(row.get("importance") or ""),
            "fact_text_hash": _text_hash(str(row.get("text") or "")),
        }

    def _fact_current(self, metadata: dict[str, Any] | None, row: dict[str, Any]) -> bool:
        """Совпадает ли вектор с текущей записью SQLite.

        Дату не сравниваем: слияние может обновить уверенность, не меняя
        текст. Хеш и важность ловят всё, что влияет на выдачу.
        """
        if not metadata:
            return False
        if str(metadata.get("embedding_model") or "") != self.embedder.model:
            return False
        try:
            version = int(metadata.get("index_version") or 0)
        except (TypeError, ValueError):
            return False
        if version != self.index_version:
            return False
        if str(metadata.get("fact_importance") or "") != str(row.get("importance") or ""):
            return False
        return str(metadata.get("fact_text_hash") or "") == _text_hash(str(row.get("text") or ""))

    def _index_filter(self, day: date | None = None) -> dict[str, Any]:
        """Фильтр Chroma: своя модель и версия, при необходимости — день."""
        clauses: list[dict[str, Any]] = [
            {"embedding_model": self.embedder.model},
            {"index_version": self.index_version},
        ]
        if day is not None:
            start, end = _day_bounds(day)
            clauses.append({"created_ts": {"$gte": start}})
            clauses.append({"created_ts": {"$lt": end}})
        return {"$and": clauses}

    def forget(self, session_id: str) -> bool:
        """Убирает все векторы удалённой сессии. Сбой только логируется."""
        try:
            self.vectors.delete_session(session_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("Векторы сессии %s не удалены: %s", session_id, exc)
            return False
        return True

    def remembered(self, limit: int = 8) -> list[dict[str, Any]]:
        """Устойчивые факты о пользователе для прямого ответа."""
        if self.manager is None:
            return []
        try:
            return self.manager.remembered(limit)
        except Exception as exc:  # noqa: BLE001
            log.warning("Память: факты для ответа недоступны: %s", exc)
            return []

    def forget_facts(self, query: str) -> list[dict[str, Any]]:
        """Закрывает факты по просьбе «забудь …» и чинит индекс."""
        if self.manager is None:
            return []
        try:
            rows = self.manager.forget(query)
        except Exception as exc:  # noqa: BLE001
            log.warning("Память: не удалось забыть факты: %s", exc)
            return []
        if rows:
            self.sync_facts()
        return rows

    # ------------------------------------------------------------------- read
    def recall(
        self,
        query: str,
        *,
        now: datetime | None = None,
        day: date | None = None,
    ) -> list[MemoryHit]:
        """Ищет темы по смыслу, при необходимости только за один день.

        Смысловая близость остаётся главной; бонус за свежесть решает
        близкие случаи. Если назван день, Chroma сразу отсекает чужие
        даты по ``created_ts``. Любой сбой — пустой список.
        """
        text = " ".join((query or "").split())
        if not text:
            return []
        try:
            vector = self.embedder.embed([text], input_type="query")[0]
            hits = self.vectors.search(vector, self.top_k * 2, self._index_filter(day))
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
            session_id = _hit_session_id(hit)
            topic_no = _hit_topic_no(hit)
            saved = self.store.summary(session_id)
            if saved is None:
                continue
            topic = _topic_at(saved, topic_no)
            found.append(
                MemoryHit(
                    session_id=session_id,
                    summary=_clip(
                        topic["text"] if topic else str(saved["summary"]),
                        self.digest_item_chars,
                    ),
                    score=hit.score,
                    created_at=str(saved.get("started_at") or saved.get("created_at") or ""),
                    topic_no=topic_no,
                    topic_title=str(topic["title"]) if topic else "",
                )
            )
        found.sort(
            key=lambda hit: hit.score + self._recency_bonus(hit.created_at, now),
            reverse=True,
        )
        return found[: self.top_k]

    def recall_facts(self, query: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Ищет факты по смыслу. Возвращает строки SQLite с полем ``score``.

        Поиск не заменяет совпадения по словам: он ловит перефразировки,
        когда в вопросе нет точного слова из факта. Общие вопросы о памяти
        пропускаются. Сбой эмбеддера или Chroma — пустой список.
        """
        if self.facts_vectors is None or self.manager is None:
            return []
        count = max(0, int(limit if limit is not None else self.facts_top_k))
        text = " ".join((query or "").split())
        if count <= 0 or not text or _is_general_query(text):
            return []
        try:
            vector = self.embedder.embed([text], input_type="query")[0]
            hits = self.facts_vectors.search(vector, count * 2, self._index_filter())
        except EmbeddingError as exc:
            log.warning("Поиск по фактам пропущен: %s", exc)
            return []
        except Exception as exc:  # noqa: BLE001
            log.warning("Chroma не ответила, поиск по фактам пропущен: %s", exc)
            return []
        found: list[dict[str, Any]] = []
        seen: set[int] = set()
        for hit in sorted(hits, key=lambda item: item.score, reverse=True):
            if hit.score < self.facts_min_score:
                continue
            fact_id = _hit_fact_id(hit)
            if fact_id is None or fact_id in seen:
                continue
            try:
                row = self.store.memory_fact(fact_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("Память: факт %s недоступен: %s", fact_id, exc)
                continue
            if row is None or row["status"] != "active":
                continue
            if row["importance"] not in (HIGH, MEDIUM):
                continue
            seen.add(fact_id)
            found.append(dict(row, score=hit.score))
            if len(found) >= count:
                break
        return found

    def _recency_bonus(self, created_at: str, now: datetime | None) -> float:
        """Небольшая надбавка за свежесть: максимум в день создания."""
        moment = _parse_moment(created_at)
        if moment is None:
            return 0.0
        age = (_as_local(now) - moment).total_seconds() / 86400
        if age <= 0:
            return RECENCY_BONUS
        if age >= RECENCY_DAYS:
            return 0.0
        return RECENCY_BONUS * (1.0 - age / RECENCY_DAYS)

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

        Порядок: факты менеджера, затем резюме по времени — последние
        разговоры раньше остальных сегодняшних, затем семантические
        находки. Если в вопросе назван день («вчера»), остаются только
        этот день: свежесть здесь важнее смысловой близости.
        """
        facts = self.facts_block(query)
        day = _period_day(query, now)
        hits = self.recall(query, now=now, day=day)
        if day is not None:
            hits = [hit for hit in hits if _is_on_day(hit.created_at, day)]
        # Если поиск уверенно нашёл нужное, хронология только мешает:
        # свежий разговор может быть совсем про другое. А на общий вопрос
        # или вопрос про день она, наоборот, важнее смысла.
        specific = bool(hits) and day is None
        recent = [] if specific else self.recent(day)
        digest = [] if day is not None or specific else self.digest(now=now)

        timeline: dict[tuple[str, int], MemoryHit] = {
            (hit.session_id, hit.topic_no): hit for hit in digest
        }
        # Свежее резюме важнее дневного дайджеста: при совпадении текста
        # оставляем именно его, с полным лимитом символов.
        timeline.update({(hit.session_id, hit.topic_no): hit for hit in recent})
        if day is not None:
            # В конкретный день важен порядок, а не сходство: все
            # воспоминания этого дня раскладываем по времени и темам.
            timeline.update({(hit.session_id, hit.topic_no): hit for hit in hits})
            hits = []
        ordered = sorted(
            timeline.values(),
            key=lambda hit: (_moment_key(hit.created_at), hit.topic_no),
            reverse=True,
        )

        if not facts and not ordered and not hits:
            return ""
        lines: list[str] = []
        total = 0
        if facts:
            lines.append(facts)
            total = len(facts) + 1
        if ordered or hits:
            lines.append(MEMORY_HEADER)
            total += len(MEMORY_HEADER) + 1
        seen: set[tuple[str, int]] = set()
        for hit in ordered:
            line = _dated_line(hit, now)
            if self.max_chars and total + len(line) + 1 > self.max_chars:
                break
            lines.append(line)
            total += len(line) + 1
            seen.add((hit.session_id, hit.topic_no))
        for hit in hits:
            key = (hit.session_id, hit.topic_no)
            if key in seen:
                continue
            line = _dated_line(hit, now)
            if self.max_chars and total + len(line) + 1 > self.max_chars:
                break
            lines.append(line)
            total += len(line) + 1
            seen.add(key)
        return "\n".join(lines) if len(lines) > 1 or facts else ""

    def recent(self, day: date | None = None) -> list[MemoryHit]:
        """Последние разговоры по времени, а не по смыслу.

        Без ``day`` берутся резюме последних завершённых сессий. С днём —
        темы этого дня, последняя тема первой: так отвечаем на «что было
        вчера», не полагаясь на векторное сходство. Ошибка чтения базы не
        ломает диалог.
        """
        if self.recent_limit <= 0:
            return []
        try:
            rows = self.store.summaries_with_sessions(limit=max(20, self.recent_limit * 10))
        except Exception as exc:  # noqa: BLE001
            log.warning("Память: последние разговоры недоступны: %s", exc)
            return []
        found: list[MemoryHit] = []
        sessions = 0
        for row in rows:
            if str(row.get("status") or "") != "completed":
                continue
            started = _parse_moment(str(row.get("started_at") or ""))
            if started is None:
                continue
            if day is not None and started.date() != day:
                continue
            items = _summary_items(row)
            if not items:
                continue
            session_id = str(row["session_id"])
            created_at = str(row.get("started_at") or row.get("created_at") or "")
            if day is None:
                text = str(row.get("summary") or "").strip()
                if not text:
                    text = " ".join(item["text"] for item in items)
                found.append(
                    MemoryHit(
                        session_id=session_id,
                        summary=_clip(text, self.recent_item_chars),
                        score=0.0,
                        created_at=created_at,
                        topic_no=-1,
                    )
                )
            else:
                for item in reversed(items):
                    found.append(
                        MemoryHit(
                            session_id=session_id,
                            summary=_clip(item["text"], self.recent_item_chars),
                            score=0.0,
                            created_at=created_at,
                            topic_no=int(item["topic_no"]),
                            topic_title=str(item["title"]),
                        )
                    )
            sessions += 1
            if sessions >= self.recent_limit:
                break
        return found

    def facts_block(self, query: str = "") -> str:
        """Факты менеджера и смысловые находки. Сбой не ломает промпт."""
        if self.manager is None:
            return ""
        try:
            related = self.recall_facts(query)
        except Exception as exc:  # noqa: BLE001
            log.warning("Память: смысловой поиск фактов недоступен: %s", exc)
            related = []
        try:
            return self.manager.block(query, related=related)
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
            rows = self.store.summaries_with_sessions(limit=max(20, self.digest_limit * 5))
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
                    topic_no=-1,
                )
            )
            if len(found) >= self.digest_limit:
                break
        return found

    def close(self) -> None:
        """Освобождает ресурсы индексов при выходе программы."""
        stores = [self.vectors]
        if self.facts_vectors is not None and self.facts_vectors is not self.vectors:
            stores.append(self.facts_vectors)
        for store in stores:
            close = getattr(store, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception:  # noqa: BLE001
                log.warning("Не удалось закрыть хранилище памяти")


def _now_iso() -> str:
    return _now().isoformat()


def _now() -> datetime:
    return datetime.now().astimezone()


def _vector_id(session_id: str, topic_no: int) -> str:
    """Id вектора: сессия и номер темы. Сессии — hex, двоеточие безопасно."""
    return f"{session_id}:{int(topic_no)}"


def _fact_vector_id(fact_id: int) -> str:
    """Id вектора факта не пересекается с id тем резюме."""
    return f"{FACT_VECTOR_PREFIX}{int(fact_id)}"


def _hit_fact_id(hit: VectorHit) -> int | None:
    try:
        return int(hit.metadata.get("fact_id") or 0) or None
    except (TypeError, ValueError):
        return None


def _text_hash(text: str) -> str:
    """Короткий хеш текста: меняется текст — пересчитывается вектор."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _is_general_query(text: str) -> bool:
    """Вопрос о памяти целиком, а не о конкретной теме."""
    words = {word.strip(".,!?;:()[]{}\"'«»—–-") for word in text.lower().replace("ё", "е").split()}
    words.discard("")
    return not words or words <= _GENERAL_QUERY_WORDS


def _summary_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Единицы индексации: темы v3 с текстом или резюме целиком.

    Старые резюме (v2 и раньше) не разбиваются на темы: у них есть только
    названия без описания, индексировать их по отдельности нечем.
    """
    data = row.get("data") if isinstance(row.get("data"), dict) else {}
    raw_topics = data.get("topics") if isinstance(data, dict) else None
    items: list[dict[str, Any]] = []
    if isinstance(raw_topics, list):
        for number, raw in enumerate(raw_topics):
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            items.append(
                {
                    "topic_no": number,
                    "title": title or text[:80],
                    "text": text,
                }
            )
    if items:
        return items
    text = str(row.get("summary") or "").strip()
    if text:
        items.append({"topic_no": 0, "title": "", "text": text})
    return items


def _embedding_text(item: dict[str, Any]) -> str:
    """Текст темы для вектора: название помогает попасть в запрос."""
    title = str(item.get("title") or "").strip()
    text = str(item.get("text") or "").strip()
    if title and text:
        return f"{title}. {text}"
    return text or title


def _hit_session_id(hit: VectorHit) -> str:
    session_id = str(hit.metadata.get("session_id") or "").strip()
    if session_id:
        return session_id
    return hit.vector_id.split(":", 1)[0]


def _hit_topic_no(hit: VectorHit) -> int:
    try:
        return int(hit.metadata.get("topic_no") or 0)
    except (TypeError, ValueError):
        return 0


def _topic_at(row: dict[str, Any], topic_no: int) -> dict[str, str] | None:
    """Тема из ``data`` по номеру. Старый формат и мусор дают ``None``."""
    data = row.get("data") if isinstance(row.get("data"), dict) else {}
    topics = data.get("topics") if isinstance(data, dict) else None
    if not isinstance(topics, list) or not 0 <= topic_no < len(topics):
        return None
    raw = topics[topic_no]
    if not isinstance(raw, dict):
        return None
    title = str(raw.get("title") or "").strip()
    text = str(raw.get("text") or "").strip()
    if not text:
        return None
    return {"title": title, "text": text}


def _day_bounds(day: date) -> tuple[int, int]:
    """Начало и конец местного дня в epoch-секундах."""
    start = datetime.combine(day, time.min).astimezone()
    end = start + timedelta(days=1)
    return int(start.timestamp()), int(end.timestamp())


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


def _period_day(query: str, now: datetime | None) -> date | None:
    """День, названный в вопросе словами. Иначе ``None``.

    «Позавчера» проверяется первым: оно содержит «вчера».
    """
    text = (query or "").lower()
    today = _as_local(now).date()
    if "позавчера" in text:
        return today - timedelta(days=2)
    if "вчера" in text:
        return today - timedelta(days=1)
    if "сегодня" in text:
        return today
    return None


def _is_on_day(value: str, day: date) -> bool:
    moment = _parse_moment(value)
    return moment is not None and moment.date() == day


def _moment_key(value: str) -> float:
    """Числовой ключ сортировки: ISO с разными поясами так не сравнишь."""
    moment = _parse_moment(value)
    return moment.timestamp() if moment is not None else 0.0


def _dated_line(hit: MemoryHit, now: datetime | None) -> str:
    moment = _parse_moment(hit.created_at)
    if moment is None:
        label = ""
    else:
        day = _day_label(moment.date(), _as_local(now).date())
        label = f"{day}, {moment.strftime('%H:%M')}: "
    topic = f"«{hit.topic_title}» " if hit.topic_title else ""
    return f"- {label}{topic}{hit.summary}"


def _day_label(day: date, today: date) -> str:
    delta = (today - day).days
    if delta == 0:
        return "Сегодня"
    if delta == 1:
        return "Вчера"
    if delta == 2:
        return "Позавчера"
    return day.strftime("%d.%m.%Y")


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
