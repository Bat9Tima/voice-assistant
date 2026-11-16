"""Семантическая память по резюме прошлых разговоров.

Разделение источников:

* SQLite (``SessionStore``) — источник правды: сырые сессии, реплики и
  тексты резюме.
* ChromaDB — пересобираемый индекс: векторы резюме и минимум метаданных.
* ``profile.txt`` — постоянный профиль пользователя. Он не смешивается с
  эпизодической памятью: в промпте это отдельные блоки, профиль главнее.

Ошибки Voyage или Chroma не ломают разговор: ``remember`` просто
пропускает обновление индекса, ``recall`` возвращает пустой список.
Вектор — производные данные, его всегда можно пересчитать из резюме.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .embeddings import EmbeddingError, VoyageEmbedder
from .session_store import SessionStore

log = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.35
DEFAULT_MIN_MARGIN = 0.05
DEFAULT_MAX_CHARS = 1200
DEFAULT_COLLECTION = "summaries"
# Насколько ниже порога пускаем лучший результат, если он оторвался
# от второго. Так находятся перефразировки, а «погода» не цепляет мусор.
MARGIN_WINDOW = 0.1
MEMORY_HEADER = (
    "Эпизодическая память: краткие резюме прошлых разговоров, не текущего.\n"
    "Это не факты из профиля. Если воспоминание противоречит профилю, "
    "верь профилю.\n"
    "Используй воспоминания, только если они помогают ответить."
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

    def search(self, embedding: list[float], limit: int) -> list[VectorHit]: ...

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

    def search(self, embedding: list[float], limit: int) -> list[VectorHit]:
        limit = max(1, int(limit))
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            include=["metadatas", "distances"],
        )
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
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.vectors = vectors
        self.top_k = max(1, int(top_k))
        self.min_score = float(min_score)
        self.min_margin = max(0.0, float(min_margin))
        self.max_chars = max(0, int(max_chars))

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        store: SessionStore,
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
                {
                    "embedding_model": self.embedder.model,
                    "dim": len(vector),
                    "created_at": _now_iso(),
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Память не обновлена для сессии %s: %s", session_id, exc)
            return False
        log.info("Память: резюме сессии %s проиндексировано", session_id)
        return True

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
            hits = self.vectors.search(vector, self.top_k * 2)
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

    def block(self, query: str) -> str:
        """Готовый блок для системного промпта или пустая строка."""
        hits = self.recall(query)
        if not hits:
            return ""
        lines = [MEMORY_HEADER]
        total = len(MEMORY_HEADER)
        for hit in hits:
            moment = _short_date(hit.created_at)
            line = f"- {moment}{hit.summary}" if moment else f"- {hit.summary}"
            if self.max_chars and total + len(line) > self.max_chars:
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines) if len(lines) > 1 else ""

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


def _short_date(value: str) -> str:
    """«16.11.2026 01:45» из ISO-строки; мусор даёт пустую строку."""
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return ""
    return moment.strftime("%d.%m.%Y %H:%M: ")
