"""Эмбеддинги Voyage через ProxyAPI.

Модель превращает текст в вектор, по которому потом ищутся похожие
резюме прошлых разговоров. Документы и поисковые запросы кодируются
разными ``input_type``: так модель лучше различает «что храним» и
«что ищем».

Модуль не знает ни про Chroma, ни про SQLite — только HTTP и векторы.
Ошибки заворачиваются в ``EmbeddingError``: вызывающий решает, стоит
ли повторять запрос или просто ответить без памяти.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

EMBEDDINGS_PATH = "/embeddings"
DEFAULT_MODEL = "mongodb/voyage-4"
DEFAULT_TIMEOUT_SEC = 10.0
DEFAULT_BASE_URL = "https://api.proxyapi.ru/v1"

# Транспорт для тестов: принимает JSON-payload, возвращает разобранный ответ.
Transport = Callable[[dict[str, Any]], dict[str, Any]]


class EmbeddingError(RuntimeError):
    """Не удалось получить вектор. Диалог из-за этого падать не должен."""


class VoyageEmbedder:
    """Клиент embedding-модели с одним методом ``embed``."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        timeout: float = DEFAULT_TIMEOUT_SEC,
        transport: Transport | None = None,
    ) -> None:
        self.model = str(model or DEFAULT_MODEL).strip()
        self.base_url = str(base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = str(api_key or "").strip()
        self.timeout = max(1.0, float(timeout))
        self._transport = transport or self._post_json

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> VoyageEmbedder:
        """Собирает клиента из секции ``memory`` с запасными ключами."""
        cfg = cfg or {}
        section = cfg.get("memory") or {}
        cloud = cfg.get("cloud_llm") or {}
        base_url = str(section.get("base_url") or cloud.get("base_url") or DEFAULT_BASE_URL)
        api_key = str(
            section.get("api_key")
            or cloud.get("api_key")
            or (cfg.get("tts") or {}).get("api_key")
            or ""
        ).strip()
        return cls(
            str(section.get("model") or DEFAULT_MODEL),
            base_url=base_url,
            api_key=api_key,
            timeout=float(section.get("timeout_sec") or DEFAULT_TIMEOUT_SEC),
        )

    def available(self) -> bool:
        """Готов ли клиент к запросу. Сеть не проверяем."""
        return bool(self.model and self.api_key)

    def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        """Возвращает по вектору на каждый текст.

        ``input_type`` — ``document`` для резюме и ``query`` для вопроса
        пользователя. Пустой список на входе даёт пустой список.
        """
        cleaned = [str(text or "").strip() for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise EmbeddingError("пустой текст для эмбеддинга")
        if not self.available():
            raise EmbeddingError("не настроены модель или ключ эмбеддингов")

        payload: dict[str, Any] = {
            "model": self.model,
            "input": cleaned,
            "input_type": input_type,
        }
        try:
            data = self._transport(payload)
        except EmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"сервис эмбеддингов недоступен: {exc}") from exc

        items = data.get("data") or []
        if not isinstance(items, list) or len(items) != len(cleaned):
            raise EmbeddingError("в ответе неверное число векторов")
        ordered = sorted(items, key=lambda item: int(item.get("index", 0)))
        vectors: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise EmbeddingError("в ответе пустой вектор")
            vectors.append([float(value) for value in vector])

        usage = data.get("usage") or {}
        log.info(
            "Voyage: %s текстов за %.0f токенов, размерность %s",
            len(cleaned),
            float(usage.get("total_tokens") or 0),
            len(vectors[0]),
        )
        return vectors

    # ------------------------------------------------------------------- HTTP
    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + EMBEDDINGS_PATH,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise EmbeddingError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EmbeddingError(f"сервер недоступен: {exc}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise EmbeddingError("сервер вернул не JSON") from exc
