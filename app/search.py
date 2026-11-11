from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"


class SearchError(RuntimeError):
    """Поисковый сервис недоступен или не настроен."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    content: str
    score: float = 0.0


class TavilySearch:
    """Поиск через Tavily: сниппеты и очищенный текст страниц."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        section = ((cfg or {}).get("qa") or {}).get("search") or {}
        self.enabled = bool(section.get("enabled", True))
        self.api_key = str(section.get("api_key") or "").strip()
        self.max_results = max(1, min(int(section.get("max_results") or 5), 10))
        self.search_depth = str(section.get("search_depth") or "basic")
        self.topic = str(section.get("topic") or "general")
        self.news_days = max(1, min(int(section.get("news_days") or 7), 30))
        self.timeout = float(section.get("timeout_sec") or 15)
        self.news_domains = tuple(
            str(domain).strip()
            for domain in section.get("news_domains") or []
            if str(domain).strip()
        )

    def available(self) -> bool:
        return self.enabled and bool(self.api_key)

    def search(self, query: str, *, news: bool = False) -> list[SearchResult]:
        query = (query or "").strip()
        if not query:
            raise SearchError("пустой поисковый запрос")
        if not self.available():
            raise SearchError("не задан ключ Tavily (qa.search.api_key)")

        payload: dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": self.search_depth,
            "topic": "news" if news else self.topic,
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        if news:
            payload["days"] = self.news_days
            if self.news_domains:
                payload["include_domains"] = list(self.news_domains)

        data = self._post(payload)
        results = data.get("results") or []
        parsed: list[SearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()
            if not url or not content:
                continue
            try:
                score = float(item.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            parsed.append(SearchResult(title or url, url, content, score))
        return parsed

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            TAVILY_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise SearchError(f"Tavily HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SearchError(f"Tavily недоступен: {exc}") from exc
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise SearchError("Tavily вернул не JSON") from exc
        if not isinstance(data, dict):
            raise SearchError("Tavily вернул неверный ответ")
        return data
