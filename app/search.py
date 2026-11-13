from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


class SearchError(RuntimeError):
    """Поисковый сервис недоступен или не настроен."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    content: str
    score: float = 0.0


def _tokens(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[а-яёa-z0-9]+", (text or "").lower()) if len(token) >= 4
    ]


def _token_match(left: str, right: str) -> bool:
    limit = min(5, len(left), len(right))
    return limit >= 4 and left[:limit] == right[:limit]


def _overlap(text: str, query: str) -> int:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0
    text_tokens = _tokens(text)
    return sum(
        1
        for query_token in query_tokens
        if any(_token_match(query_token, text_token) for text_token in text_tokens)
    )


def _best_excerpt(text: str, query: str, limit: int = 3000) -> str:
    """Берёт окно вокруг первого упоминания темы, а не начало страницы."""
    lowered = (text or "").lower()
    positions = [lowered.find(token) for token in _tokens(query) if lowered.find(token) >= 0]
    if not positions:
        return (text or "")[:limit]
    start = max(0, min(positions) - 600)
    return (text or "")[start : start + limit]


def _strip_title(text: str, title: str) -> str:
    if not title:
        return text or ""
    parts = [title]
    for separator in (" - ", " — ", " – ", " | "):
        if separator in title:
            parts.extend(title.split(separator))
    for part in sorted({part.strip() for part in parts}, key=len, reverse=True):
        if len(part) >= 10:
            text = re.sub(re.escape(part), " ", text or "", flags=re.IGNORECASE)
    return text or ""


class TavilySearch:
    """Поиск через Tavily: сниппеты и очищенный текст страниц."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        section = ((cfg or {}).get("qa") or {}).get("search") or {}
        self.enabled = bool(section.get("enabled", True))
        self.api_key = str(section.get("api_key") or "").strip()
        self.max_results = max(1, min(int(section.get("max_results") or 5), 10))
        self.news_max_results = max(1, min(int(section.get("news_max_results") or 8), 10))
        self.search_depth = str(section.get("search_depth") or "basic")
        self.extract_depth = str(section.get("extract_depth") or "basic")
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
            "max_results": self.news_max_results if news else self.max_results,
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

    def search_article(self, query: str, *, limit: int = 3) -> list[SearchResult]:
        """Ищет разборы по теме новости, а не только ленты СМИ."""
        query = (query or "").strip()
        if not query:
            raise SearchError("пустой поисковый запрос")
        if not self.available():
            raise SearchError("не задан ключ Tavily (qa.search.api_key)")
        payload: dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced",
            "topic": self.topic,
            "max_results": max(1, min(limit, 5)),
            "include_answer": False,
            "include_raw_content": True,
        }
        data = self._post(payload)
        results = data.get("results") or []
        query_token_count = len(_tokens(query))
        if query_token_count <= 1:
            threshold = query_token_count or 1
        else:
            threshold = max(2, min(3, (query_token_count + 1) // 2))
        parsed: list[SearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()
            raw = str(item.get("raw_content") or "").strip()
            if not url:
                continue
            content = _strip_title(content, title)
            if len(content) < 200 and raw:
                excerpt = _best_excerpt(raw, query)
                if _overlap(excerpt, query) > _overlap(content, query):
                    content = excerpt
            if len(content) < 200 or _overlap(content, query) < threshold:
                continue
            try:
                score = float(item.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            parsed.append(
                SearchResult(
                    title or url,
                    url,
                    content,
                    score + _overlap(content, query),
                )
            )
        parsed.sort(key=lambda result: result.score, reverse=True)
        return parsed

    def extract(self, url: str) -> str:
        """Полный очищенный текст статьи через Tavily Extract."""
        url = (url or "").strip()
        if not url:
            raise SearchError("пустой адрес статьи")
        if not self.available():
            raise SearchError("не задан ключ Tavily (qa.search.api_key)")
        payload: dict[str, Any] = {
            "api_key": self.api_key,
            "urls": [url],
            "extract_depth": self.extract_depth,
            "format": "text",
        }
        data = self._post(payload, TAVILY_EXTRACT_URL)
        results = data.get("results") or []
        for item in results:
            if not isinstance(item, dict):
                continue
            raw = str(item.get("raw_content") or "").strip()
            if raw:
                return raw
        raise SearchError("Tavily не вернул текст статьи")

    def _post(self, payload: dict[str, Any], url: str = TAVILY_URL) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
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
