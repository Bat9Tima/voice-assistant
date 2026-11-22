"""Обратный поиск по изображению через Yandex Search API v2.

Модуль не снимает экран и не хранит картинки: получает готовый JPEG
в байтах, кодирует его в base64 и отправляет в Яндекс. Ответ
превращается в короткий текст для облачной модели; ссылки голосом
не читаются.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from html import unescape
from typing import Any

from .vision import match_trigger

log = logging.getLogger(__name__)

ENDPOINT = "https://searchapi.api.cloud.yandex.net/v2/image/search_by_image"
MAX_IMAGE_BYTES = 3 * 1024 * 1024
DEFAULT_FAMILY_MODE = "FAMILY_MODE_MODERATE"
DEFAULT_MAX_RESULTS = 5
DEFAULT_MAX_PASSAGE_CHARS = 300
DEFAULT_TIMEOUT = 15.0
DEFAULT_COOLDOWN = 3.0

DEFAULT_TRIGGERS = (
    "поищи по фото",
    "поищи по картинке",
    "поищи в интернете по фото",
    "найди по фото",
    "найди по картинке",
    "что это за фото",
    "что за фото",
    "поиск по фото",
)

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_CITATION_RE = re.compile(r"\[\d+\]")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)")
_URL_RE = re.compile(r"https?://\S+")
_PUNCT_SPACE_RE = re.compile(r"\s+([.,!?;:])")


class ImageSearchError(RuntimeError):
    """Понятная пользователю ошибка обратного поиска."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        request_id: str = "",
    ) -> None:
        super().__init__(message)
        self.status = status
        self.request_id = request_id


@dataclass(frozen=True)
class ImageMatch:
    """Один результат обратного поиска."""

    url: str = ""
    page_url: str = ""
    page_title: str = ""
    passage: str = ""
    host: str = ""
    width: int = 0
    height: int = 0
    format: str = ""


@dataclass(frozen=True)
class ImageSearchResult:
    """Страница результатов обратного поиска."""

    images: tuple[ImageMatch, ...] = ()
    page: int = 0
    cbir_id: str = ""


def match_request(
    text: str,
    triggers: list[str] | tuple[str, ...] = DEFAULT_TRIGGERS,
) -> tuple[bool, str]:
    """Отделяет команду поиска по фото от остального вопроса."""
    trigger, rest = match_trigger(text, triggers)
    return trigger is not None, rest


class YandexImageSearch:
    """Клиент Search by Image из Yandex Search API v2."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        section = (cfg or {}).get("image_search") or {}
        self.enabled = bool(section.get("enabled", True))
        self.api_key = str(section.get("api_key") or "").strip()
        self.folder_id = str(section.get("folder_id") or "").strip()
        self.family_mode = str(section.get("family_mode") or DEFAULT_FAMILY_MODE).strip()
        self.max_results = max(1, int(section.get("max_results") or DEFAULT_MAX_RESULTS))
        self.max_passage_chars = max(
            0, int(section.get("max_passage_chars") or DEFAULT_MAX_PASSAGE_CHARS)
        )
        self.timeout = float(section.get("timeout_sec") or DEFAULT_TIMEOUT)
        raw_cooldown = section.get("cooldown_sec")
        if raw_cooldown is None:
            raw_cooldown = DEFAULT_COOLDOWN
        self.cooldown = max(0.0, float(raw_cooldown))
        self._last_request_at = 0.0
        raw_triggers = section.get("triggers")
        if not isinstance(raw_triggers, list):
            raw_triggers = list(DEFAULT_TRIGGERS)
        self.triggers = [str(item).strip() for item in raw_triggers if str(item).strip()]

    def available(self) -> bool:
        return bool(self.enabled and self.api_key and self.folder_id)

    def match(self, text: str) -> tuple[bool, str]:
        return match_request(text, self.triggers)

    def search(self, image: bytes, *, page: int = 0, site: str = "") -> ImageSearchResult:
        """Ищет похожие картинки по JPEG/PNG в памяти."""
        if not self.enabled:
            raise ImageSearchError("Поиск по фото выключен")
        if not self.api_key:
            raise ImageSearchError("Не задан ключ Yandex Search API")
        if not self.folder_id:
            raise ImageSearchError("Не задан folder_id Yandex Cloud")
        if not image:
            raise ImageSearchError("Пустой кадр")
        if len(image) > MAX_IMAGE_BYTES:
            raise ImageSearchError("Кадр больше 3 МБ")
        if self.cooldown > 0 and self._last_request_at:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.cooldown:
                raise ImageSearchError("Слишком часто: подождите пару секунд")

        payload: dict[str, Any] = {
            "folderId": self.folder_id,
            "data": base64.b64encode(image).decode("ascii"),
            # int64 в protobuf JSON передаётся строкой, как в примере Яндекса.
            "page": str(max(0, int(page))),
            "familyMode": self.family_mode,
        }
        if site:
            payload["site"] = site
        self._last_request_at = time.monotonic()
        return _parse(self._post(payload))

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        request_id = ""
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                request_id = response.headers.get("x-request-id", "")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            request_id = exc.headers.get("x-request-id", "") if exc.headers else ""
            message, code = _error_message(raw)
            log.warning(
                "Обратный поиск: HTTP %s, код %s, ответ %s, request_id %s",
                exc.code,
                code,
                message[:200],
                request_id,
            )
            raise ImageSearchError(
                _user_message(exc.code),
                status=exc.code,
                request_id=request_id,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.warning("Обратный поиск: сеть недоступна: %s", exc)
            raise ImageSearchError("Поиск по фото не ответил") from exc

        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            log.warning("Обратный поиск: некорректный JSON, request_id %s", request_id)
            raise ImageSearchError(
                "Яндекс вернул непонятный ответ",
                request_id=request_id,
            ) from exc
        if not isinstance(data, dict):
            raise ImageSearchError(
                "Яндекс вернул непонятный ответ",
                request_id=request_id,
            )
        images = data.get("images")
        count = len(images) if isinstance(images, list) else 0
        log.info("Обратный поиск: %s изображений, request_id %s", count, request_id)
        return data


def format_results(
    result: ImageSearchResult,
    *,
    limit: int = DEFAULT_MAX_RESULTS,
    max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
) -> str:
    """Короткий текст для модели: заголовки, хосты и сниппеты без URL."""
    lines: list[str] = []
    for index, item in enumerate(result.images[: max(1, limit)], 1):
        title = item.page_title or item.host or "без названия"
        host = f" — {item.host}" if item.host and item.host != title else ""
        lines.append(f"{index}. {title}{host}")
        passage = item.passage.strip()
        if passage:
            if max_passage_chars > 0 and len(passage) > max_passage_chars:
                passage = passage[:max_passage_chars].rstrip() + "…"
            lines.append(f"   {passage}")
    return "\n".join(lines)


def fallback_answer(result: ImageSearchResult) -> str:
    """Ответ без модели, если пересказ не удался."""
    if not result.images:
        return "Ничего похожего не нашёл"
    first = result.images[0]
    title = first.page_title or first.host
    if title:
        return f"Нашёл похожие изображения. Первый результат: {title}."
    return "Нашёл похожие изображения, но не смог их пересказать."


def clean_answer(text: str) -> str:
    """Убирает цитаты и ссылки из ответа перед озвучкой."""
    text = _MD_LINK_RE.sub(r"\1", text or "")
    text = _CITATION_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text)
    text = _PUNCT_SPACE_RE.sub(r"\1", text)
    return text.strip()


def _parse(data: dict[str, Any]) -> ImageSearchResult:
    raw_images = data.get("images")
    if not isinstance(raw_images, list):
        raw_images = []
    images: list[ImageMatch] = []
    for item in raw_images:
        if not isinstance(item, dict):
            continue
        match = ImageMatch(
            url=_text(item.get("url")),
            page_url=_text(item.get("pageUrl") or item.get("page_url")),
            page_title=_clean_html(_text(item.get("pageTitle") or item.get("page_title"))),
            passage=_clean_html(_text(item.get("passage"))),
            host=_clean_html(_text(item.get("host"))),
            width=_int(item.get("width")),
            height=_int(item.get("height")),
            format=_text(item.get("format")),
        )
        if not (match.page_title or match.host or match.passage or match.page_url or match.url):
            continue
        images.append(match)
    return ImageSearchResult(
        images=tuple(images),
        page=_int(data.get("page")),
        cbir_id=_text(data.get("id")),
    )


def _error_message(raw: bytes) -> tuple[str, int | None]:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "", None
    if not isinstance(data, dict):
        return "", None
    code = data.get("code")
    return str(data.get("message") or ""), code if isinstance(code, int) else None


def _user_message(status: int) -> str:
    if status in (401, 403):
        return "Яндекс не принял ключ поиска"
    if status == 429:
        return "Лимит поиска по фото исчерпан"
    if status == 400:
        return "Яндекс не принял картинку"
    if status >= 500:
        return "Яндекс не смог выполнить поиск"
    return "Поиск по фото не сработал"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _clean_html(value: str) -> str:
    value = unescape(value or "")
    value = _TAG_RE.sub(" ", value)
    value = _URL_RE.sub("", value)
    return _SPACE_RE.sub(" ", value).strip()


__all__ = [
    "DEFAULT_COOLDOWN",
    "DEFAULT_FAMILY_MODE",
    "DEFAULT_MAX_PASSAGE_CHARS",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_TIMEOUT",
    "DEFAULT_TRIGGERS",
    "ENDPOINT",
    "MAX_IMAGE_BYTES",
    "ImageMatch",
    "ImageSearchError",
    "ImageSearchResult",
    "YandexImageSearch",
    "clean_answer",
    "fallback_answer",
    "format_results",
    "match_request",
]
