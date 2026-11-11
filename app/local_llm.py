from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .actions import SITE_URLS

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen2.5:3b-instruct"

# Ключи сайтов, которые модели разрешено использовать в open_site.
# Остальные значения отбрасываются ещё до вызова браузера.
SITE_KEYS = [
    "ютуб",
    "гугл",
    "яндекс",
    "вк",
    "телеграм",
    "почта",
    "календарь",
    "диск",
    "гитхаб",
    "чатгпт",
    "википедия",
    "кинопоиск",
    "озон",
    "вайлдберриз",
    "авито",
    "карты",
    "переводчик",
    "музыка",
    "хабр",
    "саундклауд",
]

SYSTEM_PROMPT = f"""Ты — планировщик голосовых команд для браузера.
По русской фразе верни ровно один JSON-объект и ничего больше.
Схема: {{"action": "<действие>", ...аргументы}}.

Действия:
- open_site: {{"site": "<ключ>"}} — открыть сайт. Ключи: {", ".join(SITE_KEYS)}.
- goto: {{"url": "https://..."}} — открыть точный http(s)-адрес.
- youtube: {{"query": "...", "index": 1}} — найти и включить видео.
- soundcloud: {{"query": "...", "index": 1}} — найти и включить трек.
- soundcloud_likes: {{"query": "...", "play": true}} — включить трек из лайкнутых SoundCloud. Без query — случайный лайкнутый; play=false — просто открыть лайкнутые.
- google: {{"query": "..."}} — поиск в Google.
- play, pause, next, prev — управление медиа.
- newtab, closetab — вкладки.
- tab: {{"index": 2}} — переключить вкладку.
- back, forward, reload — навигация.
- scroll: {{"direction": "up"|"down"}} — прокрутка.
- volume: {{"level": 0..100}} — системная громкость.
- volume_up, volume_down, mute, unmute — громкость.
- status — что открыто в браузере.
- none — это не команда или непонятно.

Правила:
1. Не придумывай сайты и URL. Для open_site используй только ключи из списка.
2. Если фраза — вопрос, просьба совета или болтовня, верни {{"action": "none"}}.
3. Если фраза про лайки, избранное или любимые треки SoundCloud — используй soundcloud_likes, а не soundcloud.
4. Если не уверен в действии, верни {{"action": "none"}}.
5. Никакого текста, markdown и пояснений вокруг JSON.

Примеры:
«вруби на ютубе lo-fi» → {{"action": "youtube", "query": "lo-fi"}}
«зайди на гитхаб» → {{"action": "open_site", "site": "гитхаб"}}
«найди в гугле рецепт борща» → {{"action": "google", "query": "рецепт борща"}}
«включи в лайкнутых треках отпусти меня» → {{"action": "soundcloud_likes", "query": "отпусти меня"}}
«вруби что-нибудь из лайкнутых» → {{"action": "soundcloud_likes"}}
«открой мои лайки на саундклауде» → {{"action": "soundcloud_likes", "play": false}}
«сделай погромче» → {{"action": "volume_up"}}
«поставь громкость 30» → {{"action": "volume", "level": 30}}
«переключи на вторую вкладку» → {{"action": "tab", "index": 2}}
«какая погода» → {{"action": "none"}}
"""

# action -> (разрешённые аргументы, обязательные аргументы)
ACTIONS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "open_site": (frozenset({"site"}), frozenset({"site"})),
    "goto": (frozenset({"url"}), frozenset({"url"})),
    "youtube": (frozenset({"query", "index"}), frozenset({"query"})),
    "soundcloud": (frozenset({"query", "index"}), frozenset({"query"})),
    "soundcloud_likes": (frozenset({"query", "play"}), frozenset()),
    "google": (frozenset({"query"}), frozenset({"query"})),
    "play": (frozenset(), frozenset()),
    "pause": (frozenset(), frozenset()),
    "next": (frozenset(), frozenset()),
    "prev": (frozenset(), frozenset()),
    "newtab": (frozenset(), frozenset()),
    "closetab": (frozenset(), frozenset()),
    "tab": (frozenset({"index"}), frozenset({"index"})),
    "back": (frozenset(), frozenset()),
    "forward": (frozenset(), frozenset()),
    "reload": (frozenset(), frozenset()),
    "scroll": (frozenset({"direction"}), frozenset({"direction"})),
    "volume": (frozenset({"level"}), frozenset({"level"})),
    "volume_up": (frozenset(), frozenset()),
    "volume_down": (frozenset(), frozenset()),
    "mute": (frozenset(), frozenset()),
    "unmute": (frozenset(), frozenset()),
    "status": (frozenset(), frozenset()),
}

_QUERY_MAX = 200
_URL_MAX = 2048


class LocalLLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalPlan:
    action: str
    args: dict[str, Any] = field(default_factory=dict)


class LocalLLM:
    """Планировщик свободных команд через локальную instruct-модель.

    Модель только выбирает действие из белого списка. Проверка аргументов
    и исполнение остаются на стороне ActionRouter.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        section = (cfg or {}).get("llm") or {}
        self.enabled = bool(section.get("enabled", False))
        self.base_url = str(section.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self.model = str(section.get("model") or "").strip()
        self.api_key = str(section.get("api_key") or "").strip()
        self.timeout = float(section.get("timeout_sec") or 20)
        self.temperature = float(section.get("temperature") or 0.0)
        self.max_tokens = int(section.get("max_tokens") or 256)
        keep_alive = section.get("keep_alive", "30m")
        self.keep_alive = keep_alive.strip() if isinstance(keep_alive, str) else keep_alive
        self._up: bool | None = None
        self._checked_at = 0.0
        self._retry_at = 0.0

    # ------------------------------------------------------------------ public
    def available(self) -> bool:
        if not self.enabled or not self.model:
            return False
        now = time.monotonic()
        if now < self._retry_at:
            return False
        if self._up is not None and now - self._checked_at < 30.0:
            return self._up
        self._checked_at = now
        try:
            with urllib.request.urlopen(
                self.base_url + "/models",
                timeout=min(self.timeout, 2.0),
            ) as response:
                status = int(getattr(response, "status", 200))
            self._up = 200 <= status < 300
        except Exception as exc:  # noqa: BLE001
            log.debug("Локальная LLM недоступна: %s", exc)
            self._up = False
            self._retry_at = now + 30.0
        return bool(self._up)

    def plan(self, text: str) -> LocalPlan | None:
        text = (text or "").strip()
        if not text or not self.enabled:
            return None
        if not self.model:
            raise LocalLLMError("не задана модель llm.model")
        if not self.available():
            return None

        content = self._chat(text)
        data = _extract_json(content)
        if data is None:
            log.warning("Локальная LLM вернула не JSON: %s", content[:300])
            return None
        plan = _build_plan(data)
        if plan is None:
            log.info("Локальная LLM: действие отклонено: %s", data)
        else:
            log.info("Локальная LLM: %s %s", plan.action, plan.args)
        return plan

    # ------------------------------------------------------------------- HTTP
    def _chat(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.keep_alive not in (None, ""):
            payload["keep_alive"] = self.keep_alive
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            self._mark_down()
            raise LocalLLMError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._mark_down()
            raise LocalLLMError(f"сервер недоступен: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LocalLLMError("сервер вернул не JSON") from exc
        choices = data.get("choices") or []
        if not choices:
            raise LocalLLMError("в ответе нет choices")
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise LocalLLMError("пустой ответ модели")
        return content

    def _mark_down(self) -> None:
        self._up = False
        self._checked_at = time.monotonic()
        self._retry_at = self._checked_at + 30.0


# --------------------------------------------------------------------- helpers
def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _build_plan(data: dict[str, Any]) -> LocalPlan | None:
    action = str(data.get("action") or "").strip().lower()
    if action == "none" or action not in ACTIONS:
        return None
    allowed, required = ACTIONS[action]
    if not required <= set(data):
        return None
    args: dict[str, Any] = {key: data[key] for key in allowed if key in data}

    if action == "open_site":
        site = _clean_text(args.get("site"), lower=True)
        if site not in SITE_URLS:
            return None
        args["site"] = site
        return LocalPlan(action, args)

    if action == "goto":
        url = _clean_url(args.get("url"))
        if url is None:
            return None
        args["url"] = url
        return LocalPlan(action, args)

    if action in {"youtube", "soundcloud", "google"}:
        query = _clean_text(args.get("query"))
        if not query or len(query) > _QUERY_MAX:
            return None
        args["query"] = query
        if action in {"youtube", "soundcloud"}:
            index = _as_int(args.get("index"), 1)
            args["index"] = max(1, min(20, index))
        return LocalPlan(action, args)

    if action == "soundcloud_likes":
        query = _clean_text(args.get("query"))
        if len(query) > _QUERY_MAX:
            return None
        args["query"] = query
        args["play"] = _as_bool(args.get("play"), True)
        return LocalPlan(action, args)

    if action == "tab":
        index = _as_int(args.get("index"), 0)
        if index < 1 or index > 99:
            return None
        args["index"] = index
        return LocalPlan(action, args)

    if action == "scroll":
        direction = _clean_text(args.get("direction"), lower=True)
        if direction not in {"up", "down"}:
            return None
        args["direction"] = direction
        return LocalPlan(action, args)

    if action == "volume":
        level = _as_int(args.get("level"), -1)
        if level < 0 or level > 100:
            return None
        args["level"] = level
        return LocalPlan(action, args)

    return LocalPlan(action, {})


def _clean_text(value: Any, lower: bool = False) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text.lower() if lower else text


def _clean_url(value: Any) -> str | None:
    url = str(value or "").strip()
    if not url or len(url) > _URL_MAX or re.search(r"\s", url):
        return None
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return url


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "да"}:
            return True
        if text in {"false", "0", "no", "нет"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
