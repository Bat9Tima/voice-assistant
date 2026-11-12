"""Опциональный локальный планировщик свободных команд.

Маленькая instruct-модель через OpenAI-совместимый API (Ollama/LM Studio)
только выбирает действие из белого списка. Ответы на вопросы, новости и
рассуждения идут через облако (``app.cloud_llm``); этот модуль по
умолчанию выключен и нужен лишь как аварийный разбор команд.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .actions import SITE_URLS
from .tools import ToolRegistry, build_default_registry

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

REGISTRY = build_default_registry(SITE_URLS, SITE_KEYS)


def build_system_prompt(registry: ToolRegistry) -> str:
    """Собирает промпт планировщика из описаний реестра инструментов.

    Новый инструмент достаточно добавить в реестр — промпт обновится сам,
    и модель не сможет выбрать действие, которого нет в реестре.
    """
    tools = "\n".join(registry.prompt_lines())
    return f"""Ты — планировщик голосовых команд для браузера.
По русской фразе верни ровно один JSON-объект и ничего больше.
Схема: {{"action": "<действие>", ...аргументы}}.

Действия:
{tools}
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
«стоп» → {{"action": "stop"}}
«останови на ютубе» → {{"action": "stop", "site": "youtube"}}
«перемотай на 30 секунд» → {{"action": "seek", "seconds": 30}}
«перемотай назад на 10 секунд» → {{"action": "seek", "seconds": 10, "direction": "backward"}}
«продолжи в саундклауде» → {{"action": "play", "site": "soundcloud"}}
«какая погода» → {{"action": "none"}}
"""


SYSTEM_PROMPT = build_system_prompt(REGISTRY)


class LocalPlannerError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalPlan:
    action: str
    args: dict[str, Any] = field(default_factory=dict)


class LocalPlanner:
    """Планировщик свободных команд через локальную instruct-модель.

    Модель только выбирает действие из белого списка. Проверка аргументов
    и исполнение остаются на стороне ActionRouter.
    """

    def __init__(
        self, cfg: dict[str, Any], registry: ToolRegistry | None = None
    ) -> None:
        section = (cfg or {}).get("local_planner") or {}
        self.registry = registry or REGISTRY
        self.system_prompt = build_system_prompt(self.registry)
        self.enabled = bool(section.get("enabled", False))
        self.base_url = str(section.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self.model = str(section.get("model") or DEFAULT_MODEL).strip()
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
            log.debug("Локальный планировщик недоступен: %s", exc)
            self._up = False
            self._retry_at = now + 30.0
        return bool(self._up)

    def plan(self, text: str) -> LocalPlan | None:
        text = (text or "").strip()
        if not text or not self.enabled:
            return None
        if not self.model:
            raise LocalPlannerError("не задана модель local_planner.model")
        if not self.available():
            return None

        content = self._chat(text)
        data = _extract_json(content)
        if data is None:
            log.warning("Локальный планировщик вернул не JSON: %s", content[:300])
            return None
        plan = _build_plan(data, self.registry)
        if plan is None:
            log.info("Локальный планировщик: действие отклонено: %s", data)
        else:
            log.info("Локальный планировщик: %s %s", plan.action, plan.args)
        return plan

    # ------------------------------------------------------------------- HTTP
    def _chat(self, text: str) -> str:
        return self._complete(self.system_prompt, text)

    def _complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        model: str | None = None,
    ) -> str:
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": (
                self.temperature if temperature is None else float(temperature)
            ),
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
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
            request_timeout = self.timeout if timeout is None else float(timeout)
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            self._mark_down()
            raise LocalPlannerError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._mark_down()
            raise LocalPlannerError(f"сервер недоступен: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LocalPlannerError("сервер вернул не JSON") from exc
        choices = data.get("choices") or []
        if not choices:
            raise LocalPlannerError("в ответе нет choices")
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise LocalPlannerError("пустой ответ модели")
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


def _build_plan(
    data: dict[str, Any], registry: ToolRegistry = REGISTRY
) -> LocalPlan | None:
    """Превращает ответ модели в план, проверяя его по реестру.

    Реестр сам отбрасывает неизвестные действия, лишние аргументы,
    неверные типы и значения вне диапазона.
    """
    action = str(data.get("action") or "").strip().lower()
    if action == "none":
        return None
    args = registry.validate(action, data)
    if args is None:
        return None
    return LocalPlan(action, args)
