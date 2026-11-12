"""Облачная LLM для ответов на вопросы, новостей и рассуждений.

Единственный клиент с методом ``answer()``. Работает через единый
OpenAI-совместимый API ProxyAPI. Если основная модель недоступна или
вернула пустой ответ, пробуем запасные модели из конфига. Локальной
модели в этом пути нет.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from .prompts import ANSWER_PROMPTS, ANSWER_SYSTEM, RAG_PROMPTS, RAG_SYSTEM

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.proxyapi.ru/v1"
DEFAULT_MODEL = "anthropic/claude-haiku-5-5"
DEFAULT_FALLBACK_MODELS = ("xiaomi/mimo-v2.6-flash",)

# Рассуждающие модели тратят часть лимита на скрытые размышления.
# Ниже этого порога ответы обрезаются или приходят пустыми.
MIN_COMPLETION_TOKENS = 2000


class LLMError(RuntimeError):
    """Ошибка облачной модели. retryable — имеет смысл повторить запрос."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class CloudLLM:
    """Клиент ответов через облако с запасными моделями."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        cfg = cfg or {}
        section = cfg.get("cloud_llm") or {}
        self.enabled = bool(section.get("enabled", True))
        self.base_url = str(section.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self.model = str(section.get("model") or DEFAULT_MODEL).strip()
        self.article_model = str(section.get("article_model") or "").strip()
        self.api_key = str(section.get("api_key") or "").strip()
        if not self.api_key:
            # Оба сервиса работают через ProxyAPI, поэтому ключ TTS подходит
            # и для ответов. Явный cloud_llm.api_key всё равно важнее.
            self.api_key = str((cfg.get("tts") or {}).get("api_key") or "").strip()
        fallbacks = section.get("fallback_models")
        if fallbacks is None:
            fallbacks = list(DEFAULT_FALLBACK_MODELS)
        self.fallback_models = [
            str(item).strip()
            for item in fallbacks
            if str(item).strip() and str(item).strip() != self.model
        ]
        self.reasoning_effort = str(
            section.get("reasoning_effort") or "minimal"
        ).strip()
        self.max_completion_tokens = max(
            int(section.get("max_completion_tokens") or MIN_COMPLETION_TOKENS),
            MIN_COMPLETION_TOKENS,
        )
        self.temperature = float(section.get("temperature", 0.2))
        self.timeout = float(section.get("timeout_sec") or 90)
        self.retries = max(int(section.get("retries") or 2), 0)

    # ------------------------------------------------------------------ public
    def available(self) -> bool:
        """Готов ли клиент к запросу. Сеть не проверяем, чтобы не ждать зря."""
        return bool(self.enabled and self.model and self.api_key)

    def answer(
        self,
        question: str,
        context: str = "",
        *,
        style: str = "normal",
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        model: str | None = None,
    ) -> str:
        """Отвечает на вопрос: по источникам (RAG) или из своих знаний."""
        question = (question or "").strip()
        if not question:
            raise LLMError("пустой вопрос")
        if not self.enabled:
            raise LLMError("облачная LLM выключена")
        if not self.model:
            raise LLMError("не задана модель cloud_llm.model")
        if not self.api_key:
            raise LLMError("не задан ключ cloud_llm.api_key")

        style = (style or "normal").strip().lower()
        context = (context or "").strip()
        if context:
            system = RAG_PROMPTS.get(style, RAG_SYSTEM)
            user = f"Вопрос: {question}\n\nИсточники:\n{context}"
        else:
            system = ANSWER_PROMPTS.get(style, ANSWER_SYSTEM)
            user = question

        limit = max(int(max_tokens or 0), self.max_completion_tokens)
        temperature = self.temperature if temperature is None else float(temperature)
        timeout = self.timeout if timeout is None else float(timeout)
        preferred = model or (self.article_model if style == "article" else "")
        candidates = [preferred or self.model]
        candidates.extend(
            item for item in self.fallback_models if item not in candidates
        )

        errors: list[str] = []
        for candidate in candidates:
            for attempt in range(1, self.retries + 2):
                try:
                    return self._complete(
                        candidate,
                        system,
                        user,
                        max_tokens=limit,
                        temperature=temperature,
                        timeout=timeout,
                    )
                except LLMError as exc:
                    errors.append(f"{candidate}: {exc}")
                    log.warning(
                        "CloudLLM: %s, попытка %s: %s",
                        candidate,
                        attempt,
                        exc,
                    )
                    if not exc.retryable or attempt > self.retries:
                        break
                    time.sleep(min(0.5 * attempt, 2.0))

        detail = "; ".join(errors[-2:]) if errors else "нет ответа"
        raise LLMError(f"модель не ответила: {detail}")

    # ------------------------------------------------------------------- HTTP
    def _complete(
        self,
        model: str,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            retryable = exc.code == 429 or 500 <= exc.code < 600
            raise LLMError(f"HTTP {exc.code}: {detail}", retryable=retryable) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # Повторять тот же запрос после таймаута бессмысленно: запасная
            # модель попробует заново, а пользователь не ждёт два раза.
            raise LLMError(f"сервер недоступен: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError("сервер вернул не JSON") from exc

        choices = data.get("choices") or []
        if not choices:
            raise LLMError("в ответе нет choices")
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise LLMError("пустой ответ модели")

        elapsed = time.monotonic() - started
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        log.info(
            "CloudLLM: %s за %.2f с, токены %s→%s (reasoning %s)",
            data.get("model") or model,
            elapsed,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            details.get("reasoning_tokens"),
        )
        return content
