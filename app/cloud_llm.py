"""Облачная LLM для ответов на вопросы, новостей и рассуждений.

Единственный клиент с методом ``answer()``. Работает через единый
OpenAI-совместимый API ProxyAPI. Если основная модель недоступна или
вернула пустой ответ, пробуем запасные модели из конфига. Локальной
модели в этом пути нет.

У моделей разные ограничения на параметры запроса. Такие отличия задаются
в ``cloud_llm.model_params`` по ID модели, а не кодом: значение
переопределяет поле запроса, ``null`` убирает его.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from http.client import HTTPException
from typing import Any

from .datetime_answer import datetime_context
from .prompts import (
    ANSWER_PROMPTS,
    ANSWER_SYSTEM,
    FOLLOWUP_SUFFIX,
    RAG_PROMPTS,
    RAG_SYSTEM,
)

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.proxyapi.ru/v1"
DEFAULT_MODEL = "x-ai/grok-4.3"
DEFAULT_FALLBACK_MODELS = ("xiaomi/mimo-v2.6-flash",)

# Рассуждающие модели тратят часть лимита на скрытые размышления.
# Ниже этого порога ответы обрезаются или приходят пустыми.
MIN_COMPLETION_TOKENS = 2000


class LLMError(RuntimeError):
    """Ошибка облачной модели. retryable — имеет смысл повторить запрос."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def _prepare_messages(
    messages: list[dict[str, str]],
    system: str,
) -> list[dict[str, str]]:
    """Чистит сообщения и ставит системный промпт первым."""
    cleaned: list[dict[str, str]] = []
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        cleaned.append({"role": role, "content": content.strip()})
    system = (system or "").strip()
    if system:
        cleaned.insert(0, {"role": "system", "content": system})
    return cleaned


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
        self.reasoning_effort = str(section.get("reasoning_effort") or "minimal").strip()
        # Отличия конкретных моделей. Ключ — ID модели, значение — поля
        # запроса; null убирает поле. Например, если модель не принимает
        # temperature: {"provider/model": {"temperature": null}}.
        raw_params = section.get("model_params") or {}
        self.model_params: dict[str, dict[str, Any]] = {
            str(name): dict(values)
            for name, values in raw_params.items()
            if isinstance(values, dict)
        }
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
        history: str = "",
    ) -> str:
        """Отвечает на вопрос: по источникам (RAG) или из своих знаний.

        ``history`` — предыдущая пара «вопрос — ответ» для уточняющих
        вопросов вроде «а почему?» или «а подробнее?».
        """
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
        history = (history or "").strip()
        if context:
            system = RAG_PROMPTS.get(style, RAG_SYSTEM)
            user = f"Вопрос: {question}\n\nИсточники:\n{context}"
            if history:
                user = f"{history}\n\n{user}"
        elif history:
            system = (
                ANSWER_PROMPTS.get(style, ANSWER_SYSTEM)
                + FOLLOWUP_SUFFIX
                + "\n"
                + datetime_context()
            )
            user = f"{history}\n\nУточняющий вопрос: {question}"
        else:
            system = ANSWER_PROMPTS.get(style, ANSWER_SYSTEM) + "\n" + datetime_context()
            user = question

        limit = max(int(max_tokens or 0), self.max_completion_tokens)
        temperature = self.temperature if temperature is None else float(temperature)
        timeout = self.timeout if timeout is None else float(timeout)
        preferred = model or (self.article_model if style == "article" else "")
        return self._with_fallback(
            lambda candidate: self._complete(
                candidate,
                system,
                user,
                max_tokens=limit,
                temperature=temperature,
                timeout=timeout,
            ),
            preferred=preferred,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        model: str | None = None,
    ) -> str:
        """Ответ на список сообщений в формате OpenAI."""
        cleaned = _prepare_messages(messages, system)
        if not cleaned:
            raise LLMError("пустой диалог")
        self._ensure_ready()

        limit = max(int(max_tokens or 0), self.max_completion_tokens)
        temperature = self.temperature if temperature is None else float(temperature)
        timeout = self.timeout if timeout is None else float(timeout)
        return self._with_fallback(
            lambda candidate: self._post(
                candidate,
                cleaned,
                max_tokens=limit,
                temperature=temperature,
                timeout=timeout,
            ),
            preferred=model or "",
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        """Отдаёт ответ кусками по мере генерации (SSE).

        Запасная модель включается, только пока не пришёл первый кусок:
        после начала речи переключаться уже нельзя.
        """
        cleaned = _prepare_messages(messages, system)
        if not cleaned:
            raise LLMError("пустой диалог")
        self._ensure_ready()

        limit = max(int(max_tokens or 0), self.max_completion_tokens)
        temperature = self.temperature if temperature is None else float(temperature)
        timeout = self.timeout if timeout is None else float(timeout)
        return self._stream_with_fallback(
            cleaned,
            max_tokens=limit,
            temperature=temperature,
            timeout=timeout,
            preferred=model or "",
        )

    def _ensure_ready(self) -> None:
        """Проверяет, что клиент вообще может идти в сеть."""
        if not self.enabled:
            raise LLMError("облачная LLM выключена")
        if not self.model:
            raise LLMError("не задана модель cloud_llm.model")
        if not self.api_key:
            raise LLMError("не задан ключ cloud_llm.api_key")

    def _with_fallback(self, send: Callable[[str], str], *, preferred: str = "") -> str:
        """Повторяет запрос и перебирает запасные модели."""
        candidates = [preferred or self.model]
        candidates.extend(item for item in self.fallback_models if item not in candidates)

        errors: list[str] = []
        for candidate in candidates:
            for attempt in range(1, self.retries + 2):
                try:
                    return send(candidate)
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

    def _stream_with_fallback(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
        preferred: str = "",
    ) -> Iterator[str]:
        """Поток с перебором моделей, пока не пришёл первый кусок."""
        candidates = [preferred or self.model]
        candidates.extend(item for item in self.fallback_models if item not in candidates)

        errors: list[str] = []
        for candidate in candidates:
            for attempt in range(1, self.retries + 2):
                started = time.monotonic()
                first = 0.0  # время до первого куска
                yielded = False
                try:
                    for piece in self._post_stream(
                        candidate,
                        messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=timeout,
                    ):
                        if not yielded:
                            first = time.monotonic() - started
                        yielded = True
                        yield piece
                    if not yielded:
                        raise LLMError("пустой ответ модели")
                    log.info(
                        "CloudLLM: поток %s за %.2f с (первый кусок %.2f с)",
                        candidate,
                        time.monotonic() - started,
                        first,
                    )
                    return
                except LLMError as exc:
                    errors.append(f"{candidate}: {exc}")
                    if yielded:
                        # Часть ответа уже прозвучала: тихо начинать заново
                        # нельзя, иначе пользователь услышит два начала.
                        log.warning(
                            "CloudLLM: поток %s оборвался после начала ответа: %s",
                            candidate,
                            exc,
                        )
                        raise LLMError(f"поток прерван: {exc}") from exc
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
    def _payload(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> dict[str, Any]:
        """Собирает тело запроса с учётом ограничений конкретной модели."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": stream,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        for key, value in self.model_params.get(model, {}).items():
            if value is None:
                payload.pop(key, None)
            else:
                payload[key] = value
        return payload

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
        return self._post(
            model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

    def _post(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> str:
        payload: dict[str, Any] = self._payload(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
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
                str(part.get("text") or "") for part in content if isinstance(part, dict)
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

    def _post_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> Iterator[str]:
        """Читает SSE-поток и отдаёт только текстовые куски ответа."""
        payload: dict[str, Any] = self._payload(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self.api_key}",
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        # Служебные и битые строки не должны рвать ответ.
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0] or {}).get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        yield content
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            retryable = exc.code == 429 or 500 <= exc.code < 600
            raise LLMError(f"HTTP {exc.code}: {detail}", retryable=retryable) from exc
        except (urllib.error.URLError, TimeoutError, OSError, HTTPException) as exc:
            raise LLMError(f"сервер недоступен: {exc}") from exc
