"""Сборка контекста одного хода диалога.

Одно место, где решается, что именно уходит в модель: порядок блоков
системного промпта, окно истории и общий бюджет. Блоки идут по важности:
правила и дата/время обязательны, дальше профиль пользователя, факты
Луны и память. При переполнении бюджета режется самый нижний блок, а
история — с самых старых реплик. Текущая реплика и правила не режутся
никогда.

Бюджет считается в токенах приблизительно: точного токенизатора под
конкретную модель здесь нет, а для русского текста хватает оценки по
символам. Ошибка оценки в обе стороны покрывается запасом бюджета.

Сюда же позже встанет умный отбор контекста: заменить ``ContextBuilder``
на движок с тем же входом и выходом можно, не трогая память, сессии и
SQLite. Пока это сознательно детерминированные правила без лишнего
LLM-вызова в голосовом цикле.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

log = logging.getLogger(__name__)

DEFAULT_BUDGET_TOKENS = 6000
DEFAULT_CHARS_PER_TOKEN = 2.5
# Блок короче этого не оставляем: обрубок промпта только мешает.
MIN_PART_CHARS = 80
ELLIPSIS = "…"


@dataclass(frozen=True)
class ContextPart:
    """Кусок системного промпта. ``required`` не режется никогда."""

    name: str
    text: str
    required: bool = False
    # Факты личности обрезаются с конца: свежие важнее старых.
    keep_tail: bool = False


@dataclass
class ContextStats:
    """Что вошло в контекст: для логов, тестов и будущих метрик."""

    tokens: int = 0
    parts: dict[str, int] = field(default_factory=dict)
    history_messages: int = 0
    history_dropped: int = 0
    trimmed: list[str] = field(default_factory=list)


@dataclass
class BuiltContext:
    system: str
    messages: list[dict[str, str]]
    stats: ContextStats


class ContextBuilder:
    """Собирает системный промпт и сообщения под общий бюджет токенов."""

    def __init__(
        self,
        max_tokens: int = DEFAULT_BUDGET_TOKENS,
        chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    ) -> None:
        # Ноль отключает бюджет: полезно для отладки и тестов.
        self.max_tokens = max(0, int(max_tokens))
        self.chars_per_token = max(1.0, float(chars_per_token))

    def estimate(self, text: str) -> int:
        """Оценка токенов по символам: точный токенизатор тут не нужен."""
        return math.ceil(len(text) / self.chars_per_token) if text else 0

    def build(
        self,
        *,
        parts: Iterable[ContextPart],
        history: Iterable[dict[str, str]],
        user_text: str,
        memory: str = "",
        tail: Iterable[ContextPart] = (),
    ) -> BuiltContext:
        """Собирает контекст и при переполнении режет его по приоритетам.

        ``tail`` идёт после памяти: свежая явная просьба пользователя
        должна быть последней, иначе старый факт памяти спорит с ней.
        """
        system_parts = [part for part in parts if part.text.strip()]
        if memory.strip():
            # Память — последняя из обычных блоков и первая под нож.
            system_parts.append(ContextPart("memory", memory.strip()))
        system_parts.extend(part for part in tail if part.text.strip())
        messages = [item for item in history if str(item.get("content") or "").strip()]

        total = sum(self.estimate(part.text) for part in system_parts)
        total += sum(self.estimate(str(item.get("content") or "")) for item in messages)
        total += self.estimate(user_text)

        stats = ContextStats(tokens=total)
        trimmed = False
        if self.max_tokens and total > self.max_tokens:
            system_parts, messages = self._trim(system_parts, messages, user_text, stats)
            total = sum(self.estimate(part.text) for part in system_parts)
            total += sum(self.estimate(str(item.get("content") or "")) for item in messages)
            total += self.estimate(user_text)
            trimmed = bool(stats.trimmed) or bool(stats.history_dropped)

        stats.tokens = total
        stats.parts = {part.name: len(part.text) for part in system_parts}
        stats.history_messages = len(messages)
        if trimmed:
            log.info(
                "Контекст: обрезано %s, выброшено реплик %s, итого ~%s токенов",
                ", ".join(stats.trimmed) or "—",
                stats.history_dropped,
                stats.tokens,
            )
        else:
            log.debug(
                "Контекст: ~%s токенов, блоки %s, история %s",
                stats.tokens,
                stats.parts,
                stats.history_messages,
            )
        return BuiltContext(
            system="\n\n".join(part.text for part in system_parts),
            messages=[*messages, {"role": "user", "content": user_text}],
            stats=stats,
        )

    # --------------------------------------------------------------- private
    def _trim(
        self,
        system_parts: list[ContextPart],
        messages: list[dict[str, str]],
        user_text: str,
        stats: ContextStats,
    ) -> tuple[list[ContextPart], list[dict[str, str]]]:
        parts = list(system_parts)
        history = list(messages)

        # Необязательные блоки с конца: память, личность, профиль.
        index = len(parts) - 1
        while index >= 0:
            overflow = self._overflow(parts, history, user_text)
            if overflow <= 0:
                break
            part = parts[index]
            if part.required:
                index -= 1
                continue
            allowed = len(part.text) - int(round(overflow * self.chars_per_token))
            if allowed < MIN_PART_CHARS:
                parts.pop(index)
                stats.trimmed.append(part.name)
            else:
                parts[index] = replace(
                    part,
                    text=self._clip(part.text, allowed, tail=part.keep_tail),
                )
                stats.trimmed.append(part.name)
            index -= 1

        # История уходит с самых старых реплик.
        while history and self._overflow(parts, history, user_text) > 0:
            history.pop(0)
            stats.history_dropped += 1
        if stats.history_dropped:
            stats.trimmed.append("history")

        if self._overflow(parts, history, user_text) > 0:
            log.warning(
                "Контекст: обязательные блоки не влезают в бюджет %s токенов",
                self.max_tokens,
            )
        return parts, history

    def _overflow(
        self,
        parts: list[ContextPart],
        history: list[dict[str, str]],
        user_text: str,
    ) -> int:
        total = sum(self.estimate(part.text) for part in parts)
        total += sum(self.estimate(str(item.get("content") or "")) for item in history)
        total += self.estimate(user_text)
        return total - self.max_tokens

    def _clip(self, text: str, limit: int, *, tail: bool) -> str:
        """Обрезает блок до ``limit`` символов, оставляя начало или конец."""
        if limit >= len(text):
            return text
        if limit <= 1:
            return ELLIPSIS
        if tail:
            return ELLIPSIS + text[-(limit - 1) :]
        return text[: limit - 1].rstrip() + ELLIPSIS
