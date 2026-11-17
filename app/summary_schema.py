"""Структурированное резюме завершённой сессии.

Модель возвращает JSON с полями ``facts``, ``decisions``, ``unfinished``,
``topics`` и ``summary``. У каждого пункта первых трёх списков есть
``evidence`` — дословная цитата пользователя. Пункт без подтверждения
отбрасывается: память не должна пополняться догадками модели.

В SQLite структура хранится целиком в ``summaries.data``, а в
``summaries.summary`` лежит обычный текст. Текст нужен для эмбеддинга,
поиска и совместимости со старыми записями, у которых поля ещё не было.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Слишком короткая цитата («да», «угу») ничего не подтверждает.
MIN_EVIDENCE_CHARS = 5
MAX_ITEMS = 10
MAX_TOPICS = 7
_ITEM_KEYS = ("facts", "decisions", "unfinished")
_SPACES = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]")


class SummaryFormatError(ValueError):
    """Ответ модели не удалось разобрать как JSON нужной формы."""


@dataclass(frozen=True)
class SummaryItem:
    """Один пункт памяти: формулировка и цитата, которая её подтверждает."""

    text: str
    evidence: str


@dataclass
class StructuredSummary:
    """Разобранное резюме сессии. Пустые списки — нормальный результат."""

    facts: list[SummaryItem] = field(default_factory=list)
    decisions: list[SummaryItem] = field(default_factory=list)
    unfinished: list[SummaryItem] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Форма для ``summaries.data``: только простые типы."""
        return {
            "facts": [{"text": item.text, "evidence": item.evidence} for item in self.facts],
            "decisions": [
                {"text": item.text, "evidence": item.evidence} for item in self.decisions
            ],
            "unfinished": [
                {"text": item.text, "evidence": item.evidence} for item in self.unfinished
            ],
            "topics": list(self.topics),
            "summary": self.summary,
        }


def parse_summary(answer: str, user_text: str) -> StructuredSummary:
    """Разбирает ответ модели и оставляет только подтверждённые пункты.

    ``user_text`` — реплики пользователя, склеенные в одну строку. Пункт
    попадает в результат, только если его ``evidence`` встречается там
    дословно (регистр, пунктуация и лишние пробелы не важны).
    """
    payload = _extract_json(answer)
    if payload is None:
        raise SummaryFormatError("ответ модели не содержит JSON-объекта")
    confirmed = _normalize(user_text)
    summary = StructuredSummary(
        topics=_topics(payload.get("topics")),
        summary=_text(payload.get("summary")),
    )
    for key in _ITEM_KEYS:
        items = _items(payload.get(key), confirmed)
        setattr(summary, key, items)
    return summary


def render_summary(summary: StructuredSummary) -> str:
    """Обычный текст для эмбеддинга, памяти и чтения человеком.

    Факты идут первыми: при обрезке длинного резюме в дайджесте важное
    остаётся в начале. Цитаты в текст не попадают — они нужны для
    проверки и лежат в JSON.
    """
    lines: list[str] = []
    blocks = (
        ("Факты о пользователе:", summary.facts),
        ("Решения:", summary.decisions),
        ("Осталось:", summary.unfinished),
    )
    for title, items in blocks:
        if not items:
            continue
        lines.append(title)
        lines.extend(f"- {item.text}" for item in items)
    if summary.topics:
        lines.append("Темы: " + ", ".join(summary.topics))
    if summary.summary:
        lines.append(summary.summary)
    return "\n".join(lines).strip()


# ------------------------------------------------------------------ parsing
def _extract_json(answer: str) -> dict[str, Any] | None:
    """Достаёт JSON-объект из ответа, даже если модель обернула его в текст."""
    text = (answer or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        # ```json ... ```: убираем первую и последнюю строки обёртки.
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
    for candidate in (text, _balanced_object(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _balanced_object(text: str) -> str:
    """Вырезает первый объект с учётом строк и вложенных скобок."""
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : position + 1]
    return ""


def _items(value: Any, user_text: str) -> list[SummaryItem]:
    """Проверяет список пунктов: текст непустой, цитата есть в разговоре."""
    if not isinstance(value, list):
        return []
    found: list[SummaryItem] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict):
            continue
        text = _text(raw.get("text"))
        evidence = _text(raw.get("evidence"))
        if not text or not evidence:
            continue
        normalized = _normalize(evidence)
        if len(normalized) < MIN_EVIDENCE_CHARS or normalized not in user_text:
            continue
        key = _normalize(text)
        if not key or key in seen:
            continue
        seen.add(key)
        found.append(SummaryItem(text=text, evidence=evidence))
        if len(found) >= MAX_ITEMS:
            break
    return found


def _topics(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    topics: list[str] = []
    seen: set[str] = set()
    for raw in value:
        topic = _text(raw)
        key = _normalize(topic)
        if not topic or not key or key in seen:
            continue
        seen.add(key)
        topics.append(topic)
        if len(topics) >= MAX_TOPICS:
            break
    return topics


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _normalize(text: str) -> str:
    """Сравнение цитат: без регистра, пунктуации и различий ё/е."""
    lowered = (text or "").casefold().replace("ё", "е")
    cleaned = _NON_WORD.sub(" ", lowered)
    return _SPACES.sub(" ", cleaned).strip()
