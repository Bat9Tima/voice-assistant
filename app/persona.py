"""Факты Луны о себе: отдельная память личности.

Пользовательский профиль (``state/profile.txt``) — это правила персонажа и
факты о пользователе. Здесь — то, что Луна сказала о себе: возраст, город,
вкусы, привычки, события. Факты извлекаются из завершённого разговора и
подмешиваются в системный промпт отдельным блоком, чтобы она не
противоречила себе в следующих сессиях.

Файл простой: одна строка — один факт. Его можно править руками. Ошибки
чтения или записи не ломают диалог: просто не будет блока или новых фактов.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_LIMIT = 2000
PERSONA_HEADER = "Твои собственные факты (это твоя жизнь, не противоречь им):"
EMPTY_ANSWERS = {"нет", "нет фактов"}
_NUMBERED = re.compile(r"^\d+[.)]\s*")


class PersonaMemory:
    """Хранит и отдаёт факты Луны о себе. Пустой путь — память выключена."""

    def __init__(self, path: Path | str | None = None, limit: int = DEFAULT_LIMIT) -> None:
        self.path = Path(path) if path else None
        self.limit = max(0, int(limit))

    def facts(self) -> list[str]:
        """Все факты из файла в порядке записи. Ошибки дают пустой список."""
        if self.path is None or not self.path.exists():
            return []
        try:
            data = self.path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            log.warning("Факты о Луне не прочитаны: %s", exc)
            return []
        return _parse_lines(data)

    def text(self) -> str:
        """Факты списком для промпта, не длиннее ``limit`` символов.

        При переполнении остаются последние факты: они свежее.
        """
        facts = self.facts()
        if not facts:
            return ""
        text = "\n".join(f"- {fact}" for fact in facts)
        if self.limit and len(text) > self.limit:
            text = text[-self.limit :]
            boundary = text.find("\n")
            if boundary != -1:
                text = text[boundary + 1 :]
        return text

    def block(self) -> str:
        """Готовый кусок системного промпта или пустая строка."""
        text = self.text()
        if not text:
            return ""
        return f"{PERSONA_HEADER}\n{text}"

    def merge(self, facts: Iterable[str]) -> int:
        """Добавляет новые факты без точных повторов. Возвращает число новых."""
        if self.path is None:
            return 0
        existing = self.facts()
        known = {_key(fact) for fact in existing}
        added: list[str] = []
        for raw in facts:
            fact = _clean(raw)
            if not fact:
                continue
            key = _key(fact)
            if key in known:
                continue
            known.add(key)
            added.append(fact)
        if not added:
            return 0
        self._write([*existing, *added])
        return len(added)

    def _write(self, facts: list[str]) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text("\n".join(facts) + "\n", encoding="utf-8", newline="\n")
        os.replace(temp, self.path)


def parse_facts(answer: str) -> list[str]:
    """Разбирает ответ модели: одна строка — один факт.

    Снимает маркеры списка и нумерацию, выбрасывает «НЕТ» и заголовки.
    """
    facts: list[str] = []
    for raw in (answer or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ", "• ")):
            line = line[2:].strip()
        line = _NUMBERED.sub("", line).strip()
        if not line:
            continue
        if line.endswith(":") and len(line) < 40:
            continue
        fact = _clean(line)
        if not fact:
            continue
        facts.append(fact)
    return facts


def _parse_lines(data: str) -> list[str]:
    facts: list[str] = []
    for raw in data.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ", "• ")):
            line = line[2:].strip()
        if line:
            facts.append(line)
    return facts


def _clean(text: str) -> str:
    line = " ".join(str(text or "").split()).strip(" -•*\t")
    if not line or line.casefold().rstrip(".!;:") in EMPTY_ANSWERS:
        return ""
    return line


def _key(text: str) -> str:
    return _clean(text).casefold().rstrip(".!;:")
