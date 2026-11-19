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
import threading
from collections.abc import Iterable
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_LIMIT = 2000
# Сколько фактов хранить в файле. Старые вытесняются: промпт всё равно
# читает только хвост, а файл не должен расти бесконечно.
DEFAULT_MAX_FACTS = 100
# Близкие повторы: доля общих трёхграмм не меньше этой. 0.8 ловит
# переформулировки вроде «любит чай с печеньем» и «люблю пить чай с
# печеньем», но не трогает разные факты про чай и кофе.
NEAR_DUPLICATE = 0.8
NEAR_MIN_CHARS = 12
PERSONA_HEADER = "Твои собственные факты (это твоя жизнь, не противоречь им):"
EMPTY_ANSWERS = {"нет", "нет фактов"}
_NUMBERED = re.compile(r"^\d+[.)]\s*")
_WORDS = re.compile(r"\w+")


class PersonaMemory:
    """Хранит и отдаёт факты Луны о себе. Пустой путь — память выключена."""

    def __init__(
        self,
        path: Path | str | None = None,
        limit: int = DEFAULT_LIMIT,
        max_facts: int = DEFAULT_MAX_FACTS,
    ) -> None:
        self.path = Path(path) if path else None
        self.limit = max(0, int(limit))
        self.max_facts = max(0, int(max_facts))
        # Резюме может писать фоновый воркер и восстановление сессий;
        # без замка два merge читают один файл и теряют факты друг друга.
        self._lock = threading.Lock()

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
        """Добавляет новые факты без повторов. Возвращает число новых.

        Повтор — это не только точное совпадение: близкие переформулировки
        тоже схлопываются, остаётся более полная. Заодно файл чистится от
        старых повторов и ограничивается ``max_facts``.
        """
        if self.path is None:
            return 0
        with self._lock:
            return self._merge_locked(facts)

    def _merge_locked(self, facts: Iterable[str]) -> int:
        existing = self.facts()
        kept = _compact(existing, self.max_facts)
        added = 0
        for raw in facts:
            fact = _clean(raw)
            if not fact:
                continue
            if _absorb(kept, fact):
                added += 1
        if self.max_facts and len(kept) > self.max_facts:
            kept = kept[-self.max_facts :]
        if kept != existing:
            self._write(kept)
        return added

    def compact(self) -> int:
        """Убирает повторы и лишние старые факты. Возвращает число строк.

        Вызывается при старте, чтобы почистить файл, накопленный старой
        версией. Ошибка записи не ломает запуск.
        """
        if self.path is None:
            return 0
        with self._lock:
            existing = self.facts()
            kept = _compact(existing, self.max_facts)
            if kept == existing:
                return 0
            try:
                self._write(kept)
            except OSError as exc:
                log.warning("Факты о Луне не сжаты: %s", exc)
                return 0
            removed = len(existing) - len(kept)
            if removed > 0:
                log.info("Память личности: убрано фактов %s", removed)
            return removed

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


def _compact(facts: list[str], max_facts: int) -> list[str]:
    """Чистит список от повторов и оставляет последние ``max_facts``."""
    kept: list[str] = []
    for raw in facts:
        fact = _clean(raw)
        if fact:
            _absorb(kept, fact)
    if max_facts and len(kept) > max_facts:
        kept = kept[-max_facts:]
    return kept


def _absorb(kept: list[str], fact: str) -> bool:
    """Добавляет факт, если он не повтор. ``True`` — список изменился."""
    fact_key = _key(fact)
    for old in list(kept):
        if _key(old) == fact_key:
            if len(fact) > len(old):
                kept.remove(old)
                kept.append(fact)
                return True
            return False
    close = [old for old in kept if _near_duplicate(old, fact)]
    if not close:
        kept.append(fact)
        return True
    # Из близких формулировок оставляем самую полную и свежую.
    if all(len(fact) > len(old) for old in close):
        for old in close:
            kept.remove(old)
        kept.append(fact)
        return True
    return False


def _near_duplicate(left: str, right: str) -> bool:
    """Похожи ли факты настолько, чтобы считать их одним.

    Сравниваются трёхграммы букв: порядок слов и окончания почти не
    мешают, а разные факты («любит чай» и «любит кофе») расходятся.
    """
    left_key = _gram_key(left)
    right_key = _gram_key(right)
    if min(len(left_key), len(right_key)) < NEAR_MIN_CHARS:
        return False
    left_grams = _grams(left_key)
    right_grams = _grams(right_key)
    if not left_grams or not right_grams:
        return False
    shared = len(left_grams & right_grams)
    return shared / min(len(left_grams), len(right_grams)) >= NEAR_DUPLICATE


def _grams(text: str) -> set[str]:
    if len(text) <= 3:
        return {text}
    return {text[index : index + 3] for index in range(len(text) - 2)}


def _gram_key(text: str) -> str:
    return " ".join(_WORDS.findall(_clean(text).casefold()))
