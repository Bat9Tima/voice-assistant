from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^\w\s.]+", re.UNICODE)
_EDGE_DOT_RE = re.compile(r"(?<!\w)\.|\.(?!\w)", re.UNICODE)
_SPACE_RE = re.compile(r"\s+", re.UNICODE)


def normalize(text: str) -> str:
    """Приводит распознанный текст к виду, удобному для сравнения.

    Точка сохраняется внутри слов (example.com), но убирается на границах.
    """
    text = (text or "").lower().replace("ё", "е")
    text = _PUNCT_RE.sub(" ", text)
    text = _EDGE_DOT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (char_a != char_b)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def fuzzy_contains(text: str, phrase: str, max_distance: int = 2) -> bool:
    """Ищет фразу в тексте, допуская небольшие ошибки распознавания."""
    if not text or not phrase:
        return False
    if phrase in text:
        return True
    words = text.split()
    target = phrase.split()
    size = len(target)
    if size == 0 or len(words) < size:
        return False
    for start in range(len(words) - size + 1):
        window = " ".join(words[start : start + size])
        if levenshtein(window, phrase) <= max_distance:
            return True
    return False


_WAKE_FILLERS = {"окей", "ок", "эй", "ассистент", "ассистэнт"}


def strip_wake_word(text: str, aliases: list[str]) -> str:
    """Убирает ключевую фразу в начале, допуская ошибки распознавания.

    STT иногда теряет часть фразы («окей открой ютуб»), поэтому после точного
    совпадения убираются и отдельные слова-wake: «окей», «ассистент», «эй».
    """
    normalized = normalize(text)
    prepared = sorted(
        (normalize(alias) for alias in aliases if normalize(alias)),
        key=len,
        reverse=True,
    )
    for alias in prepared:
        if normalized.startswith(alias):
            return normalized[len(alias) :].strip()

    words = normalized.split()
    for alias in prepared:
        alias_words = alias.split()
        if len(words) < len(alias_words):
            continue
        window = " ".join(words[: len(alias_words)])
        if levenshtein(window, alias) <= 2:
            return " ".join(words[len(alias_words) :]).strip()

    while words and words[0] in _WAKE_FILLERS:
        words.pop(0)
    return " ".join(words)
