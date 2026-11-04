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


_SOUNDCLOUD_ANY_RE = re.compile(
    r"(?:сау[нтдк]?[ао]?|с[ао]ун[дт]?|с[ао]н[дт]?[ао]?)"
    r"\s*[кг]?\s*л[ао]у?[дт]\w*"
    r"|sound\s*cloud",
    re.UNICODE,
)
_LOFI_ANY_RE = re.compile(r"(?:лоу|лу|ло)\s*фай\w*", re.UNICODE)
_YOUTUBE_SHORT_RE = re.compile(
    r"(открой|включи|запусти|поставь)\s+(?:ту|тю|тюб|ют|юта|юту|юбе|юб)"
)
_SOUNDCLOUD_ENDINGS = "ауе"
_QUIET_NUMBER_RE = re.compile(
    r"^тишина\s+(?=\d|ноль|один|два|три|четыре|пять|шесть|семь|восемь|девять|десять)",
    re.UNICODE,
)


def _canonical_soundcloud(match: re.Match[str]) -> str:
    word = match.group(0).replace(" ", "")
    ending = word[-1] if word[-1] in _SOUNDCLOUD_ENDINGS else ""
    return "саундклауд" + ending


def canonicalize(text: str) -> str:
    """Приводит названия сервисов к одному написанию.

    STT слышит бренд по-разному: «санклауда», «саунглауда», «соунклауда».
    Без нормализации страдает и точность текста, и поисковый запрос.
    """
    if not text:
        return text
    result = _SOUNDCLOUD_ANY_RE.sub(_canonical_soundcloud, text)
    result = _LOFI_ANY_RE.sub("лоуфай", result)
    if result.startswith("тишина "):
        # GigaAM слышит «тише» как «тишина»: «тишина два» → «тише на два».
        result = re.sub(r"^тишина\s+на\s+", "тише на ", result)
        result = _QUIET_NUMBER_RE.sub("тише на ", result)
    match = _YOUTUBE_SHORT_RE.fullmatch(result)
    if match:
        return f"{match.group(1)} ютуб"
    return result


_WAKE_FILLERS = {
    "окей", "ок", "эй", "ассистент", "ассистэнт",
    "хей", "хэй", "хай",
}


def _wake_prefix_size(text: str, aliases: list[str], max_distance: int = 2) -> int:
    """Сколько первых слов занимает ключевая фраза.

    STT часто сливает «хей луна» в «хейлуна», «эйлона» или даже «эйна»,
    поэтому сравниваем первые 1–3 слова без пробелов с тем же вариантом
    алиаса. Допуск маленький: «хэй люда» (расстояние 3) остаётся командой.
    """
    words = text.split()
    targets = [normalize(alias).replace(" ", "") for alias in aliases]
    targets = [target for target in targets if target]
    if not targets:
        return 0
    for size in (1, 2, 3):
        if len(words) < size:
            continue
        window = "".join(words[:size])
        for target in targets:
            if window == target:
                return size
            if (
                abs(len(window) - len(target)) <= max_distance
                and levenshtein(window, target) <= max_distance
            ):
                return size
    return 0


def strip_wake_word(text: str, aliases: list[str]) -> str:
    """Убирает ключевую фразу в начале, допуская ошибки распознавания.

    STT иногда теряет часть фразы («окей открой ютуб») или сливает её в одно
    слово («хейлуна открой ютуб»), поэтому после точного совпадения
    проверяются слитные варианты, а затем отдельные слова-wake.
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
    size = _wake_prefix_size(normalized, prepared)
    if size:
        return " ".join(words[size:]).strip()

    while words and words[0] in _WAKE_FILLERS:
        words.pop(0)
    return " ".join(words)


def split_agent_trigger(text: str, triggers: list[str]) -> tuple[bool, str]:
    """Отделяет явный вызов агента от текста запроса.

    Триггер учитывается только в начале фразы, поэтому «включи фильм
    агент 007» остаётся обычной командой. Возвращает пару
    «сработал ли триггер» и остаток запроса.
    """
    normalized = normalize(text)
    prepared = {normalize(trigger) for trigger in triggers}
    prepared.discard("")
    for trigger in sorted(prepared, key=len, reverse=True):
        if normalized == trigger:
            return True, ""
        if normalized.startswith(trigger + " "):
            return True, normalized[len(trigger):].strip()
    return False, text
