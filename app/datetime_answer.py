"""Локальные ответы о дате и времени.

Дата и время — факт из системных часов Windows, поэтому для прямых
вопросов не нужны ни модель, ни поиск. Модуль понимает короткие
формулировки вроде «который час» и «какое завтра число», но не берётся
за длительности («сколько дней до нового года») и произвольные даты.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

MONTHS = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)

WEEKDAYS = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)

# Слова вежливости, которые не меняют смысл вопроса.
FILLER_WORDS = {
    "а",
    "можно",
    "мне",
    "ну",
    "подскажи",
    "подскажите",
    "пожалуйста",
    "скажи",
    "скажите",
    "так",
    "узнать",
}

_TIME_RE = re.compile(
    r"^(?:"
    r"который(?:\s+(?:сейчас|теперь))?\s+час(?:\s+(?:сейчас|теперь))?"
    r"|сколько(?:\s+(?:сейчас|теперь))?\s+(?:времени|время|на\s+часах)"
    r"|какое(?:\s+(?:сейчас|теперь))?\s+время"
    r"|(?:текущее\s+)?время(?:\s+сейчас)?"
    r")$"
)

_QUESTION_WORDS = {"какое", "какая", "какой", "какого", "каком"}

_DATE_WORDS = {
    "был",
    "была",
    "было",
    "будет",
    "в",
    "дата",
    "день",
    "дня",
    "есть",
    "завтра",
    "какая",
    "какое",
    "какой",
    "на",
    "недели",
    "неделю",
    "сегодня",
    "сегодняшнее",
    "сегодняшний",
    "сегодняшняя",
    "сейчас",
    "текущее",
    "текущий",
    "текущая",
    "теперь",
    "число",
    "вчера",
}

_YEAR_WORDS = {
    "в",
    "год",
    "году",
    "дворе",
    "живем",
    "какой",
    "каком",
    "мы",
    "на",
    "нынешний",
    "нынче",
    "сейчас",
    "текущий",
    "теперь",
    "этот",
}


@dataclass(frozen=True)
class _Request:
    kind: str  # time, date или year
    shift: int = 0


def _normalized(text: str) -> str:
    """Приводит фразу к виду без регистра, ё и знаков препинания."""
    lowered = (text or "").strip().lower().replace("ё", "е")
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    words = lowered.split()
    while words and words[0] in FILLER_WORDS:
        words.pop(0)
    while words and words[-1] in FILLER_WORDS:
        words.pop()
    return " ".join(words)


def _shift(tokens: list[str]) -> int:
    if "завтра" in tokens:
        return 1
    if "вчера" in tokens:
        return -1
    return 0


def _parse(text: str) -> _Request | None:
    normalized = _normalized(text)
    if not normalized:
        return None
    if _TIME_RE.match(normalized):
        return _Request("time")

    tokens = normalized.split()
    token_set = set(tokens)

    if token_set & {"год", "году"} and token_set & _QUESTION_WORDS and token_set <= _YEAR_WORDS:
        return _Request("year")

    has_date_target = bool(token_set & {"число", "дата", "день", "дня"})
    if has_date_target and token_set & _QUESTION_WORDS and token_set <= _DATE_WORDS:
        return _Request("date", _shift(tokens))
    return None


def is_datetime_question(text: str) -> bool:
    """Прямой вопрос о текущей дате, дне недели, годе или времени."""
    return _parse(text) is not None


def _plural(value: int, one: str, few: str, many: str) -> str:
    value = abs(value) % 100
    if 11 <= value <= 14:
        return many
    value %= 10
    if value == 1:
        return one
    if 2 <= value <= 4:
        return few
    return many


def _time_text(current: datetime) -> str:
    hour = "ноль" if current.hour == 0 else str(current.hour)
    hour_word = _plural(current.hour, "час", "часа", "часов")
    if current.minute == 0:
        return f"Сейчас {hour} {hour_word} ровно."
    minute_word = _plural(current.minute, "минута", "минуты", "минут")
    return f"Сейчас {hour} {hour_word} {current.minute} {minute_word}."


def _date_text(current: datetime, shift: int) -> str:
    target = current + timedelta(days=shift)
    prefix = {1: "Завтра", -1: "Вчера"}.get(shift, "Сегодня")
    month = MONTHS[target.month - 1]
    weekday = WEEKDAYS[target.weekday()]
    return f"{prefix} {target.day} {month} {target.year} года, {weekday}."


def _answer(text: str, current: datetime) -> str | None:
    request = _parse(text)
    if request is None:
        return None
    if request.kind == "time":
        return _time_text(current)
    if request.kind == "year":
        return f"Сейчас {current.year} год."
    return _date_text(current, request.shift)


def answer_datetime(text: str, now: datetime | None = None) -> str | None:
    """Русский ответ для озвучки или ``None``, если это не про дату/время."""
    current = now or datetime.now().astimezone()
    return _answer(text, current)


def datetime_context(now: datetime | None = None) -> str:
    """Короткий контекст даты и времени для системного промпта модели."""
    current = now or datetime.now().astimezone()
    month = MONTHS[current.month - 1]
    weekday = WEEKDAYS[current.weekday()]
    return (
        f"Сегодня {current.day} {month} {current.year} года, {weekday}. "
        f"Сейчас {current.hour:02d}:{current.minute:02d}."
    )
