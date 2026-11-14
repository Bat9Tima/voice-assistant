"""Локальные ответы о дате и времени.

Дата и время — факт из системных часов Windows, поэтому для прямых
вопросов не нужны ни модель, ни поиск. Модуль понимает не только
«который час» и «какое завтра число», но и арифметику дат: сдвиги,
countdown до события, разницу между датами и день недели для конкретной
даты.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal

from .text import numbers_to_digits

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

MONTHS_PREPOSITIONAL = (
    "январе",
    "феврале",
    "марте",
    "апреле",
    "мае",
    "июне",
    "июле",
    "августе",
    "сентябре",
    "октябре",
    "ноябре",
    "декабре",
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

_MONTH_FORMS: dict[str, int] = {name: number for number, name in enumerate(MONTHS, start=1)}
_MONTH_FORMS.update({name: number for number, name in enumerate(MONTHS_PREPOSITIONAL, start=1)})
_MONTH_FORMS.update(
    {
        "январь": 1,
        "февраль": 2,
        "март": 3,
        "апрель": 4,
        "май": 5,
        "июнь": 6,
        "июль": 7,
        "август": 8,
        "сентябрь": 9,
        "октябрь": 10,
        "ноябрь": 11,
        "декабрь": 12,
    }
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
    "нынешнее",
    "нынешний",
    "нынешняя",
    "нынче",
    "послезавтра",
    "позавчера",
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

_YEAR_FORMS = {"год", "года", "году", "годом", "годе"}
_MONTH_FORMS_SET = {"месяц", "месяца", "месяцу", "месяцем", "месяце"}
_OLD_YEAR_WORDS = {"старый", "старого", "старому", "старым", "старом"}
_NEW_YEAR_WORDS = {
    "новый",
    "нового",
    "новому",
    "новым",
    "новом",
    "новые",
    "новогодний",
    "новогоднего",
    "новогодние",
}
_CHRISTMAS_WORDS = {
    "рождество",
    "рождества",
    "рождеству",
    "рождеством",
    "рождестве",
}
_SPRING_WORDS = {"весна", "весны", "весне", "весну", "весной", "весною"}
_SUMMER_WORDS = {"лето", "лета", "лету", "летом"}
_AUTUMN_WORDS = {"осень", "осени", "осенью"}
_WINTER_WORDS = {"зима", "зимы", "зиме", "зиму", "зимой", "зимою"}
_END_WORDS = {"конец", "конца", "концу", "концом", "конце"}

# Слова других тем: «погода через 10 дней» и «новости за сегодня» должны
# уходить своим обработчикам, а не в арифметику дат.
_NOT_DATETIME_WORDS = {
    "погода",
    "погоду",
    "погоды",
    "погоде",
    "погодой",
    "погодам",
    "погодах",
    "прогноз",
    "прогноза",
    "прогнозе",
    "прогнозом",
    "прогнозы",
    "прогнозов",
    "температура",
    "температуру",
    "температуры",
    "температурой",
    "новости",
    "новость",
    "новостью",
    "новостей",
    "новостям",
    "новостями",
    "новостях",
    "курс",
    "курса",
    "курсом",
    "курсы",
    "курсов",
    "биткоин",
    "биткоина",
    "биткоином",
    "биткоине",
    "доллар",
    "доллара",
    "долларом",
    "доллары",
    "долларов",
    "евро",
    "акции",
    "акций",
    "акция",
    "акцию",
    "акцией",
    "цена",
    "цены",
    "цену",
    "ценой",
    "ценам",
    "ценах",
    "стоимость",
    "стоимости",
    "стоимостью",
    "гороскоп",
    "гороскопа",
    "гороскопом",
    "гороскопы",
}

# Порядковые числительные в формах, которые встречаются в датах:
# «первое января» и «первого января». Составные («двадцать первого»)
# собираются из десятка и единицы.
_ORDINAL_DAYS = {
    "первое": 1,
    "первого": 1,
    "второе": 2,
    "второго": 2,
    "третье": 3,
    "третьего": 3,
    "четвертое": 4,
    "четвертого": 4,
    "пятое": 5,
    "пятого": 5,
    "шестое": 6,
    "шестого": 6,
    "седьмое": 7,
    "седьмого": 7,
    "восьмое": 8,
    "восьмого": 8,
    "девятое": 9,
    "девятого": 9,
    "десятое": 10,
    "десятого": 10,
    "одиннадцатое": 11,
    "одиннадцатого": 11,
    "двенадцатое": 12,
    "двенадцатого": 12,
    "тринадцатое": 13,
    "тринадцатого": 13,
    "четырнадцатое": 14,
    "четырнадцатого": 14,
    "пятнадцатое": 15,
    "пятнадцатого": 15,
    "шестнадцатое": 16,
    "шестнадцатого": 16,
    "семнадцатое": 17,
    "семнадцатого": 17,
    "восемнадцатое": 18,
    "восемнадцатого": 18,
    "девятнадцатое": 19,
    "девятнадцатого": 19,
    "двадцатое": 20,
    "двадцатого": 20,
    "тридцатое": 30,
    "тридцатого": 30,
}

_DAY_SUFFIXES = {"го", "е", "ое", "ого", "ый", "ой", "ая"}
_TENS_CARDINAL = {"двадцать": 20, "тридцать": 30}
_APPROX_WORDS = {"полтора", "полторы", "пол", "около", "примерно", "почти"}
_APPROX_DURATIONS = {
    "полгода": (6, "month"),
    "полмесяца": (15, "day"),
}

_RELATIVE_UNITS = {
    "день": "day",
    "дня": "day",
    "дней": "day",
    "неделя": "week",
    "недели": "week",
    "неделю": "week",
    "недель": "week",
    "месяц": "month",
    "месяца": "month",
    "месяцев": "month",
    "год": "year",
    "года": "year",
    "лет": "year",
}

_COUNTDOWN_UNITS = {
    "день": "day",
    "дня": "day",
    "дней": "day",
    "неделя": "week",
    "недели": "week",
    "неделю": "week",
    "недель": "week",
    "час": "hour",
    "часа": "hour",
    "часов": "hour",
    "минута": "minute",
    "минуты": "minute",
    "минут": "minute",
    "время": "time",
    "времени": "time",
}

_RELATIVE_UNIT_PATTERN = "|".join(
    sorted((re.escape(unit) for unit in _RELATIVE_UNITS), key=len, reverse=True)
)
_RELATIVE_FUTURE_RE = re.compile(
    rf"\bчерез\s+(?P<amount>\d+)?\s*(?P<unit>{_RELATIVE_UNIT_PATTERN})\b"
)
_RELATIVE_PAST_RE = re.compile(
    rf"\b(?P<amount>\d+)?\s*(?P<unit>{_RELATIVE_UNIT_PATTERN})\s+назад\b"
)

_EVENT_DATES = {
    "new_year": (1, 1),
    "old_new_year": (1, 14),
    "christmas": (1, 7),
    "spring": (3, 1),
    "summer": (6, 1),
    "autumn": (9, 1),
    "winter": (12, 1),
}

_END_SEASON_MONTHS = {
    "end_of_spring": 5,
    "end_of_summer": 8,
    "end_of_autumn": 11,
    "end_of_winter": 2,
}

_EVENT_LABELS = {
    "new_year": ("Новый год", "нового года"),
    "old_new_year": ("Старый Новый год", "старого Нового года"),
    "christmas": ("Рождество", "Рождества"),
    "spring": ("Весна", "весны"),
    "summer": ("Лето", "лета"),
    "autumn": ("Осень", "осени"),
    "winter": ("Зима", "зимы"),
    "end_of_spring": ("Конец весны", "конца весны"),
    "end_of_summer": ("Конец лета", "конца лета"),
    "end_of_autumn": ("Конец осени", "конца осени"),
    "end_of_winter": ("Конец зимы", "конца зимы"),
    "end_of_month": ("Конец месяца", "конца месяца"),
    "end_of_next_month": ("Конец следующего месяца", "конца следующего месяца"),
    "end_of_previous_month": ("Конец прошлого месяца", "конца прошлого месяца"),
    "end_of_year": ("Конец года", "конца года"),
    "end_of_next_year": ("Конец следующего года", "конца следующего года"),
    "end_of_previous_year": ("Конец прошлого года", "конца прошлого года"),
}

DateKind = Literal[
    "time",
    "date",
    "year",
    "relative",
    "countdown",
    "difference",
    "weekday",
    "days_in_month",
    "days_in_year",
]


@dataclass(frozen=True)
class DateSpec:
    """Дата, месяц или событие. Год может быть неизвестен."""

    day: int | None = None
    month: int | None = None
    year: int | None = None
    event: str = ""


@dataclass(frozen=True)
class DateTimeRequest:
    """Разобранный вопрос о дате, времени или годе."""

    kind: DateKind
    shift: int = 0
    amount: int = 0
    unit: str = "day"
    direction: int = 1
    target: DateSpec | None = None
    start: DateSpec | None = None
    end: DateSpec | None = None
    hint: str = ""


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


def _to_int(token: str, low: int, high: int) -> int | None:
    if not token.isdigit():
        return None
    value = int(token)
    return value if low <= value <= high else None


def _parse_day_token(tokens: list[str], index: int) -> tuple[int, int] | None:
    """День месяца по цифре или порядковому числительному."""
    token = tokens[index]
    value = _to_int(token, 1, 31)
    if value is not None:
        return value, 1
    value = _ORDINAL_DAYS.get(token)
    if value is None:
        return None
    if value < 10 and index > 0:
        tens = _TENS_CARDINAL.get(tokens[index - 1])
        if tens is not None:
            return tens + value, 2
    return value, 1


def _parse_date_spec(
    tokens: list[str], start: int = 0, end: int | None = None
) -> tuple[DateSpec, int, int] | None:
    """Ищет дату в токенах и возвращает спецификацию и границы совпадения."""
    limit = len(tokens) if end is None else end

    for index in range(start, limit - 2):
        day = _to_int(tokens[index], 1, 31)
        month = _to_int(tokens[index + 1], 1, 12)
        year = _to_int(tokens[index + 2], 1900, 2100)
        if day is not None and month is not None and year is not None:
            return DateSpec(day=day, month=month, year=year), index, index + 3

    for index in range(start, limit):
        month = _MONTH_FORMS.get(tokens[index])
        if month is None:
            continue
        day = None
        begin = index
        previous = index - 1
        if previous >= start and tokens[previous] in _DAY_SUFFIXES:
            previous -= 1
        if previous >= start:
            parsed = _parse_day_token(tokens, previous)
            if parsed is not None:
                day, consumed = parsed
                begin = previous - consumed + 1
        year = None
        after = index + 1
        if after < limit:
            maybe_year = _to_int(tokens[after], 1900, 2100)
            if maybe_year is not None:
                year = maybe_year
                after += 1
        if after < limit and tokens[after] in _YEAR_FORMS:
            after += 1
        return DateSpec(day=day, month=month, year=year), begin, after

    for index in range(start, limit):
        year = _to_int(tokens[index], 1900, 2100)
        if year is None:
            continue
        before = tokens[index - 1] if index > start else ""
        after = tokens[index + 1] if index + 1 < limit else ""
        if before in {"в", "с", "до", "к", "по"} or after in _YEAR_FORMS:
            return DateSpec(year=year), index, index + 1
    return None


def _find_year(tokens: list[str]) -> int | None:
    for token in tokens:
        year = _to_int(token, 1900, 2100)
        if year is not None:
            return year
    return None


def _has_new_year(tokens: list[str]) -> bool:
    words = set(tokens)
    return bool(words & _NEW_YEAR_WORDS) and bool(words & _YEAR_FORMS)


def _relative_hint(tokens: list[str]) -> str:
    """Направление «следующий» или «прошлый» для периода."""
    for token in tokens:
        if token.startswith("прошл"):
            return "previous"
        if token.startswith(("следующ", "будущ")):
            return "next"
    return ""


def _event_spec(tokens: list[str]) -> DateSpec | None:
    words = set(tokens)
    if words & _END_WORDS:
        relative = _relative_hint(tokens)
        if words & _SPRING_WORDS:
            return None if relative else DateSpec(event="end_of_spring")
        if words & _SUMMER_WORDS:
            return None if relative else DateSpec(event="end_of_summer")
        if words & _AUTUMN_WORDS:
            return None if relative else DateSpec(event="end_of_autumn")
        if words & _WINTER_WORDS:
            return None if relative else DateSpec(event="end_of_winter")
        if words & _MONTH_FORMS_SET:
            if relative == "previous":
                return DateSpec(event="end_of_previous_month")
            if relative == "next":
                return DateSpec(event="end_of_next_month")
            return DateSpec(event="end_of_month")
        for token in tokens:
            month = _MONTH_FORMS.get(token)
            if month is not None:
                if relative == "previous":
                    return None
                return DateSpec(month=month, event="end_of_month")
        if words & _YEAR_FORMS:
            if relative == "previous":
                return DateSpec(event="end_of_previous_year")
            if relative == "next":
                return DateSpec(event="end_of_next_year")
            return DateSpec(event="end_of_year")
    if _has_new_year(tokens):
        if words & _OLD_YEAR_WORDS:
            return DateSpec(month=1, day=14, event="old_new_year")
        return DateSpec(month=1, day=1, event="new_year")
    if words & _CHRISTMAS_WORDS:
        return DateSpec(month=1, day=7, event="christmas")
    if words & _SPRING_WORDS:
        return DateSpec(month=3, day=1, event="spring")
    if words & _SUMMER_WORDS:
        return DateSpec(month=6, day=1, event="summer")
    if words & _AUTUMN_WORDS:
        return DateSpec(month=9, day=1, event="autumn")
    if words & _WINTER_WORDS:
        return DateSpec(month=12, day=1, event="winter")
    return None


def _extract_target(tokens: list[str]) -> DateSpec | None:
    event = _event_spec(tokens)
    if event is not None:
        year = _find_year(tokens)
        if year is not None:
            return DateSpec(day=event.day, month=event.month, year=year, event=event.event)
        return event
    parsed = _parse_date_spec(tokens)
    if parsed is not None:
        return parsed[0]
    return None


def _parse_days_in(tokens: list[str]) -> DateTimeRequest | None:
    words = set(tokens)
    if "сколько" not in words or "в" not in words:
        return None
    if not words & {"день", "дня", "дней"}:
        return None
    relative = _relative_hint(tokens)
    if relative:
        past = relative == "previous"
        if words & _MONTH_FORMS_SET:
            event = "previous_month" if past else "next_month"
            return DateTimeRequest("days_in_month", target=DateSpec(event=event))
        if words & _YEAR_FORMS:
            event = "previous_year" if past else "next_year"
            return DateTimeRequest("days_in_year", target=DateSpec(event=event))
        month = next(
            (_MONTH_FORMS[token] for token in tokens if token in _MONTH_FORMS),
            None,
        )
        if month is not None:
            event = "previous_year_month" if past else "next_year_month"
            return DateTimeRequest("days_in_month", target=DateSpec(month=month, event=event))
        return None
    parsed = _parse_date_spec(tokens)
    if parsed is not None and parsed[0].month is not None:
        spec = parsed[0]
        return DateTimeRequest(
            "days_in_month",
            target=DateSpec(month=spec.month, year=spec.year),
        )
    if any(token.startswith("високосн") for token in tokens):
        return DateTimeRequest("days_in_year", target=DateSpec(event="leap"))
    if words & _MONTH_FORMS_SET:
        return DateTimeRequest("days_in_month", target=DateSpec())
    if parsed is not None and parsed[0].year is not None:
        return DateTimeRequest("days_in_year", target=DateSpec(year=parsed[0].year))
    if words & _YEAR_FORMS:
        return DateTimeRequest("days_in_year", target=DateSpec())
    return None


def _countdown_unit(tokens: list[str]) -> str | None:
    for token in tokens:
        unit = _COUNTDOWN_UNITS.get(token)
        if unit is not None:
            return unit
    return None


def _difference_unit(tokens: list[str]) -> str | None:
    unit = _countdown_unit(tokens)
    if unit in {"day", "week"}:
        return unit
    return None


def _parse_difference(tokens: list[str]) -> DateTimeRequest | None:
    words = set(tokens)
    if "сколько" not in words:
        return None
    unit = _difference_unit(tokens)
    if unit is None:
        return None

    if words & {"прошло", "прошла", "прошел"} and "с" in tokens:
        index = tokens.index("с")
        spec = _parse_date_spec(tokens, index + 1)
        if spec is not None:
            return DateTimeRequest("difference", start=spec[0], unit=unit, hint="past")

    if "между" in tokens:
        between = tokens.index("между")
        try:
            conjunction = tokens.index("и", between + 1)
        except ValueError:
            return None
        left = _parse_date_spec(tokens, between + 1, conjunction)
        right = _parse_date_spec(tokens, conjunction + 1)
        if left is not None and right is not None:
            return DateTimeRequest("difference", start=left[0], end=right[0], unit=unit)
        return None

    if "с" in tokens and "по" in tokens:
        start_index = tokens.index("с")
        end_index = tokens.index("по", start_index + 1)
        left = _parse_date_spec(tokens, start_index + 1, end_index)
        right = _parse_date_spec(tokens, end_index + 1)
        if left is not None and right is not None:
            return DateTimeRequest("difference", start=left[0], end=right[0], unit=unit)
        return None

    if "от" in tokens and "до" in tokens:
        start_index = tokens.index("от")
        end_index = tokens.index("до", start_index + 1)
        left = _parse_date_spec(tokens, start_index + 1, end_index)
        right = _parse_date_spec(tokens, end_index + 1)
        if left is not None and right is not None:
            return DateTimeRequest("difference", start=left[0], end=right[0], unit=unit)
    return None


def _parse_countdown(tokens: list[str]) -> DateTimeRequest | None:
    words = set(tokens)
    if "сколько" not in words:
        return None

    if "до" in tokens:
        index = tokens.index("до")
        unit = _countdown_unit(tokens[:index])
        if unit is None:
            return None
        target = _extract_target(tokens[index + 1 :])
        if target is None:
            return None
        return DateTimeRequest("countdown", target=target, unit=unit)

    if "через" in tokens:
        index = tokens.index("через")
        if index + 1 >= len(tokens) or tokens[index + 1] != "сколько":
            return None
        rest = tokens[index + 2 :]
        unit_index = next(
            (position for position, token in enumerate(rest) if token in _COUNTDOWN_UNITS),
            None,
        )
        if unit_index is None:
            return None
        unit = _COUNTDOWN_UNITS[rest[unit_index]]
        target = _extract_target(rest[unit_index + 1 :])
        if target is None:
            return None
        return DateTimeRequest("countdown", target=target, unit=unit)
    return None


def _parse_relative(tokens: list[str]) -> DateTimeRequest | None:
    words = set(tokens)
    if "сколько" in words:
        return None
    for token in tokens:
        duration = _APPROX_DURATIONS.get(token)
        if duration is None:
            continue
        if "через" in words:
            direction = 1
        elif "назад" in words:
            direction = -1
        else:
            return None
        amount, unit = duration
        return DateTimeRequest("relative", amount=amount, unit=unit, direction=direction)
    text = numbers_to_digits(" ".join(tokens))
    future = _RELATIVE_FUTURE_RE.search(text)
    past = _RELATIVE_PAST_RE.search(text)
    match = future or past
    if match is None:
        return None
    if match.group("amount") is None:
        # «через неделю» — это одна неделя, а «через полтора месяца» —
        # не одна: без числа принимаем только соседнюю единицу времени.
        if future is not None:
            after = text[match.start() :].split()
            if len(after) < 2 or after[1] != match.group("unit"):
                return None
        else:
            before = text[: match.end()].split()
            if len(before) < 2 or before[-2] != match.group("unit"):
                return None
            if len(before) >= 3 and before[-3] in _APPROX_WORDS:
                return None
    amount = int(match.group("amount") or 1)
    if amount <= 0:
        return None
    unit = _RELATIVE_UNITS[match.group("unit")]
    direction = 1 if future is not None else -1
    return DateTimeRequest("relative", amount=amount, unit=unit, direction=direction)


def _parse_weekday(tokens: list[str]) -> DateTimeRequest | None:
    words = set(tokens)
    if "день" not in words or "недели" not in words:
        return None
    target = _extract_target(tokens)
    if target is None or target.day is None or target.month is None:
        return None
    if any(token in {"был", "была", "было"} or token.startswith("прошл") for token in tokens):
        hint = "past"
    elif any(token in {"будет"} or token.startswith(("будущ", "следующ")) for token in tokens):
        hint = "future"
    else:
        hint = ""
    return DateTimeRequest("weekday", target=target, hint=hint)


def _shift(tokens: list[str]) -> int:
    if "послезавтра" in tokens:
        return 2
    if "завтра" in tokens:
        return 1
    if "позавчера" in tokens:
        return -2
    if "вчера" in tokens:
        return -1
    return 0


def parse_datetime_question(text: str) -> DateTimeRequest | None:
    """Разбирает вопрос о дате/времени. ``None`` — это не такой вопрос."""
    normalized = _normalized(text)
    if not normalized:
        return None
    if _TIME_RE.match(normalized):
        return DateTimeRequest("time")

    tokens = normalized.split()
    token_set = set(tokens)
    if token_set & _NOT_DATETIME_WORDS:
        return None

    for parser in (
        _parse_days_in,
        _parse_difference,
        _parse_countdown,
        _parse_relative,
        _parse_weekday,
    ):
        request = parser(tokens)
        if request is not None:
            return request

    if token_set & {"год", "году"} and token_set & _QUESTION_WORDS and token_set <= _YEAR_WORDS:
        return DateTimeRequest("year")

    has_date_target = bool(token_set & {"число", "дата", "день", "дня"})
    if has_date_target and token_set & _QUESTION_WORDS and token_set <= _DATE_WORDS:
        return DateTimeRequest("date", _shift(tokens))
    return None


def is_datetime_question(text: str) -> bool:
    """Прямой вопрос о дате, дне недели, годе или времени."""
    return parse_datetime_question(text) is not None


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


def _is_singular(value: int) -> bool:
    return value % 10 == 1 and value % 100 != 11


def _remaining(amount: int, one: str, few: str, many: str, feminine: bool = False) -> str:
    if _is_singular(amount):
        word = "осталась" if feminine else "остался"
    else:
        word = "осталось"
    return f"{word} {amount} {_plural(amount, one, few, many)}"


def _passed(amount: int, one: str, few: str, many: str, feminine: bool = False) -> str:
    if _is_singular(amount):
        word = "прошла" if feminine else "прошел"
    else:
        word = "прошло"
    return f"{word} {amount} {_plural(amount, one, few, many)}"


def _plural_unit(amount: int, unit: str) -> str:
    if unit == "week":
        return _plural(amount, "неделю", "недели", "недель")
    if unit == "month":
        return _plural(amount, "месяц", "месяца", "месяцев")
    if unit == "year":
        return _plural(amount, "год", "года", "лет")
    return _plural(amount, "день", "дня", "дней")


def _time_text(current: datetime) -> str:
    hour = "ноль" if current.hour == 0 else str(current.hour)
    hour_word = _plural(current.hour, "час", "часа", "часов")
    if current.minute == 0:
        return f"Сейчас {hour} {hour_word} ровно."
    minute_word = _plural(current.minute, "минута", "минуты", "минут")
    return f"Сейчас {hour} {hour_word} {current.minute} {minute_word}."


def _date_text(current: datetime, shift: int) -> str:
    target = current + timedelta(days=shift)
    prefix = {1: "Завтра", 2: "Послезавтра", -1: "Вчера", -2: "Позавчера"}.get(shift, "Сегодня")
    return (
        f"{prefix} {target.day} {MONTHS[target.month - 1]} {target.year} года, "
        f"{WEEKDAYS[target.weekday()]}."
    )


def _valid_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _add_months(start: date, months: int) -> date:
    index = start.month - 1 + months
    year = start.year + index // 12
    month = index % 12 + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _relative_date(start: date, request: DateTimeRequest) -> date:
    amount = request.amount * request.direction
    if request.unit == "week":
        return start + timedelta(days=7 * amount)
    if request.unit == "month":
        return _add_months(start, amount)
    if request.unit == "year":
        return _add_months(start, 12 * amount)
    return start + timedelta(days=amount)


def _yearless_date(spec: DateSpec, current: datetime, hint: str) -> date | None:
    if spec.month is None or spec.day is None:
        return None
    today = current.date()
    if hint == "past":
        years = range(today.year, today.year - 9, -1)
    else:
        years = range(today.year, today.year + 9)
    for year in years:
        candidate = _valid_date(year, spec.month, spec.day)
        if candidate is None:
            continue
        if hint == "past" and candidate <= today:
            return candidate
        if hint != "past" and candidate >= today:
            return candidate
    return None


def _resolve_past(spec: DateSpec, current: datetime) -> date | None:
    if spec.month is None or spec.day is None:
        return None
    if spec.year is not None:
        return _valid_date(spec.year, spec.month, spec.day)
    return _yearless_date(spec, current, "past")


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _yearless_month_end(month: int, current: datetime, hint: str) -> date | None:
    today = current.date()
    if hint == "past":
        years = range(today.year, today.year - 9, -1)
    else:
        years = range(today.year, today.year + 9)
    for year in years:
        candidate = _month_end(year, month)
        if hint == "past" and candidate <= today:
            return candidate
        if hint != "past" and candidate >= today:
            return candidate
    return None


def _shift_month_end(current: datetime, step: int) -> date:
    target = _add_months(date(current.year, current.month, 1), step)
    return _month_end(target.year, target.month)


def _event_date(spec: DateSpec, current: datetime, hint: str) -> date | None:
    if spec.event == "end_of_month":
        if spec.month is not None:
            if spec.year is not None:
                return _month_end(spec.year, spec.month)
            return _yearless_month_end(spec.month, current, hint)
        year = spec.year or current.year
        return _month_end(year, current.month)
    if spec.event == "end_of_next_month":
        return _shift_month_end(current, 1)
    if spec.event == "end_of_previous_month":
        return _shift_month_end(current, -1)
    if spec.event in _END_SEASON_MONTHS:
        month = _END_SEASON_MONTHS[spec.event]
        if spec.year is not None:
            return _month_end(spec.year, month)
        return _yearless_month_end(month, current, hint)
    if spec.event == "end_of_year":
        year = spec.year or current.year
        return date(year, 12, 31)
    if spec.event == "end_of_next_year":
        return date(current.year + 1, 12, 31)
    if spec.event == "end_of_previous_year":
        return date(current.year - 1, 12, 31)
    month, day = _EVENT_DATES[spec.event]
    if spec.year is not None:
        return _valid_date(spec.year, month, day)
    return _yearless_date(DateSpec(day=day, month=month), current, hint)


def _target_date(spec: DateSpec, current: datetime, hint: str) -> date | None:
    if spec.event:
        return _event_date(spec, current, hint)
    if spec.month is None:
        if spec.year is not None:
            return date(spec.year, 1, 1)
        return None
    day = spec.day if spec.day is not None else 1
    if spec.year is not None:
        return _valid_date(spec.year, spec.month, day)
    return _yearless_date(DateSpec(day=day, month=spec.month), current, hint)


def _format_date(spec: DateSpec) -> str:
    if spec.day is not None and spec.month is not None:
        text = f"{spec.day} {MONTHS[spec.month - 1]}"
        if spec.year is not None:
            text += f" {spec.year} года"
        return text
    if spec.month is not None:
        text = MONTHS[spec.month - 1]
        if spec.year is not None:
            text += f" {spec.year} года"
        return text
    if spec.year is not None:
        return f"{spec.year} год"
    return ""


def _format_date_with_year(target: date) -> str:
    return f"{target.day} {MONTHS[target.month - 1]} {target.year} года"


def _missing_date_text(spec: DateSpec | None) -> str:
    if spec is not None and spec.day is not None and spec.month is not None:
        return f"Такой даты нет: {_format_date(spec)}."
    return "Не понял дату."


def _relative_text(request: DateTimeRequest, current: datetime) -> str:
    target = _relative_date(current.date(), request)
    if target == current.date():
        return _date_text(current, 0)
    if request.direction > 0:
        if request.unit == "day" and request.amount == 1:
            prefix = "Завтра"
        elif request.unit == "day" and request.amount == 2:
            prefix = "Послезавтра"
        else:
            prefix = f"Через {request.amount} {_plural_unit(request.amount, request.unit)} будет"
    else:
        if request.unit == "day" and request.amount == 1:
            prefix = "Вчера"
        elif request.unit == "day" and request.amount == 2:
            prefix = "Позавчера"
        else:
            prefix = f"{request.amount} {_plural_unit(request.amount, request.unit)} назад было"
    return (
        f"{prefix} {target.day} {MONTHS[target.month - 1]} {target.year} года, "
        f"{WEEKDAYS[target.weekday()]}."
    )


def _event_label(spec: DateSpec, target: date, genitive: bool) -> str:
    if spec.event == "end_of_month" and spec.month is not None:
        month = MONTHS[spec.month - 1]
        return f"конца {month}" if genitive else f"Конец {month}"
    if spec.event:
        labels = _EVENT_LABELS[spec.event]
        return labels[1] if genitive else labels[0]
    return _format_date_with_year(target)


def _countdown_text(request: DateTimeRequest, current: datetime) -> str:
    spec = request.target
    if spec is None:
        return "Не понял дату."
    target = _target_date(spec, current, "future")
    if target is None:
        return _missing_date_text(spec)

    days = (target - current.date()).days
    if days < 0:
        if spec.event:
            return f"{_event_label(spec, target, genitive=False)} уже в прошлом."
        return f"{_format_date_with_year(target)} уже в прошлом."
    if days == 0:
        if spec.event:
            return f"{_event_label(spec, target, genitive=False)} сегодня."
        return f"Сегодня {_format_date_with_year(target)}."

    genitive = _event_label(spec, target, genitive=True)
    if request.unit == "day":
        return f"До {genitive} {_remaining(days, 'день', 'дня', 'дней')}."
    if request.unit == "week":
        weeks, remainder = divmod(days, 7)
        if weeks == 0:
            return f"До {genitive} {_remaining(days, 'день', 'дня', 'дней')}."
        if remainder == 0:
            return (
                f"До {genitive} {_remaining(weeks, 'неделя', 'недели', 'недель', feminine=True)}."
            )
        return (
            f"До {genitive} "
            f"{_remaining(weeks, 'неделя', 'недели', 'недель', feminine=True)} "
            f"и {remainder} {_plural(remainder, 'день', 'дня', 'дней')}."
        )

    target_midnight = datetime.combine(target, time.min, tzinfo=current.tzinfo)
    seconds = int((target_midnight - current).total_seconds())
    if seconds <= 0:
        return f"{_event_label(spec, target, genitive=False)} уже наступило."
    if request.unit == "hour":
        hours = seconds // 3600
        if hours == 0:
            minutes = max(1, seconds // 60)
            return (
                f"До {genitive} {_remaining(minutes, 'минута', 'минуты', 'минут', feminine=True)}."
            )
        return f"До {genitive} {_remaining(hours, 'час', 'часа', 'часов')}."
    if request.unit == "minute":
        minutes = max(1, seconds // 60)
        return f"До {genitive} {_remaining(minutes, 'минута', 'минуты', 'минут', feminine=True)}."

    day_part, remainder = divmod(seconds, 86400)
    hour_part, remainder = divmod(remainder, 3600)
    minute_part = remainder // 60
    parts: list[str] = []
    if day_part:
        parts.append(f"{day_part} {_plural(day_part, 'день', 'дня', 'дней')}")
    if hour_part:
        parts.append(f"{hour_part} {_plural(hour_part, 'час', 'часа', 'часов')}")
    if minute_part and not day_part:
        parts.append(f"{minute_part} {_plural(minute_part, 'минута', 'минуты', 'минут')}")
    if not parts:
        return f"До {genitive} меньше минуты."
    if day_part:
        word = "остался" if _is_singular(day_part) else "осталось"
    elif hour_part:
        word = "остался" if _is_singular(hour_part) else "осталось"
    else:
        word = "осталась" if _is_singular(minute_part) else "осталось"
    return f"До {genitive} {word} {' '.join(parts)}."


def _resolve_pair(
    start_spec: DateSpec | None,
    end_spec: DateSpec | None,
    current: datetime,
) -> tuple[date | None, date | None]:
    if start_spec is None or end_spec is None:
        return None, None
    if start_spec.month is None or start_spec.day is None:
        return None, None
    if end_spec.month is None or end_spec.day is None:
        return None, None

    if start_spec.year is not None:
        start = _valid_date(start_spec.year, start_spec.month, start_spec.day)
        end_year = end_spec.year if end_spec.year is not None else start_spec.year
        end = _valid_date(end_year, end_spec.month, end_spec.day)
        if end is not None and end < start and end_spec.year is None:
            end = _valid_date(end_year + 1, end_spec.month, end_spec.day)
        return start, end

    if end_spec.year is not None:
        end = _valid_date(end_spec.year, end_spec.month, end_spec.day)
        start = _valid_date(end_spec.year, start_spec.month, start_spec.day)
        if start is not None and end is not None and start > end:
            start = _valid_date(end_spec.year - 1, start_spec.month, start_spec.day)
        return start, end

    start = _valid_date(current.year, start_spec.month, start_spec.day)
    end = _valid_date(current.year, end_spec.month, end_spec.day)
    if start is not None and end is not None and end < start:
        end = _valid_date(current.year + 1, end_spec.month, end_spec.day)
    return start, end


def _difference_text(request: DateTimeRequest, current: datetime) -> str:
    if request.end is None:
        if request.start is None:
            return "Не понял дату."
        start = _resolve_past(request.start, current)
        if start is None:
            return _missing_date_text(request.start)
        days = (current.date() - start).days
        label = _format_date_with_year(start)
        if days < 0:
            return f"{label} ещё не наступило."
        if days == 0:
            return f"С {label} не прошло ни дня."
        if request.unit == "week":
            weeks, remainder = divmod(days, 7)
            if weeks and remainder:
                return (
                    f"С {label} прошло {weeks} "
                    f"{_plural(weeks, 'неделя', 'недели', 'недель')} "
                    f"и {remainder} {_plural(remainder, 'день', 'дня', 'дней')}."
                )
            if weeks:
                return f"С {label} {_passed(weeks, 'неделя', 'недели', 'недель', feminine=True)}."
        return f"С {label} {_passed(days, 'день', 'дня', 'дней')}."

    start, end = _resolve_pair(request.start, request.end, current)
    if start is None:
        return _missing_date_text(request.start)
    if end is None:
        return _missing_date_text(request.end)
    if end < start:
        return "Вторая дата раньше первой."
    days = (end - start).days
    if days == 0:
        return f"Это одна и та же дата: {_format_date_with_year(start)}."
    start_label = _format_date_with_year(start)
    end_label = _format_date_with_year(end)
    if request.unit == "week":
        weeks, remainder = divmod(days, 7)
        if weeks and remainder:
            return (
                f"От {start_label} до {end_label} {weeks} "
                f"{_plural(weeks, 'неделя', 'недели', 'недель')} "
                f"и {remainder} {_plural(remainder, 'день', 'дня', 'дней')}."
            )
        if weeks:
            return (
                f"От {start_label} до {end_label} {weeks} "
                f"{_plural(weeks, 'неделя', 'недели', 'недель')}."
            )
    return f"От {start_label} до {end_label} {days} {_plural(days, 'день', 'дня', 'дней')}."


def _weekday_text(request: DateTimeRequest, current: datetime) -> str:
    spec = request.target
    if spec is None or spec.month is None or spec.day is None:
        return "Не понял дату."
    if spec.year is not None:
        target = _valid_date(spec.year, spec.month, spec.day)
    else:
        target = _yearless_date(spec, current, request.hint)
    if target is None:
        return _missing_date_text(spec)
    return f"{_format_date_with_year(target)}, {WEEKDAYS[target.weekday()]}."


def _days_in_month_text(request: DateTimeRequest, current: datetime) -> str:
    spec = request.target or DateSpec()
    if spec.event in {"next_month", "previous_month"}:
        step = -1 if spec.event == "previous_month" else 1
        target = _add_months(date(current.year, current.month, 1), step)
        month = target.month
        year = target.year
    elif spec.event in {"next_year_month", "previous_year_month"}:
        month = spec.month if spec.month is not None else current.month
        year = current.year + (-1 if spec.event == "previous_year_month" else 1)
    else:
        month = spec.month if spec.month is not None else current.month
        year = spec.year if spec.year is not None else current.year
    days = calendar.monthrange(year, month)[1]
    return (
        f"В {MONTHS_PREPOSITIONAL[month - 1]} {year} года {days} "
        f"{_plural(days, 'день', 'дня', 'дней')}."
    )


def _days_in_year_text(request: DateTimeRequest, current: datetime) -> str:
    spec = request.target or DateSpec()
    if spec.event == "leap":
        return "В високосном году 366 дней."
    year = current.year
    if spec.event == "next_year":
        year += 1
    elif spec.event == "previous_year":
        year -= 1
    elif spec.year is not None:
        year = spec.year
    days = 366 if calendar.isleap(year) else 365
    return f"В {year} году {days} {_plural(days, 'день', 'дня', 'дней')}."


def answer_request(request: DateTimeRequest, now: datetime | None = None) -> str:
    """Готовый ответ по уже разобранному вопросу."""
    current = now or datetime.now().astimezone()
    if request.kind == "time":
        return _time_text(current)
    if request.kind == "year":
        return f"Сейчас {current.year} год."
    if request.kind == "relative":
        return _relative_text(request, current)
    if request.kind == "countdown":
        return _countdown_text(request, current)
    if request.kind == "difference":
        return _difference_text(request, current)
    if request.kind == "weekday":
        return _weekday_text(request, current)
    if request.kind == "days_in_month":
        return _days_in_month_text(request, current)
    if request.kind == "days_in_year":
        return _days_in_year_text(request, current)
    return _date_text(current, request.shift)


def answer_datetime(text: str, now: datetime | None = None) -> str | None:
    """Русский ответ для озвучки или ``None``, если это не про дату/время."""
    request = parse_datetime_question(text)
    if request is None:
        return None
    return answer_request(request, now)


def datetime_context(now: datetime | None = None) -> str:
    """Короткий контекст даты и времени для системного промпта модели."""
    current = now or datetime.now().astimezone()
    month = MONTHS[current.month - 1]
    weekday = WEEKDAYS[current.weekday()]
    return (
        f"Сегодня {current.day} {month} {current.year} года, {weekday}. "
        f"Сейчас {current.hour:02d}:{current.minute:02d}."
    )
