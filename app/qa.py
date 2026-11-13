from __future__ import annotations

import html
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .cloud_llm import CloudLLM, LLMError
from .search import SearchError, SearchResult, TavilySearch
from .weather import (
    WeatherClient,
    WeatherError,
    format_current,
    format_day,
    format_report,
)

log = logging.getLogger(__name__)

QUESTION_STARTS = {
    "что",
    "кто",
    "где",
    "когда",
    "почему",
    "зачем",
    "как",
    "какой",
    "какая",
    "какое",
    "какие",
    "сколько",
    "куда",
    "откуда",
    "чей",
    "чья",
    "чьи",
    "расскажи",
    "объясни",
    "подскажи",
    "посоветуй",
    "порекомендуй",
}

FRESH_MARKERS = (
    "новост",
    "нового",
    "новое",
    "случилось",
    "происходит",
    "слышно",
    "последн",
    "свеж",
    "недавн",
    "сегодня",
    "сейчас",
    "вчера",
    "завтра",
    "в этом году",
    "в 2026",
    "курс",
    "цена",
    "цены",
    "стоимость",
    "сколько стоит",
    "стоить",
    "пробк",
    "матч",
    "счет",
    "выиграл",
    "проиграл",
    "вышел",
    "вышла",
    "релиз",
    "обновлен",
    "выборы",
    "происшеств",
    "авари",
    "катастроф",
    "инфляц",
    "санкц",
    "биткоин",
    "доллар",
    "евро",
    "акци",
    "котировк",
    "когда",
    "кто президент",
    "кто сейчас",
)

NEWS_MARKERS = (
    "новост",
    "нового",
    "новое",
    "случилось",
    "происходит",
    "слышно",
    "последн",
    "свеж",
    "недавн",
)

NEWS_ONLY = re.compile(
    r"^(?:(?:какие|что|последние|свежие|недавние|новые)\s+)?"
    r"(?:новости|нового|новое|случилось|происходит)"
    r"(?:\s+(?:(?:в|по|о)\s+)?(?:россии|мире|стране))?$"
)

# Временные уточнения в новостных просьбах: «за последний день», «за вчера»,
# «на сегодня», «за последние 24 часа». Служат только для распознавания
# свободной формулировки и не попадают в итоговый запрос.
NEWS_RELATIVE_TIME_RE = re.compile(
    r"(?i)(?:^|\s)(?:(?:за|на|в)\s+)?"
    r"(?:(?:последн\w*|прошл\w*|прошедш\w*|текущ\w*|эт\w*)\s+)?"
    r"(?:\d+\s+|несколько\s+|пару\s+)?"
    r"(?:день|дня|дней|дни|сутки|суток|неделю|недели|недель|месяц|месяца|"
    r"месяцев|час|часа|часов|сегодня|вчера|позавчера)\b"
)

# Служебные слова новостной просьбы. Если после их удаления темы не осталось,
# запрос считается общим и заменяется стабильной формулировкой.
NEWS_FILLER_WORDS = {
    "а",
    "без",
    "бы",
    "в",
    "во",
    "вчера",
    "главная",
    "главнее",
    "главное",
    "главные",
    "главный",
    "говорят",
    "да",
    "дай",
    "дайте",
    "дайджест",
    "день",
    "дня",
    "дней",
    "до",
    "ещё",
    "еще",
    "за",
    "и",
    "из",
    "как",
    "какая",
    "какие",
    "какой",
    "какое",
    "коротко",
    "кратко",
    "лента",
    "ленту",
    "месяц",
    "месяца",
    "месяцев",
    "мне",
    "можешь",
    "можете",
    "мир",
    "мире",
    "на",
    "над",
    "недавние",
    "несколько",
    "недели",
    "недель",
    "неделю",
    "нового",
    "новое",
    "новенькие",
    "новенькое",
    "новенького",
    "новостей",
    "новости",
    "новостная",
    "новостного",
    "новостной",
    "новостную",
    "новостные",
    "новостных",
    "новостям",
    "новостях",
    "новостями",
    "новость",
    "новостью",
    "новые",
    "новый",
    "о",
    "об",
    "от",
    "пара",
    "пару",
    "под",
    "пожалуйста",
    "позавчера",
    "покажи",
    "покажите",
    "повестка",
    "повестку",
    "пишет",
    "пишут",
    "по",
    "после",
    "последние",
    "последний",
    "последних",
    "последнюю",
    "при",
    "про",
    "произошло",
    "происходит",
    "происходящее",
    "прошлые",
    "прошлый",
    "прошлую",
    "прямо",
    "расскажи",
    "расскажите",
    "с",
    "свежие",
    "свежих",
    "сводка",
    "сводку",
    "сегодня",
    "сейчас",
    "скажи",
    "скажите",
    "слышно",
    "случается",
    "случилось",
    "со",
    "события",
    "событий",
    "событие",
    "страна",
    "стране",
    "сутки",
    "суток",
    "также",
    "там",
    "творится",
    "текущие",
    "текущий",
    "текущую",
    "теперь",
    "фон",
    "час",
    "часа",
    "часов",
    "через",
    "что",
    "эта",
    "эти",
    "этих",
    "это",
    "этой",
    "этом",
    "этот",
    "россии",
    "россия",
}

STYLE_SHORT = (
    "кратко",
    "кратенько",
    "коротко",
    "покороче",
    "вкратце",
    "тезисно",
    "сжато",
    "в двух словах",
    "одним предложением",
    "без подробностей",
)

STYLE_DETAILED = (
    "подробно",
    "подробнее",
    "поподробнее",
    "детально",
    "развернуто",
    "развёрнуто",
    "в деталях",
    "с подробностями",
    "обстоятельно",
    "полностью",
    "расширенно",
)

STYLE_RE = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(marker)
        for marker in sorted(STYLE_SHORT + STYLE_DETAILED, key=len, reverse=True)
    )
    + r")\b",
    re.IGNORECASE,
)

_STYLE_SHORT_SET = {marker.lower().replace("ё", "е") for marker in STYLE_SHORT}
_STYLE_DETAILED_SET = {marker.lower().replace("ё", "е") for marker in STYLE_DETAILED}


def answer_style(text: str) -> str:
    """Режим ответа по словам-маркерам: «short», «detailed» или «normal»."""
    style = "normal"
    for match in STYLE_RE.finditer(text or ""):
        marker = match.group(0).lower().replace("ё", "е")
        if marker in _STYLE_DETAILED_SET:
            style = "detailed"
        elif marker in _STYLE_SHORT_SET:
            style = "short"
    return style


def without_style(text: str) -> str:
    """Убирает просьбы о формате ответа из вопроса и поискового запроса."""
    cleaned = STYLE_RE.sub(" ", text or "")
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"^[\s,;:—–-]+", "", cleaned)
    cleaned = re.sub(r"[\s,;:—–-]+$", "", cleaned)
    return cleaned.strip()


NEWS_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s*(.+?)\s*$")

NEWS_NUMBER_RE = re.compile(
    r"(?i)(?:\bновост[ьи]\s*(?:номер\s*)?(\d+)\b|"
    r"\bномер\s*(\d+)\b|"
    r"\b(\d+)\s*(?:-?[яю]|-?ой)?\s*новост)"
)

NEWS_ORDINAL_RE = re.compile(
    r"(?i)\b(?:перв(?:ый|ая|ое|ую|ой|ого|ому|ым|ом)|"
    r"втор(?:ой|ая|ое|ую|ого|ому|ым|ом)|"
    r"трет(?:ий|ья|ье|ью|ьей|ьего|ьему|ьим|ьем)|"
    r"четверт(?:ый|ая|ое|ую|ой|ого|ому|ым|ом)|"
    r"пят(?:ый|ая|ое|ую|ой|ого|ому|ым|ом)|"
    r"шест(?:ой|ая|ое|ую|ого|ому|ым|ом)|"
    r"седьм(?:ой|ая|ое|ую|ого|ому|ым|ом)|"
    r"восьм(?:ой|ая|ое|ую|ого|ому|ым|ом)|"
    r"девят(?:ый|ая|ое|ую|ой|ого|ому|ым|ом)|"
    r"десят(?:ый|ая|ое|ую|ой|ого|ому|ым|ом))\b"
)

NEWS_ORDINAL_STEMS = (
    ("перв", 1),
    ("втор", 2),
    ("трет", 3),
    ("четверт", 4),
    ("пят", 5),
    ("шест", 6),
    ("седьм", 7),
    ("восьм", 8),
    ("девят", 9),
    ("десят", 10),
)

NEWS_TOPIC_STOPWORDS = {
    "расскажи",
    "расскажите",
    "подробнее",
    "подробно",
    "детальнее",
    "детально",
    "что",
    "там",
    "про",
    "об",
    "о",
    "с",
    "со",
    "а",
    "и",
    "но",
    "да",
    "новость",
    "новости",
    "новостях",
    "новостью",
    "новостей",
    "это",
    "этот",
    "эта",
    "эти",
    "этом",
    "этой",
    "этих",
    "мне",
    "пожалуйста",
    "можешь",
    "можете",
    "скажи",
    "скажите",
    "какой",
    "какая",
    "какое",
    "какие",
    "ещё",
    "еще",
    "по",
    "на",
    "в",
    "во",
    "к",
    "ко",
    "из",
    "от",
    "до",
    "за",
    "под",
    "над",
    "при",
    "для",
    "без",
    "через",
    "между",
    "перед",
    "около",
    "после",
    "также",
    "теперь",
    "сейчас",
    "вообще",
    "главное",
    "главный",
    "главная",
    "суть",
    "детали",
    "деталь",
}

EXPLICIT_SEARCH = (
    "поищи",
    "погугли",
    "найди в интернете",
    "посмотри в интернете",
    "поиск в интернете",
)

WEATHER_WORDS = (
    "погода",
    "погоду",
    "погоде",
    "прогноз",
    "температура",
    "градус",
    "ветер",
)

WEATHER_EVENTS = ("дождь", "снег", "гололед")
WEATHER_TIME = ("будет", "сегодня", "завтра", "сейчас", "ожидается")

CITY_PATTERNS = (
    re.compile(r"(?:погода|погоду|прогноз)\s+(?:в|во|на)\s+(.+)$"),
    re.compile(r"(?:в|во)\s+(.+?)\s+(?:погода|погоду|прогноз)$"),
    re.compile(r"(?:погода|погоду|прогноз)\s+(.+)$"),
    re.compile(r"(.+?)\s+(?:погода|погоду|прогноз)$"),
)

TIME_WORDS = {
    "сегодня",
    "завтра",
    "послезавтра",
    "сейчас",
    "неделю",
    "выходные",
}

# Слова, которые не могут быть городом: «погода будет», «какая погода».
CITY_STOP_WORDS = TIME_WORDS | {
    "погода",
    "погоду",
    "погоды",
    "прогноз",
    "будет",
    "будут",
    "ли",
    "какая",
    "какой",
    "какое",
    "какие",
    "что",
    "где",
    "когда",
    "сколько",
    "там",
    "мне",
    "меня",
    "нас",
    "интересно",
    "подскажи",
    "скажи",
    "расскажи",
    "можно",
    "узнать",
    "хочу",
    "хотелось",
    "бы",
    "не",
    "а",
    "и",
    "но",
    "на",
    "за",
    "по",
    "у",
    "в",
    "во",
    "ожидается",
}

TRAILING_TIME = re.compile(
    r"\s+(?:сегодня|завтра|послезавтра|сейчас|на завтра|на сегодня|на неделю)$"
)


def _normalized(text: str) -> str:
    return (text or "").strip().lower().replace("ё", "е")


def _word(value: str) -> str:
    return value.strip(".,!?;:()«»\"'")


def is_question(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    if text.endswith("?"):
        return True
    words = [_word(word) for word in _normalized(text).split()]
    words = [word for word in words if word]
    if not words:
        return False
    first = words[0]
    if first in {"а", "и", "но"} and len(words) > 1:
        first = words[1]
    if first == "мне" and len(words) > 1:
        return words[1] == "интересно"
    if first == "хотелось" and len(words) > 2:
        return words[1] == "бы" and words[2] in {"узнать", "спросить", "уточнить"}
    if first == "можно" and len(words) > 1:
        return words[1] == "ли" or words[1] in {"узнать", "спросить", "уточнить"}
    if first in {"правда", "верно", "есть", "будет"}:
        return len(words) > 1 and words[1] == "ли"
    if first == "интересно" and len(words) > 1:
        return words[1] in QUESTION_STARTS
    if first == "не" and len(words) > 1:
        return words[1] in {"подскажешь", "скажешь", "знаешь"}
    if first in {"можешь", "можете"} and len(words) > 1:
        return words[1] in {"рассказать", "объяснить", "подсказать", "ответить"}
    return first in QUESTION_STARTS


def is_weather_question(text: str) -> bool:
    lowered = _normalized(text)
    if any(word in lowered for word in WEATHER_WORDS):
        return True
    return any(word in lowered for word in WEATHER_EVENTS) and any(
        word in lowered for word in WEATHER_TIME
    )


def is_news_question(text: str) -> bool:
    lowered = _normalized(text)
    return any(marker in lowered for marker in NEWS_MARKERS)


def needs_search(text: str) -> bool:
    lowered = _normalized(text)
    if any(marker in lowered for marker in EXPLICIT_SEARCH):
        return True
    return any(marker in lowered for marker in FRESH_MARKERS)


def _news_only_request(text: str) -> bool:
    """Свободная просьба про новости без конкретной темы.

    «новости за последний день», «что нового за неделю», «что там по
    новостям» — все это общий новостной запрос, а не тема для поиска.
    Временные уточнения отбрасываются: Tavily плохо ищет по ним в news-режиме.
    """
    stripped = NEWS_RELATIVE_TIME_RE.sub(" ", text)
    tokens = re.findall(r"[а-яёa-z0-9]+", stripped)
    return bool(tokens) and all(token in NEWS_FILLER_WORDS for token in tokens)


def search_query(text: str, *, news: bool) -> str:
    """Убирает просьбы о формате и уточняет слишком общий новостной запрос."""
    query = without_style(text)
    if not news:
        return query or text
    lowered = " ".join(_normalized(query).split())
    if NEWS_ONLY.match(lowered) or _news_only_request(lowered):
        return "главные новости России сегодня"
    return query or text


def _looks_like_city(city: str) -> bool:
    words = [_word(part) for part in _normalized(city).split()]
    words = [word for word in words if word]
    return bool(words) and not any(word in CITY_STOP_WORDS for word in words)


def extract_city(text: str) -> str | None:
    text = (text or "").strip()
    for pattern in CITY_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        city = match.group(1).strip(" .,!?:;")
        city = TRAILING_TIME.sub("", city)
        city = re.sub(r"^(?:г\.|город|городе)\s+", "", city, flags=re.IGNORECASE)
        lowered = city.lower()
        if lowered.startswith(("в ", "во ")):
            city = city.split(" ", 1)[1].strip()
            lowered = city.lower()
        if " в " in lowered:
            city = city.rsplit(" в ", 1)[1].strip()
        if city.lower() in TIME_WORDS:
            return None
        if city and _looks_like_city(city):
            return city
    return None


def first_sentence(text: str, limit: int = 250) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    match = re.match(r"(.+?[.!?])(?:\s|$)", text)
    sentence = match.group(1) if match else text
    if len(sentence) > limit:
        sentence = sentence[: limit - 1].rstrip() + "…"
    return sentence


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def _clean_answer(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = CJK_RE.sub("", text)
    text = re.sub(
        r"(?i)^\s*на основе (?:предоставленных |найденных |указанных |этих )?"
        r"(?:источников|данных|материалов)[^:\n]{0,140}:\s*",
        "",
        text,
    )
    text = re.sub(
        r"(?i)^\s*на основе [^,\n]{0,120},\s*"
        r"(?:в эти дни\s+)?(?:были опубликованы|опубликованы)\s+"
        r"(?:следующие\s+)?новости(?:\s+о том)?,?\s*что\s+",
        "",
        text,
    )
    text = re.sub(
        r"(?i)^\s*на основе (?:предоставленных |найденных |указанных |этих )?"
        r"(?:источников|данных|материалов)[,\s]+",
        "",
        text,
    )
    text = re.sub(
        r"(?i)(?:^|(?<=[.!?]\s)|(?<=\]\s))"
        r"(?:в|на) (?:источнике|новости|статье|материале|обзоре)\b"
        r".{0,160}?"
        r"(?:сообщает(?:ся)?|говорит(?:ся)?|упоминает(?:ся)?|"
        r"отмечает(?:ся)?|пишет(?:ся)?|было сообщение|есть информация)"
        r"(?:\s+о том)?,?\s+что\s+",
        "",
        text,
    )
    text = re.sub(r"(?m)^\s*\d+[.)]?\s+(?=\[\d+\])", "", text)
    text = re.sub(r"(?m)^\s*[•—–-]\s+(?=\[\d+\])", "", text)
    text = re.sub(
        r"(?m)^\s*\[(\d+)\]\s*\n\s*(?=[^\s\[])",
        r"[\1] ",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*\[\d+\]\s+)([а-яё])",
        lambda match: match.group(1) + match.group(2).upper(),
        text,
    )
    text = re.sub(
        r"(?i)(?:^|(?<=[.!?]\s)|(?<=\]\s))не знаю[.!?]?\s*",
        "",
        text,
    )
    text = re.sub(
        r"(?i)\s*в (?:представленных |найденных )?источниках "
        r"(?:нет|отсутствует)[^.!?]*[.!?]?",
        "",
        text,
    )
    text = re.sub(r"(?m)^\s*\[\d+\]\s*$", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _has_citation(text: str) -> bool:
    return bool(re.search(r"\[\d+\]", text or ""))


REFUSAL_RE = re.compile(
    r"(?i)\b(?:не содержит|не содержат|не указан[аоы]?|"
    r"нет данных|отсутствует информация|не удалось найти|"
    r"не могу найти|не располагаю|неизвестно)\b"
)

# Ответы, после которых имеет смысл один раз сходить в поиск.
# Модель может сформулировать отказ по-разному, поэтому маркеров несколько.
UNCERTAIN_RE = re.compile(
    r"(?i)(?:"
    r"\bне знаю\b|"
    r"\bне знает\b|"
    r"\bне уверен(?:а|ы)?\b|"
    r"\bне могу (?:ответить|сказать|подтвердить|вспомнить|найти)\b|"
    r"\bне удалось (?:найти|установить|выяснить)\b|"
    r"\bне располагаю\b|"
    r"\bне имею (?:точных )?(?:сведений|данных|информации)\b|"
    r"\bне обладаю (?:такой )?информацией\b|"
    r"\bнет точных (?:сведений|данных|цифр)\b|"
    r"\bнет (?:никакой )?информации\b|"
    r"\bне хватает (?:информации|данных|сведений)\b|"
    r"\bзатрудняюсь (?:ответить|сказать)\b|"
    r"\bмне (?:это )?неизвестно\b|"
    r"\bстоит поискать\b|"
    r"\bпоищи(?:те)? в интернете\b|"
    r"\bлучше (?:уточнить|проверить|посмотреть) в интернете\b"
    r")"
)


def _uncertain_answer(text: str) -> bool:
    """Модель прямо признала, что не знает ответа."""
    return bool(UNCERTAIN_RE.search(text or ""))


def _usable_answer(text: str) -> bool:
    """Отсекает ответы без ссылок, ответы из одних ссылок и отказы модели."""
    if not _has_citation(text):
        return False
    if REFUSAL_RE.search(text):
        return False
    stripped = re.sub(r"\[\d+\]", " ", text)
    stripped = re.sub(r"[^\w]+", " ", stripped, flags=re.UNICODE)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return len(stripped) >= 5


def _clip(text: str, limit: int = 900) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


JUNK_LINES = {
    "видео",
    "фото",
    "назад",
    "дальше",
    "загрузка",
    "все новости",
    "rbc.group",
    "лента",
    "редакция",
    "реклама",
    "контакты",
    "пресс-релизы",
    "техподдержка",
    "спецпроекты",
    "вакансии",
    "rss",
    "правовая информация",
}

JUNK_PREFIXES = (
    "часть первая",
    "часть вторая",
    "часть третья",
    "часть четвертая",
    "часть четвёртая",
    "купить подписку",
    "получать оперативные новости",
    "читайте также",
    "подписывайтесь",
)

RUBRIC_WORDS = (
    "Интернет и СМИ",
    "Бывший СССР",
    "Наука и техника",
    "В мире",
    "Происшествия",
    "Технологии",
    "Политика",
    "Общество",
    "Экономика",
    "Культура",
    "Финансы",
    "Бизнес",
    "Спорт",
    "Россия",
    "Мир",
)

RUBRIC_SPLIT = re.compile(
    r"(?<![.:!?])\s+(?:"
    + "|".join(re.escape(word) for word in RUBRIC_WORDS)
    + r")\s+"
)

RUBRIC_TIME = re.compile(
    r"\s+[А-ЯЁ][а-яё]+,\s*(?:\d{1,2}\s+[а-яё]{3,8},\s*)?\d{1,2}:\d{2}$"
)
INLINE_RUBRIC_TIME = re.compile(
    r"\s+[А-ЯЁ][а-яё]+,\s*(?:\d{1,2}\s+[а-яё]{3,8},\s*)?\d{1,2}:\d{2}\s+"
)
INLINE_DATE_TIME = re.compile(r"\s+\d{1,2}\s+[а-яё]{3,8},\s*\d{1,2}:\d{2}\s+")
LEADING_RUBRIC_TIME = re.compile(
    r"^[А-ЯЁ][а-яё]+,\s*(?:\d{1,2}\s+[а-яё]{3,8},\s*)?\d{1,2}:\d{2}\s+"
)
GLUED_CAPS = re.compile(r"\b([А-ЯЁ]{8,}(?:\s+[А-ЯЁ]{2,})*)\s+(?=[А-ЯЁ][а-яё])")
PHOTO_CREDIT = re.compile(r"\s+Фото:\s*.*?(?=\s+[А-ЯЁ][а-яё]+\s+[а-яё]+)")


def _split_glued(text: str) -> str:
    """Разделяет склеенные заголовки и теги: «ПутинДмитрий» → «Путин. Дмитрий»."""
    text = GLUED_CAPS.sub(r"\1. ", text)
    text = re.sub(r"(?<=[а-яё])«", ". «", text)
    text = re.sub(r"»(?=[А-ЯЁа-яёA-Za-z])", "». ", text)
    text = re.sub(r"(?<=[а-яё])(?=[А-ЯЁ])", ". ", text)
    text = re.sub(r"(?<=[А-ЯЁ])(?=[А-ЯЁ][а-яё])", ". ", text)
    text = PHOTO_CREDIT.sub(". ", text)
    text = INLINE_RUBRIC_TIME.sub(". ", text)
    text = INLINE_DATE_TIME.sub(". ", text)
    return RUBRIC_SPLIT.sub(". ", text)


def _clean_sentence(value: str) -> str:
    value = value.strip(" \t-–—•·|")
    value = LEADING_RUBRIC_TIME.sub("", value)
    value = RUBRIC_TIME.sub("", value)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) < 20:
        return ""
    lowered = value.lower()
    if lowered in JUNK_LINES or lowered.startswith(JUNK_PREFIXES):
        return ""
    words = re.findall(r"[А-Яа-яЁёA-Za-z]+", value)
    if len(words) < 3:
        return ""
    lower_words = sum(1 for word in words if word[0].islower())
    if lower_words < 2:
        return ""
    capitals = sum(1 for word in words if word[0].isupper())
    if len(words) >= 6 and capitals / len(words) > 0.55:
        return ""
    if value.count("|") >= 2:
        return ""
    return value


def _snippet_parts(text: str, limit: int = 3) -> list[str]:
    text = (text or "").replace("\r", "\n")
    text = re.sub(r"#+\s*", "", text)
    text = text.replace("[...]", " ")
    text = re.sub(r"[*_`>]+", " ", text)
    parts: list[str] = []
    for raw_line in text.split("\n"):
        line = " ".join(raw_line.split())
        if not line:
            continue
        line = _split_glued(line)
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = _clean_sentence(sentence)
            if not sentence:
                continue
            parts.append(sentence.rstrip(".!?"))
            if len(parts) >= limit:
                return parts
    return parts


def _clean_snippet(
    text: str, limit: int = 400, title: str = "", sentences: int = 3
) -> str:
    if title:
        parts = [title]
        for separator in (" - ", " — ", " – ", " | "):
            if separator in title:
                parts.extend(title.split(separator))
        for part in sorted({part.strip() for part in parts}, key=len, reverse=True):
            if len(part) >= 10:
                text = re.sub(re.escape(part), " ", text or "", flags=re.IGNORECASE)
    parts = _snippet_parts(text, limit=max(1, sentences))
    if not parts:
        return ""
    return _clip(". ".join(parts), limit)


def _snippets_text(
    results: list[SearchResult], limit: int = 3, sentences: int = 3
) -> str:
    parts = []
    for index, result in enumerate(results[:limit], 1):
        snippet = _clean_snippet(
            result.content, title=result.title, sentences=sentences
        )
        block = f"[{index}] {result.title}"
        if snippet:
            block += f"\n{snippet}"
        parts.append(block)
    return "\n\n".join(parts)


def _looks_stale(text: str) -> bool:
    """Отсекает сюжеты, где все упомянутые годы старше прошлого."""
    years = [int(year) for year in re.findall(r"\b(20\d{2})\b", text or "")]
    if not years:
        return False
    return max(years) < datetime.now().year - 1


def _title_only(fact: str, title: str) -> bool:
    """Пункт сводки — только заголовок источника, без самого факта."""
    fact_key = re.sub(r"[^\w]+", " ", (fact or "").lower()).strip()
    title_key = re.sub(r"[^\w]+", " ", (title or "").lower()).strip()
    return bool(fact_key) and fact_key == title_key


def _context_title(result: SearchResult) -> str:
    """Для служебных заголовков лент берёт первый факт из сниппета."""
    if not GENERIC_NEWS_TITLE_RE.search(result.title):
        return result.title
    snippet = _clean_snippet(result.content, limit=160, title=result.title)
    return snippet or result.title


@dataclass(frozen=True)
class QASource:
    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True)
class NewsItem:
    index: int
    fact: str
    title: str
    url: str
    snippet: str = ""


def _news_reference(text: str) -> int | None:
    """Номер новости из фразы: «первая», «про 2-ю», «новость номер 3»."""
    lowered = _normalized(text)
    match = NEWS_NUMBER_RE.search(lowered)
    if match:
        value = next((group for group in match.groups() if group), "")
        if value.isdigit():
            number = int(value)
            if 1 <= number <= 10:
                return number
    match = NEWS_ORDINAL_RE.search(lowered)
    if match:
        word = match.group(0)
        for stem, number in NEWS_ORDINAL_STEMS:
            if word.startswith(stem):
                return number
    return None


def _topic_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[а-яёa-z0-9]+", _normalized(text))
    return [
        token
        for token in tokens
        if len(token) >= 3 and token not in NEWS_TOPIC_STOPWORDS
    ]


def _tokens_match(left: str, right: str) -> bool:
    limit = min(5, len(left), len(right))
    return limit >= 3 and left[:limit] == right[:limit]


def _news_score(text: str, item: NewsItem) -> int:
    query_tokens = _topic_tokens(text)
    if not query_tokens:
        return 0
    item_tokens = _topic_tokens(f"{item.fact} {item.title}")
    return sum(
        1
        for query_token in query_tokens
        if any(_tokens_match(query_token, item_token) for item_token in item_tokens)
    )


NEWS_DETAIL_RE = re.compile(
    r"(?i)\b(?:подробн\w*|детальн\w*|развернут\w*|расскаж\w*|"
    r"уточн\w*|что\s+(?:там\s+)?(?:с|по))\b"
)

NEWS_REQUEST_RE = re.compile(
    r"(?i)\b(?:новости|нового|новое|случилось|происходит|последние|свежие)\b"
)

GENERIC_NEWS_TITLE_RE = re.compile(
    r"(?i)^(?:выпуск за|главные новости|лента новостей|последние новости|"
    r"новости за|смотреть онлайн)"
)

# Уточнения к прошлому ответу. Без контекста такие фразы бессмысленны:
# «а завтра?» после погоды, «а почему?» после объяснения.
WEATHER_DAY_PARTS = (
    ("на послезавтра", "day_after"),
    ("послезавтра", "day_after"),
    ("на завтра", "tomorrow"),
    ("завтра", "tomorrow"),
    ("сегодня", "today"),
    ("на сегодня", "today"),
    ("сейчас", "current"),
    ("теперь", "current"),
)

WEATHER_FOLLOWUP_RE = re.compile(
    r"^(?:"
    r"(?:завтра|послезавтра|сегодня|сейчас|теперь|на\s+завтра|на\s+послезавтра)"
    r"(?:\s+в\s+.+)?"
    r"|в\s+.+"
    r"|какая\s+погода|какой\s+прогноз|что\s+с\s+погодой"
    r"|там"
    r")$"
)

GENERAL_FOLLOWUP_RE = re.compile(
    r"^(?:"
    r"(?:почему|зачем|как|когда|где|куда|откуда|сколько)"
    r"(?:\s+(?:это|так|там|же|всё|все))?"
    r"(?:\s+(?:важно|нужно|работает|происходит|будет|было|можно|стоит|"
    r"значит|получается|так|сделать|произошло|случилось|называется))?"
    r"|(?:подробнее|поподробнее|детальнее|детально|развернуто|развёрнуто|"
    r"кратко|коротко|короче|сжато|вкратце)"
    r"|(?:что|чего)\s+(?:там|ещё|еще|дальше)"
    r"|(?:и\s+)?что\s+(?:это|он|она|они)\s+(?:значит|означает)"
    r"|что\s+(?:это|это\s+такое)"
    r"|кто\s+(?:это|он|она|они)"
    r"|(?:расскажи|объясни|уточни)\s+(?:подробнее|почему|как|что)"
    r")$"
)

CITY_FOLLOWUP_RE = re.compile(
    r"(?:^|\s)(?:в|во)\s+([А-ЯЁа-яё][А-ЯЁа-яё\- ]{1,40})$",
    re.IGNORECASE,
)


def _followup_core(text: str) -> str:
    """Убирает вводное «а/и/но» и знаки, чтобы сравнить с шаблоном."""
    core = _normalized(text).strip(" ?!.,;:—-")
    core = re.sub(r"^(?:а|и|но)\s+", "", core)
    return core.strip(" ?!.,;:—-")


def _city_from_followup(text: str) -> str | None:
    match = CITY_FOLLOWUP_RE.search(text.strip(" ?!.,;:—-"))
    if not match:
        return None
    city = TRAILING_TIME.sub("", match.group(1)).strip()
    return city if _looks_like_city(city) else None


def _select_news_item(
    text: str, items: list[NewsItem]
) -> tuple[NewsItem | None, bool]:
    """Выбирает новость по номеру или ключевым словам.

    Второе значение — неоднозначность: несколько новостей подходят одинаково.
    """
    if not items:
        return None, False
    reference = _news_reference(text)
    if reference is not None:
        if 1 <= reference <= len(items):
            return items[reference - 1], False
        return None, False
    scores = [_news_score(text, item) for item in items]
    best = max(scores)
    if best <= 0:
        return None, False
    winners = [
        item
        for item, score in zip(items, scores, strict=True)
        if score == best
    ]
    if len(winners) > 1:
        return None, True
    return winners[0], False


def _prepare_news_digest(
    answer: str, results: list[SearchResult]
) -> tuple[str, list[QASource], list[NewsItem]]:
    """Перенумеровывает пункты сводки и запоминает их для уточнений."""
    items: list[NewsItem] = []
    sources: list[QASource] = []
    lines: list[str] = []
    for line in answer.splitlines():
        match = NEWS_LINE_RE.match(line)
        if not match:
            lines.append(line)
            continue
        source_index = int(match.group(1))
        if not 1 <= source_index <= len(results):
            lines.append(line)
            continue
        result = results[source_index - 1]
        fact = match.group(2).strip()
        if _looks_stale(fact):
            continue
        if _title_only(fact, result.title):
            continue
        display_index = len(items) + 1
        title = result.title
        if GENERIC_NEWS_TITLE_RE.search(title):
            title = first_sentence(fact, limit=160) or fact
        items.append(
            NewsItem(
                display_index,
                fact,
                title,
                result.url,
                result.content,
            )
        )
        sources.append(QASource(title, result.url, result.content))
        lines.append(f"[{display_index}] {fact}")
    if not items:
        return answer, [QASource(r.title, r.url, r.content) for r in results], []
    return "\n".join(lines), sources, items


def _news_items_from_results(
    results: list[SearchResult], limit: int
) -> list[NewsItem]:
    items: list[NewsItem] = []
    for result in results[:limit]:
        snippet = _clean_snippet(result.content, title=result.title)
        fact = first_sentence(snippet) or result.title
        if _looks_stale(fact):
            continue
        title = result.title
        if GENERIC_NEWS_TITLE_RE.search(title):
            title = fact
        items.append(
            NewsItem(len(items) + 1, fact, title, result.url, result.content)
        )
    return items


@dataclass
class QAAnswer:
    ok: bool
    question: str
    short: str
    full: str
    kind: str
    sources: list[QASource] = field(default_factory=list)
    error: str = ""


@dataclass(frozen=True)
class FollowUp:
    """Разобранное уточнение к прошлому ответу."""

    kind: str  # weather или general
    style: str = "normal"
    city: str = ""
    weather_part: str = "full"


@dataclass
class DialogTurn:
    """Последняя успешная пара «вопрос — ответ» для уточнений."""

    question: str
    answer: str
    kind: str
    city: str = ""
    root: str = ""
    at: float = 0.0


class QAService:
    """Вопросы: погода через Open-Meteo, свежее через Tavily, остальное — LLM."""

    def __init__(
        self,
        cfg: dict[str, Any],
        llm: CloudLLM | None = None,
        browser: Any = None,
    ) -> None:
        self.cfg = cfg or {}
        section = self.cfg.get("qa") or {}
        self.enabled = bool(section.get("enabled", True))
        self.weather = WeatherClient(self.cfg)
        self.search = TavilySearch(self.cfg)
        self.llm = llm
        self.browser = browser
        answer_cfg = section.get("answer") or {}
        self.max_tokens = int(answer_cfg.get("max_tokens") or 400)
        self.temperature = float(answer_cfg.get("temperature") or 0.2)
        self.timeout = float(answer_cfg.get("timeout_sec") or 45)
        self.news_ttl = float(answer_cfg.get("news_ttl_sec") or 900)
        self.context_ttl = max(
            0.0, float(answer_cfg.get("context_ttl_sec") or 180)
        )
        self.search_fallback = bool(answer_cfg.get("search_fallback", True))
        self.last_news: list[NewsItem] = []
        self._news_at = 0.0
        self._last_turn: DialogTurn | None = None
        self._last_weather_city = ""

    # ------------------------------------------------------------------ public
    def handles(self, text: str) -> bool:
        if not self.enabled:
            return False
        return (
            is_weather_question(text)
            or is_question(text)
            or needs_search(text)
            or self._is_news_followup(text)
            or self._followup(text) is not None
        )

    def kind_of(self, text: str) -> str:
        """Категория фразы-подводки: weather, news или wait."""
        if is_weather_question(text):
            return "weather"
        if is_news_question(text) or self._is_news_followup(text):
            return "news"
        followup = self._followup(text)
        if followup is not None and followup.kind == "weather":
            return "weather"
        return "wait"

    def answer(self, text: str) -> QAAnswer:
        text = (text or "").strip()
        if not text:
            return self._error(text, "Пустой вопрос", "пустой вопрос")

        style = answer_style(text)

        if self._is_news_followup(text):
            answer = self._answer_news_detail(text, style=style)
            self._remember(text, answer)
            return answer

        followup = self._followup(text)
        if followup is not None:
            answer = self._answer_followup(text, followup)
            self._remember(text, answer, followup=followup)
            return answer

        answer = self._answer_question(text, style=style)
        self._remember(text, answer)
        return answer

    def _answer_question(self, text: str, *, style: str) -> QAAnswer:
        if is_weather_question(text):
            return self._answer_weather(text)

        if needs_search(text):
            if not self.search.available():
                return self._error(
                    text,
                    "Нужен ключ Tavily",
                    "Для свежих данных добавьте qa.search.api_key в config.json.",
                    error="нет ключа Tavily",
                )
            return self._answer_search(
                text, news=is_news_question(text), style=style
            )

        if self._llm_available():
            return self._answer_general(text, style=style)

        if self.search.available():
            return self._answer_search(text, news=False, style=style)

        return self._error(
            text,
            "Не могу ответить",
            "Модель недоступна, ключ поиска не задан.",
            error="нет доступных источников ответа",
        )

    def render_html(self, answer: QAAnswer) -> str:
        kind = {
            "weather": "Погода",
            "search": "Поиск в интернете",
            "general": "Модель",
        }.get(answer.kind, "Ответ")
        body = "".join(
            f"<p>{html.escape(line.strip())}</p>"
            for line in answer.full.splitlines()
            if line.strip()
        )
        sources = ""
        if answer.sources:
            items = "".join(
                f'<li><a href="{html.escape(source.url, quote=True)}" '
                f'target="_blank" rel="noopener">{html.escape(source.title)}</a></li>'
                for source in answer.sources
            )
            sources = f"<h2>Источники</h2><ol>{items}</ol>"
        generated = datetime.now().strftime("%d.%m.%Y %H:%M")
        return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ответ ассистента</title>
<style>
:root {{ color-scheme: light dark; }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 32px 16px;
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f6f7f9; color: #1d1f23;
}}
main {{
  max-width: 760px; margin: 0 auto; padding: 28px 32px;
  background: #fff; border-radius: 16px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, .08);
}}
.badge {{
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  background: #e8f0fe; color: #1a56db; font-size: 13px; font-weight: 600;
}}
h1 {{ font-size: 24px; line-height: 1.3; margin: 14px 0 18px; }}
h2 {{ font-size: 16px; margin: 26px 0 10px; }}
p {{ margin: 0 0 14px; }}
ol {{ padding-left: 22px; }}
li {{ margin-bottom: 8px; }}
a {{ color: #1a56db; word-break: break-word; }}
footer {{ margin-top: 28px; color: #6b7280; font-size: 13px; }}
@media (prefers-color-scheme: dark) {{
  body {{ background: #111318; color: #e6e8eb; }}
  main {{ background: #1a1d23; box-shadow: none; }}
  .badge {{ background: #1e3a8a; color: #bfdbfe; }}
  a {{ color: #93c5fd; }}
  footer {{ color: #9ca3af; }}
}}
</style>
</head>
<body>
<main>
<span class="badge">{html.escape(kind)}</span>
<h1>{html.escape(answer.question)}</h1>
{body}
{sources}
<footer>Сгенерировано {generated}</footer>
</main>
</body>
</html>"""

    def show(self, answer: QAAnswer) -> bool:
        if self.browser is None or not answer.ok:
            return False
        try:
            self.browser.call("render", html=self.render_html(answer))
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Q&A: не удалось открыть вкладку: %s", exc)
            return False

    # ---------------------------------------------------------------- private
    def _llm_available(self) -> bool:
        if self.llm is None:
            return False
        try:
            return bool(self.llm.available())
        except Exception:  # noqa: BLE001
            return False

    def _error(
        self,
        question: str,
        short: str,
        full: str,
        *,
        kind: str = "error",
        error: str = "",
    ) -> QAAnswer:
        return QAAnswer(False, question, short, full, kind, error=error or full)

    def _last_news_valid(self) -> bool:
        return bool(self.last_news) and (
            time.monotonic() - self._news_at <= self.news_ttl
        )

    def _remember_news(self, items: list[NewsItem]) -> None:
        if not items:
            return
        self.last_news = list(items)
        self._news_at = time.monotonic()

    def _is_news_followup(self, text: str) -> bool:
        if not self._last_news_valid():
            return False
        lowered = _normalized(text)
        if not NEWS_DETAIL_RE.search(lowered):
            return False
        if _news_reference(text) is not None:
            return True
        if NEWS_REQUEST_RE.search(lowered):
            return False
        if max((_news_score(text, item) for item in self.last_news), default=0) > 0:
            return True
        return not _topic_tokens(text)

    def _followup(self, text: str) -> FollowUp | None:
        turn = self._last_turn
        if turn is None:
            return None
        if time.monotonic() - turn.at > self.context_ttl:
            self._last_turn = None
            return None
        style = answer_style(text)
        if turn.kind == "weather":
            weather = self._weather_followup(text, turn, style=style)
            if weather is not None:
                return weather
        if turn.kind in {"general", "search"}:
            return self._general_followup(text, style=style)
        return None

    def _weather_followup(
        self, text: str, turn: DialogTurn, *, style: str
    ) -> FollowUp | None:
        core = _followup_core(text)
        part = "full"
        for marker, name in WEATHER_DAY_PARTS:
            if re.search(rf"(?:^|\s){re.escape(marker)}(?:$|\s)", core):
                part = name
                break
        city = _city_from_followup(text) or ""
        if part == "full" and not city:
            if not WEATHER_FOLLOWUP_RE.match(core):
                return None
        return FollowUp(
            "weather",
            style=style,
            city=city or turn.city or self._last_weather_city,
            weather_part=part,
        )

    def _general_followup(self, text: str, *, style: str) -> FollowUp | None:
        core = _followup_core(text)
        if not GENERAL_FOLLOWUP_RE.match(core):
            return None
        return FollowUp("general", style=style)

    def _answer_followup(self, text: str, followup: FollowUp) -> QAAnswer:
        if followup.kind == "weather":
            return self._answer_weather(
                text,
                city=followup.city or None,
                part=followup.weather_part,
            )
        if self._llm_available():
            return self._answer_general(
                text,
                style=followup.style,
                history=self._history_text(),
                followup_query=self._followup_question(text),
            )
        if self.search.available():
            return self._answer_search(
                self._followup_question(text),
                news=False,
                style=followup.style,
            )
        return self._error(
            text,
            "Не могу ответить",
            "Модель недоступна, ключ поиска не задан.",
            error="нет доступных источников ответа",
        )

    def _followup_question(self, text: str) -> str:
        turn = self._last_turn
        if turn is None:
            return text
        return f"{turn.question} {without_style(text) or text}".strip()

    def _history_text(self) -> str:
        turn = self._last_turn
        if turn is None:
            return ""
        lines = [
            f"Предыдущий вопрос: {turn.question}",
            f"Предыдущий ответ: {_clip(turn.answer, 1200)}",
        ]
        if turn.root and turn.root != turn.question:
            lines.append(f"Тема разговора: {turn.root}")
        return "\n".join(lines)

    def _remember(
        self,
        text: str,
        answer: QAAnswer,
        *,
        followup: FollowUp | None = None,
    ) -> None:
        if not answer.ok or self.context_ttl <= 0:
            return
        question = without_style(text) or text
        root = ""
        if followup is not None and self._last_turn is not None:
            root = self._last_turn.root or self._last_turn.question
        self._last_turn = DialogTurn(
            question=question,
            answer=_clip(answer.full, 1200),
            kind=answer.kind,
            city=self._last_weather_city if answer.kind == "weather" else "",
            root=root,
            at=time.monotonic(),
        )

    def _news_sources(self) -> list[QASource]:
        return [
            QASource(item.title, item.url, item.snippet) for item in self.last_news
        ]

    def _news_clarification(self, text: str, message: str) -> QAAnswer:
        lines = "\n".join(
            f"[{item.index}] {item.fact}" for item in self.last_news
        )
        return QAAnswer(
            True,
            text,
            message,
            f"{message}\n\n{lines}",
            "search",
            self._news_sources(),
        )

    def _answer_news_detail(
        self, text: str, *, style: str = "normal"
    ) -> QAAnswer:
        item, ambiguous = _select_news_item(text, self.last_news)
        if item is None:
            message = (
                "Под несколько новостей подходит, уточните номер."
                if ambiguous
                else "Не понял, о какой новости речь. Назовите номер."
            )
            return self._news_clarification(text, message)

        related: list[SearchResult] = []
        query = first_sentence(item.fact, limit=200) or item.title
        try:
            related = self.search.search_article(query, limit=3)
        except SearchError as exc:
            log.info("Q&A: целевой поиск по новости не удался: %s", exc)

        article = ""
        if not related:
            try:
                article = self.search.extract(item.url)
            except SearchError as exc:
                log.info("Q&A: полный текст статьи недоступен: %s", exc)
            article = _clean_snippet(
                article, limit=2000, title=item.title, sentences=12
            )
            if article and _news_score(article, item) < 4:
                article = ""

        blocks = [f"[1] {item.title}\n{item.url}\n{item.fact}"]
        if article:
            blocks[0] += f"\n{article}"
        sources = [QASource(item.title, item.url, item.snippet)]
        for result in related:
            snippet = _clean_snippet(
                result.content, limit=1200, title=result.title, sentences=10
            )
            if not snippet:
                continue
            number = len(blocks) + 1
            blocks.append(f"[{number}] {result.title}\n{result.url}\n{snippet}")
            sources.append(QASource(result.title, result.url, result.content))
            if len(blocks) >= 3:
                break
        context = "\n\n".join(blocks)

        max_tokens = min(self.max_tokens, 300) if style == "short" else max(
            self.max_tokens, 700
        )
        answer = ""
        if self._llm_available() and self.llm is not None:
            try:
                answer = self.llm.answer(
                    without_style(text) or text,
                    context,
                    style="article",
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    timeout=max(self.timeout, 90),
                )
            except LLMError as exc:
                log.warning("Q&A: модель не ответила по статье: %s", exc)
        answer = _clean_answer(answer)
        if answer and _usable_answer(answer):
            full = f"[{item.index}] {item.title}\n\n{answer}"
            return QAAnswer(
                True,
                text,
                first_sentence(answer),
                full,
                "search",
                sources,
            )

        best = ""
        if related:
            best = _clean_snippet(
                related[0].content,
                limit=1500,
                title=related[0].title,
                sentences=10,
            )
        if not best:
            best = _clean_snippet(
                item.snippet, limit=1200, title=item.title, sentences=10
            )
        if not best:
            best = item.fact
        full = f"[{item.index}] {item.title}\n\n{item.fact}"
        if best and best != item.fact:
            full += f"\n\n{best}"
        return QAAnswer(
            True,
            text,
            first_sentence(best) or item.fact,
            full,
            "search",
            sources,
        )

    def _answer_weather(
        self,
        text: str,
        *,
        city: str | None = None,
        part: str = "full",
    ) -> QAAnswer:
        city = (city or "").strip() or extract_city(text)
        try:
            report = self.weather.forecast(city)
        except WeatherError as exc:
            return self._error(
                text,
                f"Не смог узнать погоду: {exc}",
                f"Не смог узнать погоду: {exc}",
                kind="weather",
                error=str(exc),
            )
        self._last_weather_city = report.place.name
        if part == "current":
            full = format_current(report)
        elif part in {"today", "tomorrow", "day_after"}:
            index = {"today": 0, "tomorrow": 1, "day_after": 2}[part]
            full = format_day(report, index)
        else:
            full = format_report(report)
        source = QASource(
            "Open-Meteo",
            "https://open-meteo.com/",
            "Данные погоды Open-Meteo",
        )
        return QAAnswer(
            True,
            text,
            first_sentence(full),
            full,
            "weather",
            [source],
        )

    def _answer_general(
        self,
        text: str,
        *,
        style: str = "normal",
        history: str = "",
        followup_query: str = "",
    ) -> QAAnswer:
        if self.llm is None:
            return self._error(text, "Модель недоступна", "Модель недоступна.")
        style = style if style in {"short", "detailed"} else "normal"
        max_tokens = self.max_tokens
        if style == "short":
            max_tokens = min(max_tokens, 220)
        elif style == "detailed":
            max_tokens = max(max_tokens, 600)
        try:
            raw = self.llm.answer(
                without_style(text) or text,
                style=style,
                max_tokens=max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
                history=history,
            )
        except LLMError as exc:
            return self._error(
                text,
                f"Модель: {exc}",
                f"Модель: {exc}",
                error=str(exc),
            )
        uncertain = _uncertain_answer(raw)
        answer = _clean_answer(raw)
        if uncertain:
            found = self._search_fallback(
                text,
                style=style,
                query=followup_query or without_style(text) or text,
                history=history,
            )
            if found is not None:
                return found
            message = "Не знаю, стоит поискать в интернете."
            return QAAnswer(True, text, message, message, "general")
        if not answer:
            return self._error(text, "Пустой ответ модели", "Модель вернула пустой ответ.")
        return QAAnswer(True, text, first_sentence(answer), answer, "general")

    def _search_fallback(
        self,
        text: str,
        *,
        style: str,
        query: str,
        history: str = "",
    ) -> QAAnswer | None:
        """Один дополнительный поиск, если модель признала, что не знает."""
        if not self.search_fallback or not self.search.available():
            return None
        log.info("Q&A: модель не уверена, ищу в интернете: %s", query)
        found = self._answer_search(
            text,
            news=False,
            style=style,
            query=query,
            history=history,
        )
        return found if found.ok else None

    def _answer_search(
        self,
        text: str,
        *,
        news: bool,
        style: str = "normal",
        query: str | None = None,
        history: str = "",
    ) -> QAAnswer:
        search_text = search_query(query or text, news=news)
        if news and style == "short":
            effective = "news_short"
            source_limit, snippet_limit = 7, 500
            max_tokens = max(self.max_tokens, 800)
        elif news and style == "detailed":
            effective = "news_detailed"
            source_limit, snippet_limit = 6, 700
            max_tokens = max(self.max_tokens, 900)
        elif news:
            effective = "news"
            source_limit, snippet_limit = 6, 600
            max_tokens = max(self.max_tokens, 800)
        elif style == "short":
            effective = "short"
            source_limit, snippet_limit = 3, 400
            max_tokens = min(self.max_tokens, 220)
        elif style == "detailed":
            effective = "detailed"
            source_limit, snippet_limit = 5, 600
            max_tokens = max(self.max_tokens, 600)
        else:
            effective = "normal"
            source_limit, snippet_limit = 4, 500
            max_tokens = self.max_tokens
        try:
            results = self.search.search(search_text, news=news)
        except SearchError as exc:
            return self._error(
                text,
                f"Поиск не удался: {exc}",
                f"Поиск не удался: {exc}",
                kind="search",
                error=str(exc),
            )
        if not results:
            return self._error(
                text,
                "Ничего не нашёл",
                "По запросу не нашлось результатов.",
                kind="search",
                error="пустая выдача",
            )
        if news:
            fresh = [
                result
                for result in results
                if not _looks_stale(
                    _clean_snippet(result.content, title=result.title)
                )
            ]
            if fresh:
                results = fresh

        sources = [
            QASource(result.title, result.url, result.content) for result in results
        ]
        answer = ""
        if self._llm_available() and self.llm is not None:
            blocks = []
            for index, result in enumerate(results[:source_limit], 1):
                snippet = _clean_snippet(
                    result.content, limit=snippet_limit, title=result.title
                )
                if not snippet:
                    continue
                blocks.append(
                    f"[{index}] {_context_title(result)}\n{result.url}\n{snippet}"
                )
            context = "\n\n".join(blocks)
            llm_timeout = max(self.timeout, 90) if news else self.timeout
            try:
                answer = self.llm.answer(
                    without_style(text) or text,
                    context,
                    style=effective,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    timeout=llm_timeout,
                    history=history,
                )
            except LLMError as exc:
                log.warning("Q&A: модель не ответила: %s", exc)
        answer = _clean_answer(answer)

        if answer and _usable_answer(answer):
            if news:
                answer, news_sources, items = _prepare_news_digest(
                    answer, results[:source_limit]
                )
                self._remember_news(
                    items or _news_items_from_results(results, source_limit)
                )
                sources = news_sources or sources
            return QAAnswer(
                True,
                text,
                first_sentence(answer),
                answer,
                "search",
                sources,
            )

        note = ""
        if answer:
            note = "Модель не смогла пересказать источники. Вот главное из найденного:\n\n"
        else:
            note = "Вот главное из найденного:\n\n"
        if effective == "news_short":
            fallback_limit, fallback_sentences = 6, 2
        elif effective in {"news", "news_detailed"}:
            fallback_limit, fallback_sentences = 5, 4
        elif effective == "short":
            fallback_limit, fallback_sentences = 3, 3
        else:
            fallback_limit, fallback_sentences = 4, 3
        if news:
            self._remember_news(_news_items_from_results(results, fallback_limit))
        full = note + _snippets_text(
            results, limit=fallback_limit, sentences=fallback_sentences
        )
        short = (
            first_sentence(
                _clean_snippet(results[0].content, title=results[0].title)
            )
            or results[0].title
        )
        return QAAnswer(
            True,
            text,
            short,
            full,
            "search",
            sources,
        )
