from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .local_llm import LocalLLM, LocalLLMError
from .search import SearchError, SearchResult, TavilySearch
from .weather import WeatherClient, WeatherError, format_report

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


def search_query(text: str, *, news: bool) -> str:
    """Уточняет слишком общий новостной запрос вроде «какие новости»."""
    if not news:
        return text
    lowered = " ".join(_normalized(text).split())
    if NEWS_ONLY.match(lowered):
        return "главные новости России сегодня"
    return text


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
        r"(?i)(?:^|(?<=[.!?]\s)|(?<=\]\s))"
        r"(?:в|на) (?:источнике|новости|статье|материале|обзоре)\b"
        r"[^.!?]{0,160}?"
        r"(?:сообщается|говорится|упоминается|отмечается|пишется|"
        r"было сообщение|есть информация)(?:\s+о том)?,?\s+что\s+",
        "",
        text,
    )
    text = re.sub(
        r"(\[\d+\]\s+)([а-яё])",
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


def _usable_answer(text: str) -> bool:
    """Отсекает ответы без ссылок и ответы из одних ссылок вроде «[1] [2]»."""
    if not _has_citation(text):
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


def _clean_snippet(text: str, limit: int = 400, title: str = "") -> str:
    if title:
        text = re.sub(re.escape(title), " ", text or "", flags=re.IGNORECASE)
    parts = _snippet_parts(text)
    if not parts:
        return ""
    return _clip(". ".join(parts), limit)


def _snippets_text(results: list[SearchResult], limit: int = 3) -> str:
    parts = []
    for index, result in enumerate(results[:limit], 1):
        snippet = _clean_snippet(result.content, title=result.title)
        block = f"[{index}] {result.title}"
        if snippet:
            block += f"\n{snippet}"
        parts.append(block)
    return "\n\n".join(parts)


@dataclass(frozen=True)
class QASource:
    title: str
    url: str
    snippet: str = ""


@dataclass
class QAAnswer:
    ok: bool
    question: str
    short: str
    full: str
    kind: str
    sources: list[QASource] = field(default_factory=list)
    error: str = ""


class QAService:
    """Вопросы: погода через Open-Meteo, свежее через Tavily, остальное — LLM."""

    def __init__(
        self,
        cfg: dict[str, Any],
        llm: LocalLLM | None = None,
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

    # ------------------------------------------------------------------ public
    def handles(self, text: str) -> bool:
        if not self.enabled:
            return False
        return (
            is_weather_question(text)
            or is_question(text)
            or needs_search(text)
        )

    def answer(self, text: str) -> QAAnswer:
        text = (text or "").strip()
        if not text:
            return self._error(text, "Пустой вопрос", "пустой вопрос")

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
            return self._answer_search(text, news=is_news_question(text))

        if self._llm_available():
            return self._answer_local(text)

        if self.search.available():
            return self._answer_search(text, news=False)

        return self._error(
            text,
            "Не могу ответить",
            "Локальная LLM недоступна, ключ поиска не задан.",
            error="нет доступных источников ответа",
        )

    def render_html(self, answer: QAAnswer) -> str:
        kind = {
            "weather": "Погода",
            "search": "Поиск в интернете",
            "local": "Локальная модель",
        }.get(answer.kind, "Ответ")
        body = "".join(
            f"<p>{html.escape(paragraph)}</p>"
            for paragraph in answer.full.split("\n\n")
            if paragraph.strip()
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

    def _answer_weather(self, text: str) -> QAAnswer:
        city = extract_city(text)
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

    def _answer_local(self, text: str) -> QAAnswer:
        if self.llm is None:
            return self._error(text, "Локальная LLM недоступна", "Локальная LLM недоступна.")
        try:
            answer = self.llm.answer(
                text,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
            )
        except LocalLLMError as exc:
            return self._error(
                text,
                f"Локальная LLM: {exc}",
                f"Локальная LLM: {exc}",
                error=str(exc),
            )
        answer = _clean_answer(answer)
        if not answer:
            return self._error(text, "Пустой ответ модели", "Модель вернула пустой ответ.")
        return QAAnswer(True, text, first_sentence(answer), answer, "local")

    def _answer_search(self, text: str, *, news: bool) -> QAAnswer:
        query = search_query(text, news=news)
        try:
            results = self.search.search(query, news=news)
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

        sources = [
            QASource(result.title, result.url, result.content) for result in results
        ]
        answer = ""
        if self._llm_available() and self.llm is not None:
            blocks = []
            for index, result in enumerate(results[:4], 1):
                snippet = _clean_snippet(
                    result.content, limit=500, title=result.title
                )
                if not snippet:
                    continue
                blocks.append(f"[{index}] {result.title}\n{result.url}\n{snippet}")
            context = "\n\n".join(blocks)
            try:
                answer = self.llm.answer(
                    text,
                    context,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout,
                )
            except LocalLLMError as exc:
                log.warning("Q&A: локальная LLM не ответила: %s", exc)
        answer = _clean_answer(answer)

        if answer and _usable_answer(answer):
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
        full = note + _snippets_text(results)
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
