from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

DEFAULT_TURN_TTL_SEC = 180.0
DEFAULT_NEWS_TTL_SEC = 900.0


@dataclass(frozen=True)
class NewsItem:
    index: int
    fact: str
    title: str
    url: str
    snippet: str = ""


@dataclass
class DialogTurn:
    """Последняя успешная пара «вопрос — ответ» для уточнений."""

    question: str
    answer: str
    kind: str
    city: str = ""
    root: str = ""
    at: float = 0.0


def clip_text(text: str, limit: int = 900) -> str:
    """Схлопывает пробелы и обрезает длинный текст многоточием."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


class DialogContext:
    """Состояние диалога без ввода-вывода: последний обмен, новости, город.

    Всё состояние живёт в одном месте, чтобы правила TTL и очистки не
    расползались по сервису. Часы инжектируются, поэтому истечение
    контекста проверяется в тестах без ожидания и подмены глобального времени.
    """

    def __init__(
        self,
        turn_ttl_sec: float = DEFAULT_TURN_TTL_SEC,
        news_ttl_sec: float = DEFAULT_NEWS_TTL_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.turn_ttl = max(0.0, float(turn_ttl_sec))
        self.news_ttl = max(0.0, float(news_ttl_sec))
        self._clock = clock
        self._turn: DialogTurn | None = None
        self._news: list[NewsItem] = []
        self._news_at = 0.0
        self.weather_city = ""

    # ------------------------------------------------------------------ turn
    def turn(self) -> DialogTurn | None:
        """Последний обмен, если он ещё не истёк."""
        if self._turn is None:
            return None
        if self._clock() - self._turn.at > self.turn_ttl:
            self._turn = None
            return None
        return self._turn

    def remember_turn(
        self,
        question: str,
        answer: str,
        kind: str,
        *,
        city: str = "",
        followup: bool = False,
    ) -> DialogTurn | None:
        """Запоминает успешный обмен. Уточнение сохраняет исходную тему."""
        if self.turn_ttl <= 0:
            return None
        root = ""
        if followup:
            previous = self.turn()
            if previous is not None:
                root = previous.root or previous.question
        self._turn = DialogTurn(
            question=question,
            answer=clip_text(answer, 1200),
            kind=kind,
            city=city,
            root=root,
            at=self._clock(),
        )
        return self._turn

    def history(self) -> str:
        """Прошлый обмен в виде текста для промпта модели."""
        turn = self.turn()
        if turn is None:
            return ""
        lines = [
            f"Предыдущий вопрос: {turn.question}",
            f"Предыдущий ответ: {clip_text(turn.answer, 1200)}",
        ]
        if turn.root and turn.root != turn.question:
            lines.append(f"Тема разговора: {turn.root}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ news
    def news(self) -> list[NewsItem]:
        """Сводка новостей, если её TTL ещё не истёк."""
        if not self._news:
            return []
        if self._clock() - self._news_at > self.news_ttl:
            self._news = []
            return []
        return list(self._news)

    def remember_news(self, items: Iterable[NewsItem]) -> None:
        """Пустая выдача не должна затирать прошлую сводку."""
        items = list(items)
        if not items:
            return
        self._news = items
        self._news_at = self._clock()

    # ------------------------------------------------------------------ misc
    def reset(self) -> None:
        self._turn = None
        self._news = []
        self._news_at = 0.0
        self.weather_city = ""
