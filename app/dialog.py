from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .context_builder import (
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_CHARS_PER_TOKEN,
    ContextBuilder,
    ContextPart,
)
from .datetime_answer import datetime_context
from .persona import PersonaMemory
from .profile import Profile
from .prompts import DIALOG_SYSTEM
from .session import (
    DEFAULT_HISTORY_CHARS,
    DEFAULT_HISTORY_MESSAGES,
    SessionMemory,
)
from .text import normalize

DEFAULT_TURN_TTL_SEC = 180.0
DEFAULT_NEWS_TTL_SEC = 900.0
DEFAULT_ENTRY_PHRASES = ("давай поговорим", "давай поболтаем", "поговорим", "поболтаем")
DEFAULT_EXIT_PHRASES = (
    "хватит",
    "закончим",
    "закончим разговор",
    "заверши разговор",
    "до свидания",
)
# Вводные слова, с которых начинается устная фраза: «ну давай поговорим».
_ENTRY_FILLERS = ("ну", "ладно", "окей", "ок", "так", "хорошо", "давай")


def _limit(value: Any, default: int) -> int:
    """Лимит из конфига: ноль — без ограничения, мусор — значение по умолчанию."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


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


class DialogService:
    """Режим непрерывного разговора с историей текущей сессии.

    История живёт только в памяти: начали разговор — она пустая, вышли —
    сессия уходит в ``SessionMemory.on_finish`` и из диалога пропадает.
    Постоянный контекст задаёт текстовый файл профиля, который пишет сам
    пользователь.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        llm: Any,
        profile: Profile | None = None,
        sessions: SessionMemory | None = None,
        memory: Any | None = None,
        persona: PersonaMemory | None = None,
    ) -> None:
        section = (cfg or {}).get("dialog") or {}
        self.llm = llm
        self.profile = profile
        # Факты Луны о себе: отдельный блок, не смешивается с профилем
        # пользователя и эпизодической памятью.
        self.persona = persona
        # Эпизодическая память: резюме прошлых разговоров. Подключается
        # только в режиме диалога — сюда не ходят разовые команды.
        self.memory = memory
        self.entry_phrases = self._phrases(section.get("entry_phrases"), DEFAULT_ENTRY_PHRASES)
        self.exit_phrases = self._phrases(section.get("exit_phrases"), DEFAULT_EXIT_PHRASES)
        self.idle_timeout = max(1.0, float(section.get("idle_timeout_sec") or 60))
        self.max_utterance = max(1.0, float(section.get("max_utterance_sec") or 60))
        self.no_speech_timeout = max(1.0, float(section.get("no_speech_timeout_sec") or 10))
        self.silence_ms = max(200, int(section.get("silence_ms") or 1200))
        self.max_errors = max(1, int(section.get("max_errors") or 2))
        self.max_tokens = max(1, int(section.get("max_tokens") or 400))
        self.temperature = float(section.get("temperature", 0.5))
        self.timeout = float(section.get("timeout_sec") or 60)
        self.history_messages = _limit(section.get("history_messages"), DEFAULT_HISTORY_MESSAGES)
        self.history_chars = _limit(section.get("history_chars"), DEFAULT_HISTORY_CHARS)
        context_cfg = section.get("context") or {}
        budget = context_cfg.get("budget_tokens")
        self.context = ContextBuilder(
            max_tokens=DEFAULT_BUDGET_TOKENS if budget is None else int(budget),
            chars_per_token=float(context_cfg.get("chars_per_token") or DEFAULT_CHARS_PER_TOKEN),
        )
        self.sessions = sessions or SessionMemory(
            self.history_messages,
            self.history_chars,
        )

    @staticmethod
    def _phrases(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
        items = value if isinstance(value, (list, tuple)) and value else default
        return tuple(phrase for phrase in (normalize(str(item)) for item in items) if phrase)

    def wants_entry(self, text: str) -> bool:
        return self._matches(text, self.entry_phrases)

    def wants_exit(self, text: str) -> bool:
        return self._matches(text, self.exit_phrases)

    @staticmethod
    def _matches(text: str, phrases: tuple[str, ...]) -> bool:
        normalized = normalize(text)
        if not normalized:
            return False
        candidates = [normalized]
        words = normalized.split()
        # «ну давай поговорим», «ладно хватит»: вводные слова не мешают.
        while words and words[0] in _ENTRY_FILLERS:
            words = words[1:]
            candidate = " ".join(words)
            if candidate:
                candidates.append(candidate)
        return any(candidate in phrases for candidate in candidates)

    def system_parts(self) -> list[ContextPart]:
        """Блоки системного промпта по важности: от обязательных к памяти."""
        parts = [
            ContextPart("rules", DIALOG_SYSTEM.strip(), required=True),
            ContextPart("datetime", datetime_context(), required=True),
        ]
        if self.profile is not None:
            block = self.profile.block()
            if block:
                parts.append(ContextPart("profile", block))
        if self.persona is not None:
            block = self.persona.block()
            if block:
                parts.append(ContextPart("persona", block, keep_tail=True))
        return parts

    def system_prompt(self) -> str:
        """Правила + дата и время + профиль пользователя + факты Луны."""
        return "\n\n".join(part.text for part in self.system_parts())

    def start(self) -> None:
        """Начинает разговор с чистого листа: прошлая сессия закрывается."""
        self.sessions.start()

    def stop(self) -> None:
        """Закрывает сессию. Сейчас она просто уходит в ``on_finish``."""
        self.sessions.finish()

    def reply(self, text: str) -> str:
        """Один ход: окно истории + текущая реплика.

        Реплики попадают в сессию только после успешного ответа, чтобы
        неудачный ход не отравлял контекст следующих вопросов. Перед
        запросом к модели подмешиваются резюме прошлых разговоров, если
        память включена и нашлось что-то по смыслу.
        """
        text = " ".join((text or "").split())
        if not text:
            raise ValueError("пустая реплика")
        built = self.context.build(
            parts=self.system_parts(),
            history=self.sessions.history(),
            user_text=text,
            memory=self._memory_block(text),
        )
        answer = self.llm.chat(
            built.messages,
            system=built.system,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        answer = " ".join((answer or "").split())
        if not answer:
            raise ValueError("модель вернула пустой ответ")
        self.sessions.add("user", text)
        self.sessions.add("assistant", answer)
        return answer

    def _memory_block(self, text: str) -> str:
        """Воспоминания для промпта. Любой сбой памяти — просто без блока."""
        if self.memory is None:
            return ""
        try:
            return str(self.memory.block(text) or "")
        except Exception:  # noqa: BLE001
            return ""
