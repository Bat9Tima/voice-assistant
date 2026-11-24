"""Классификатор намерений для разговорного режима.

Решает, что пользователь делает одной репликой: команду, ответ на вопрос
Луны, светскую беседу, запрос справки или уточнение. В отличие от
``app.intents`` здесь нет шаблонов команд: это отдельный слой поверх
правил, который смотрит на контекст разговора и возвращает вероятности.

Модель — decisions-эндпоинт ProxyAPI. Он не пишет текст: в ответ приходят
вероятности вариантов. Поэтому классификатор ничего не выполняет сам,
а только предлагает намерение и уверенность; решение остаётся за
роутером.

Режимы:

``off``
    Не ходит в сеть и ничего не пишет.
``shadow``
    Считает намерение в фоне и пишет решение в лог. Поведение не меняется.
``active``
    Перед маршрутом к внешним данным (погода, новости, поиск) ждёт решение
    и может отменить его, если фраза оказалась разговором, ответом или
    уточнением. Дата, время и уточнения к прошлому ответу не трогаются:
    они решаются локально и по контексту.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .qa import EXPLICIT_SEARCH
from .visual_intent import VISUAL_INTENTS, VISUAL_QUESTION

log = logging.getLogger(__name__)

COMMAND = "command"
CONTROL = "control"
ANSWER = "answer"
DIALOG = "dialog"
INFO = "info"
CLARIFY = "clarify"
UNKNOWN = "unknown"

INTENTS = frozenset({COMMAND, CONTROL, ANSWER, DIALOG, INFO, CLARIFY, UNKNOWN})

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ACTIVE = "active"
MODES = frozenset({MODE_OFF, MODE_SHADOW, MODE_ACTIVE})

SOURCE_RULES = "rules"
SOURCE_MODEL = "model"
SOURCE_FALLBACK = "fallback"

DEFAULT_BASE_URL = "https://api.proxyapi.ru/v1"
DEFAULT_MODEL = "openai/gpt-6-luna-decisions"
DEFAULT_MODE = MODE_SHADOW
DEFAULT_THRESHOLD = 0.6
DEFAULT_TIMEOUT_SEC = 4.0
DEFAULT_ANSWER_TTL_SEC = 120.0

# Слова, с которых начинается явная просьба поискать. Такие фразы
# классификатор не отменяет: пользователь прямо сказал, что ему нужен поиск.
SEARCH_VERBS = (
    "поищи",
    "погугли",
    "загугли",
    "найди",
    "поиск",
    "посмотри в интернете",
)

QUESTION_RE = re.compile(r"[^.!?\n]*\?")

# Вопрос модели. Чем точнее описания, тем стабильнее вероятности.
#
# Главное отличие от первой версии: answer — это прямой ответ на последний
# вопрос Луны, а не любая реплика, которую можно им считать. Вопрос,
# приветствие и новая тема остаются dialog, даже если Луна только что
# что-то спросила. Вопросы о самой Луне — тоже dialog, а не info.
QUESTION = {
    "type": "choice",
    "instructions": (
        "Ты определяешь намерение одной реплики пользователя в голосовом разговоре "
        "с ассистенткой Луной. Сначала реши, чем реплика является сама по себе, и "
        "только потом проверь, не ответ ли это на прошлый вопрос Луны. Вопрос, "
        "приветствие, шутка или новая тема не становятся answer только потому, что "
        "Луна что-то спросила. Если реплика сама задаёт вопрос, это не answer."
    ),
    "criteria": {
        COMMAND: (
            "Просит выполнить действие с устройством или браузером: открыть, "
            "включить, выключить, поставить, громче, тише."
        ),
        CONTROL: (
            "Просит управлять самим разговором: начать или закончить разговор, "
            "замолчать, стоп, не задавать вопросы, сменить манеру речи."
        ),
        ANSWER: (
            "Прямо отвечает на последний вопрос Луны: продолжает его тему и даёт "
            "ответ по существу. Не вопрос, не приветствие и не новая тема."
        ),
        DIALOG: (
            "Поддерживает разговор: приветствие, как дела, мнение, чувства, рассказ "
            "о себе, вопрос о Луне, её памяти, внешности или характере, шутка. Сюда "
            "же вопросы «что ты помнишь», «как ты выглядишь», «что ты думаешь»."
        ),
        INFO: (
            "Просит внешние или свежие данные: погода, новости, курс, цены, пробки, "
            "факт о мире, общие знания. Вопрос о самой Луне и её памяти сюда не входит."
        ),
        CLARIFY: (
            "Просит повторить или объяснить предыдущий ответ Луны: что ты имеешь в "
            "виду, почему, подробнее, ещё раз, не понял."
        ),
    },
}


class IntentError(RuntimeError):
    """Ошибка запроса к модели решений."""


def _one_line(text: str) -> str:
    return " ".join((text or "").split())


def _last_question(text: str) -> str:
    """Последний вопрос в ответе Луны. Пустая строка — вопросов не было."""
    matches = [match.group(0).strip() for match in QUESTION_RE.finditer(text or "")]
    return matches[-1][-300:] if matches else ""


@dataclass
class DialogState:
    """Контекст разговора для классификатора: последний вопрос Луны.

    Состояние живёт в памяти процесса. TTL не даёт старому вопросу
    влиять на новые реплики; ноль выключает состояние совсем.
    """

    ttl_sec: float = DEFAULT_ANSWER_TTL_SEC
    clock: Any = time.monotonic
    _question: str = ""
    _at: float = 0.0

    def question(self) -> str:
        """Последний вопрос, если он ещё не устарел."""
        if not self._question:
            return ""
        if self.ttl_sec <= 0 or self.clock() - self._at > self.ttl_sec:
            self._question = ""
            return ""
        return self._question

    @property
    def awaiting_answer(self) -> bool:
        return bool(self.question())

    def note_question(self, text: str) -> None:
        """Запоминает вопрос из свежего ответа Луны."""
        question = _last_question(text)
        if not question or self.ttl_sec <= 0:
            self.clear()
            return
        self._question = question
        self._at = self.clock()

    def note_reply(self, text: str) -> None:
        """Обновляет состояние после ответа Луны: был вопрос или нет."""
        if _last_question(text):
            self.note_question(text)
        else:
            self.clear()

    def note_answer(self) -> None:
        """Пользователь ответил: вопрос больше не ждёт ответа."""
        self.clear()

    def clear(self) -> None:
        self._question = ""
        self._at = 0.0


@dataclass(frozen=True)
class IntentDecision:
    """Предложение классификатора. Исполняет его роутер, не этот модуль."""

    intent: str
    confidence: float = 0.0
    raw: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)
    source: str = SOURCE_MODEL
    elapsed_sec: float = 0.0
    input_tokens: int = 0
    error: str = ""

    @property
    def proposed(self) -> str:
        """Что ответила модель до применения порога."""
        return self.raw or self.intent


class IntentClassifier:
    """Клиент decisions-модели и правила вокруг него.

    Сетевой сбой не роняет разговор: возвращается решение с источником
    ``fallback``, и вызывающий код остаётся на старом пути.
    """

    def __init__(self, cfg: dict[str, Any], *, clock: Any = time.monotonic) -> None:
        cfg = cfg or {}
        section = cfg.get("intent") or {}
        cloud = cfg.get("cloud_llm") or {}
        tts = cfg.get("tts") or {}

        self.enabled = bool(section.get("enabled", True))
        mode = str(section.get("mode") or DEFAULT_MODE).strip().lower()
        # Опечатка в режиме не должна включать новое поведение.
        self.mode = mode if mode in MODES else DEFAULT_MODE
        self.base_url = str(
            section.get("base_url") or cloud.get("base_url") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = str(section.get("model") or DEFAULT_MODEL).strip()
        self.api_key = str(
            section.get("api_key") or cloud.get("api_key") or tts.get("api_key") or ""
        ).strip()
        self.threshold = float(section.get("threshold") or DEFAULT_THRESHOLD)
        self.timeout = float(section.get("timeout_sec") or DEFAULT_TIMEOUT_SEC)
        self.state = DialogState(
            ttl_sec=float(section.get("answer_ttl_sec") or DEFAULT_ANSWER_TTL_SEC),
            clock=clock,
        )
        self.last: IntentDecision | None = None
        visual = (cfg.get("vision") or {}).get("intent")
        if not isinstance(visual, dict):
            visual = {}
        self.visual_enabled = bool(visual.get("enabled", True))
        self.visual_threshold = float(visual.get("threshold") or DEFAULT_THRESHOLD)
        self.visual_timeout = float(visual.get("timeout_sec") or DEFAULT_TIMEOUT_SEC)
        self.last_visual: IntentDecision | None = None

    # ------------------------------------------------------------------ public
    def available(self) -> bool:
        """Можно ли идти в сеть. Режим ``off`` выключает классификатор."""
        return bool(self.enabled and self.mode != MODE_OFF and self.model and self.api_key)

    def available_visual(self) -> bool:
        """Можно ли спросить модель, относится ли фраза к экрану."""
        return bool(self.enabled and self.visual_enabled and self.model and self.api_key)

    def classify_visual(self, text: str, *, in_dialog: bool) -> IntentDecision:
        """Решает, про текущий экран ли реплика. Исключений не бросает.

        Это отдельный вопрос к той же decisions-модели: картинка не
        передаётся, поэтому вызов дешёвый. Низкая уверенность становится
        ``unknown``, и вызывающий код остаётся на прежнем маршруте.
        """
        text = _one_line(text)
        if not text:
            return self._decision(UNKNOWN, source=SOURCE_RULES, error="пустая реплика")

        if not self.available_visual():
            decision = self._decision(
                UNKNOWN,
                source=SOURCE_FALLBACK,
                error="визуальный классификатор выключен или нет ключа",
            )
            self._log_decision(
                text,
                in_dialog,
                decision,
                label="Визуальный вопрос",
                mode="visual",
            )
            return decision

        started = time.monotonic()
        try:
            data = self._request(
                text,
                in_dialog=in_dialog,
                question_key="visual",
                question=VISUAL_QUESTION,
                timeout=self.visual_timeout,
            )
        except (IntentError, urllib.error.URLError, TimeoutError, OSError) as exc:
            decision = self._decision(
                UNKNOWN,
                source=SOURCE_FALLBACK,
                elapsed_sec=time.monotonic() - started,
                error=str(exc),
            )
        else:
            decision = self._parse(
                data,
                started,
                question_key="visual",
                allowed=VISUAL_INTENTS,
                threshold=self.visual_threshold,
                use_probability=True,
            )
        self.last_visual = decision
        self._log_decision(
            text,
            in_dialog,
            decision,
            label="Визуальный вопрос",
            mode="visual",
        )
        return decision

    def classify(self, text: str, *, in_dialog: bool) -> IntentDecision:
        """Считает намерение синхронно. Исключений наружу не бросает."""
        text = _one_line(text)
        if not text:
            return self._decision(UNKNOWN, source=SOURCE_RULES, error="пустая реплика")

        if self._explicit_search(text):
            decision = self._decision(
                INFO,
                confidence=1.0,
                raw=INFO,
                probabilities={INFO: 1.0},
                source=SOURCE_RULES,
            )
            self._log_decision(text, in_dialog, decision)
            return decision

        if not self.available():
            decision = self._decision(
                UNKNOWN,
                source=SOURCE_FALLBACK,
                error="классификатор выключен или нет ключа",
            )
            self._log_decision(text, in_dialog, decision)
            return decision

        started = time.monotonic()
        try:
            data = self._request(text, in_dialog=in_dialog)
        except (IntentError, urllib.error.URLError, TimeoutError, OSError) as exc:
            decision = self._decision(
                UNKNOWN,
                source=SOURCE_FALLBACK,
                elapsed_sec=time.monotonic() - started,
                error=str(exc),
            )
        else:
            decision = self._parse(data, started)
        self.last = decision
        self._log_decision(text, in_dialog, decision)
        return decision

    def observe(self, text: str, *, in_dialog: bool) -> None:
        """Считает намерение в фоне: статистика не задерживает ответ."""
        if not self.available():
            return
        threading.Thread(
            target=self._observe,
            args=(text, in_dialog),
            name="intent-shadow",
            daemon=True,
        ).start()

    def blocks_lookup(self, decision: IntentDecision) -> bool:
        """Можно ли отменить маршрут к инструменту в пользу разговора.

        Отменяем только решение самой модели: ``dialog``, ``answer``,
        ``clarify`` и ``unknown``. Неопределённость — это не «наверное,
        поиск»: инструменты вслепую не запускаем. Правило явного поиска и
        сетевой сбой оставляют старый маршрут нетронутым.
        """
        if decision.source != SOURCE_MODEL:
            return False
        return decision.intent in {ANSWER, DIALOG, CLARIFY, UNKNOWN}

    def blocks_search(self, decision: IntentDecision) -> bool:
        """Совместимое имя: раньше слой отменял только поиск."""
        return self.blocks_lookup(decision)

    # ---------------------------------------------------------------- private
    @staticmethod
    def _decision(
        intent: str,
        *,
        confidence: float = 0.0,
        raw: str = "",
        probabilities: dict[str, float] | None = None,
        source: str = SOURCE_MODEL,
        elapsed_sec: float = 0.0,
        input_tokens: int = 0,
        error: str = "",
    ) -> IntentDecision:
        return IntentDecision(
            intent=intent,
            confidence=float(confidence),
            raw=raw,
            probabilities=dict(probabilities or {}),
            source=source,
            elapsed_sec=float(elapsed_sec),
            input_tokens=int(input_tokens),
            error=error,
        )

    @staticmethod
    def _explicit_search(text: str) -> bool:
        lowered = text.lower().replace("ё", "е")
        markers = tuple(marker.replace("ё", "е") for marker in EXPLICIT_SEARCH) + SEARCH_VERBS
        return any(marker in lowered for marker in markers)

    def _request(
        self,
        text: str,
        *,
        in_dialog: bool,
        question_key: str = "intent",
        question: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "state": {
                "dialog_mode": bool(in_dialog),
                "awaiting_answer": self.state.awaiting_answer,
                "last_luna_question": self.state.question(),
                "utterance": text,
            },
            "questions": {question_key: question or QUESTION},
        }
        request = urllib.request.Request(
            self.base_url + "/decisions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout if timeout is None else float(timeout),
            ) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise IntentError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise IntentError(f"сервер недоступен: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise IntentError("сервер вернул не JSON") from exc
        if not isinstance(data, dict):
            raise IntentError("сервер вернул не объект")
        return data

    def _parse(
        self,
        data: dict[str, Any],
        started: float,
        *,
        question_key: str = "intent",
        allowed: frozenset[str] = INTENTS,
        threshold: float | None = None,
        use_probability: bool = False,
    ) -> IntentDecision:
        elapsed = time.monotonic() - started
        answers = data.get("answers")
        answer = answers.get(question_key) if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            return self._decision(
                UNKNOWN,
                source=SOURCE_FALLBACK,
                elapsed_sec=elapsed,
                error=f"в ответе нет answers.{question_key}",
            )

        raw = str(answer.get("choice") or "").strip().lower()
        probabilities: dict[str, float] = {}
        for key, value in (answer.get("probabilities") or {}).items():
            try:
                probabilities[str(key).strip().lower()] = float(value)
            except (TypeError, ValueError):
                continue

        confidence = answer.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        if confidence is None:
            confidence = probabilities.get(raw, 0.0)
        elif use_probability and raw in probabilities:
            # В decisions-ответе confidence — отрыв лидера от второго
            # варианта, а не вероятность выбора. Для бинарного вопроса
            # «экран / не экран» нужна именно вероятность: иначе
            # «почему это не работает» (0.69 против 0.31) выглядит как
            # неуверенность 0.38. Общий классификатор оставляем на отрыве.
            confidence = probabilities[raw]

        usage = data.get("usage") or {}
        try:
            tokens = int(usage.get("input_tokens") or 0)
        except (TypeError, ValueError):
            tokens = 0

        if raw not in allowed:
            return self._decision(
                UNKNOWN,
                confidence=confidence,
                raw=raw,
                probabilities=probabilities,
                elapsed_sec=elapsed,
                input_tokens=tokens,
                error=f"неизвестное намерение: {raw or 'пусто'}",
            )

        # Низкая уверенность — это unknown: инструменты вслепую не запускаем.
        limit = self.threshold if threshold is None else float(threshold)
        intent = raw if confidence >= limit else UNKNOWN
        return self._decision(
            intent,
            confidence=confidence,
            raw=raw,
            probabilities=probabilities,
            elapsed_sec=elapsed,
            input_tokens=tokens,
        )

    def _log_decision(
        self,
        text: str,
        in_dialog: bool,
        decision: IntentDecision,
        *,
        label: str = "Намерение",
        mode: str | None = None,
    ) -> None:
        """Одна строка на решение: по ней потом считается статистика."""
        proposed = (
            f", модель: {decision.raw}" if decision.raw and decision.raw != decision.intent else ""
        )
        error = f", ошибка: {decision.error}" if decision.error else ""
        log.info(
            "%s [%s%s]: %s%s (уверенность %.2f, %s, %.2f с, токены %s)%s :: %s",
            label,
            self.mode if mode is None else mode,
            ", диалог" if in_dialog else "",
            decision.intent,
            proposed,
            decision.confidence,
            decision.source,
            decision.elapsed_sec,
            decision.input_tokens,
            error,
            text[:160],
        )

    def _observe(self, text: str, in_dialog: bool) -> None:
        try:
            self.classify(text, in_dialog=in_dialog)
        except Exception:  # noqa: BLE001
            log.exception("Классификатор намерений: сбой фонового наблюдателя")
