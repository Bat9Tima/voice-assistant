"""Определение адресата реплики в режиме разговора.

STT слышит не только пользователя: телевизор, видео, разговор с другим
человеком и телефон тоже превращаются в текст. Слой решает, кому
адресована реплика — Луне, фону или это конец разговора — и не даёт
фону попасть в модель и озвучку.

Явные случаи закрывают локальные правила: обращение по имени, маркеры
«не тебе» и фразы выхода. Остальное спрашивает decisions-модель ProxyAPI,
которая возвращает вероятности, а не текст. При сетевом сбое и низкой
уверенности реплика считается адресованной: пропустить живого
пользователя хуже, чем один раз ответить на телевизор.

Режимы:

``off``
    Не ходит в сеть и ничего не меняет.
``shadow``
    Считает адресата и пишет решение в лог. Поведение не меняется.
``active``
    Фон пропускается без ответа, ``closing`` завершает разговор.
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

from .text import normalize

log = logging.getLogger(__name__)

ADDRESSED = "addressed"
BACKGROUND = "background"
CLOSING = "closing"
KINDS = frozenset({ADDRESSED, BACKGROUND, CLOSING})

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
DEFAULT_TIMEOUT_SEC = 2.0
DEFAULT_MAX_BACKGROUND = 3
DEFAULT_BACKGROUND_TIMEOUT_SEC = 15.0

# Вводные слова, с которых начинается устная фраза: «ну всё, пока».
_FILLERS = ("ну", "ладно", "окей", "ок", "так", "хорошо", "давай", "все")

# Фразы, которые завершают разговор. Сравниваются целиком, поэтому
# «всё хорошо» не станет выходом, а «ну всё, спасибо» — станет.
DEFAULT_CLOSING = (
    "хватит",
    "закончим",
    "закончим разговор",
    "заверши разговор",
    "заканчиваем",
    "до свидания",
    "пока",
    "ну все",
    "все спасибо",
    "ну все спасибо",
    "спасибо все",
    "спасибо достаточно",
    "ладно пока",
    "ну пока",
    "давай пока",
    "мне пора",
    "я пошел",
    "отключайся",
)

# Прямые указания, что реплика не для Луны.
BACKGROUND_MARKERS = (
    "не тебе",
    "это не тебе",
    "я не с тобой",
    "я не тебе",
    "не луне",
    "это не луне",
    "я не с луной",
)

_NAME_RE = re.compile(
    r"(?<!\w)лун(?:а|ы|е|у|ой|ою|очка|очки|очке|очку|очкой|ушка|ушки|ушке|ушку|ушкой)(?!\w)",
    re.UNICODE,
)

# Вопрос модели. Описания должны быть короткими и различимыми: модель
# решений не пишет текст, а выбирает вариант.
QUESTION = {
    "type": "choice",
    "instructions": (
        "Ты определяешь, кому адресована реплика в комнате с голосовым "
        "ассистентом Луной. Луна — ассистентка. Пользователь может говорить "
        "с ней, а может смотреть телевизор, разговаривать с другим человеком, "
        "говорить по телефону или сам с собой. Считай реплику адресованной "
        "Луне, только если из текста и контекста видно, что говорящий "
        "обращается к ней или продолжает разговор с ней."
    ),
    "criteria": {
        ADDRESSED: (
            "Пользователь говорит с Луной: вопрос, просьба, ответ на её "
            "вопрос или продолжение их общего разговора."
        ),
        BACKGROUND: (
            "Речь не для Луны: телевизор, видео, другой человек, телефонный "
            "разговор, чтение вслух, песня или разговор с самим собой."
        ),
        CLOSING: (
            "Пользователь явно заканчивает разговор с Луной: «всё, спасибо», "
            "«пока», «закончили», «мне пора»."
        ),
    },
}


class AddresseeError(RuntimeError):
    """Ошибка запроса к модели решений."""


def _one_line(text: str) -> str:
    return " ".join((text or "").split())


def _phrases(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    items = value if isinstance(value, (list, tuple)) and value else default
    return tuple(phrase for phrase in (normalize(str(item)) for item in items) if phrase)


@dataclass(frozen=True)
class AddresseeDecision:
    """Предложение слоя. Поведение применяет вызывающий код."""

    kind: str
    confidence: float = 0.0
    raw: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)
    source: str = SOURCE_MODEL
    elapsed_sec: float = 0.0
    error: str = ""

    @property
    def proposed(self) -> str:
        """Что ответила модель до применения порога."""
        return self.raw or self.kind


class AddresseeGate:
    """Решает, кому адресована реплика, и не пускает фон в разговор.

    Сетевой сбой не должен глушить пользователя: при любой неопределённости
    возвращается ``addressed``. Локальные правила работают без ключа и
    закрывают самые частые случаи до запроса к модели.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        cfg = cfg or {}
        dialog = cfg.get("dialog") or {}
        section = dialog.get("addressee") or {}
        intent = cfg.get("intent") or {}
        cloud = cfg.get("cloud_llm") or {}
        tts = cfg.get("tts") or {}

        self.enabled = bool(section.get("enabled", True))
        mode = str(section.get("mode") or DEFAULT_MODE).strip().lower()
        # Опечатка в режиме не должна включать новое поведение.
        self.mode = mode if mode in MODES else DEFAULT_MODE
        self.base_url = str(
            section.get("base_url")
            or intent.get("base_url")
            or cloud.get("base_url")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = str(section.get("model") or intent.get("model") or DEFAULT_MODEL).strip()
        self.api_key = str(
            section.get("api_key")
            or intent.get("api_key")
            or cloud.get("api_key")
            or tts.get("api_key")
            or ""
        ).strip()
        self.threshold = float(section.get("threshold") or DEFAULT_THRESHOLD)
        self.timeout = float(section.get("timeout_sec") or DEFAULT_TIMEOUT_SEC)
        self.first_turn_always = bool(section.get("first_turn_always_addressed", True))
        self.max_background = max(1, int(section.get("max_background") or DEFAULT_MAX_BACKGROUND))
        self.background_timeout = max(
            0.0,
            float(section.get("background_timeout_sec") or DEFAULT_BACKGROUND_TIMEOUT_SEC),
        )
        self.closing_phrases = _phrases(section.get("closing_phrases"), DEFAULT_CLOSING)
        self.last: AddresseeDecision | None = None

    # ------------------------------------------------------------------ public
    def available(self) -> bool:
        """Можно ли идти в сеть. ``off`` выключает слой."""
        return bool(self.enabled and self.mode != MODE_OFF and self.model and self.api_key)

    def classify(
        self,
        text: str,
        *,
        first_turn: bool = False,
        last_question: str = "",
    ) -> AddresseeDecision:
        """Определяет адресата. Исключений наружу не бросает."""
        text = _one_line(text)
        if not text:
            return self._finish(
                self._decision(ADDRESSED, source=SOURCE_FALLBACK, error="пустая реплика"),
                text,
            )

        normalized = normalize(text)
        if self._is_closing(normalized):
            return self._finish(
                self._decision(
                    CLOSING,
                    confidence=1.0,
                    raw=CLOSING,
                    probabilities={CLOSING: 1.0},
                    source=SOURCE_RULES,
                ),
                text,
            )
        if self._is_background_marker(normalized):
            return self._finish(
                self._decision(
                    BACKGROUND,
                    confidence=0.95,
                    raw=BACKGROUND,
                    probabilities={BACKGROUND: 0.95},
                    source=SOURCE_RULES,
                ),
                text,
            )
        if _NAME_RE.search(normalized):
            return self._finish(
                self._decision(
                    ADDRESSED,
                    confidence=0.9,
                    raw=ADDRESSED,
                    probabilities={ADDRESSED: 0.9},
                    source=SOURCE_RULES,
                ),
                text,
            )
        if first_turn and self.first_turn_always:
            return self._finish(
                self._decision(
                    ADDRESSED,
                    confidence=0.8,
                    raw=ADDRESSED,
                    probabilities={ADDRESSED: 0.8},
                    source=SOURCE_RULES,
                ),
                text,
            )

        if not self.available():
            return self._finish(
                self._decision(
                    ADDRESSED,
                    source=SOURCE_FALLBACK,
                    error="слой выключен или нет ключа",
                ),
                text,
            )

        started = time.monotonic()
        try:
            data = self._request(text, last_question=last_question)
        except (AddresseeError, urllib.error.URLError, TimeoutError, OSError) as exc:
            return self._finish(
                self._decision(
                    ADDRESSED,
                    source=SOURCE_FALLBACK,
                    elapsed_sec=time.monotonic() - started,
                    error=str(exc),
                ),
                text,
            )
        return self._finish(self._parse(data, started), text)

    def observe(
        self,
        text: str,
        *,
        first_turn: bool = False,
        last_question: str = "",
    ) -> None:
        """Считает адресата в фоне: статистика не задерживает ответ."""
        if not self.available():
            return
        threading.Thread(
            target=self._observe,
            args=(text, first_turn, last_question),
            name="addressee-shadow",
            daemon=True,
        ).start()

    # ---------------------------------------------------------------- private
    def _observe(self, text: str, first_turn: bool, last_question: str) -> None:
        try:
            self.classify(text, first_turn=first_turn, last_question=last_question)
        except Exception:  # noqa: BLE001
            log.exception("Адресат: сбой фонового наблюдателя")

    @staticmethod
    def _decision(
        kind: str,
        *,
        confidence: float = 0.0,
        raw: str = "",
        probabilities: dict[str, float] | None = None,
        source: str = SOURCE_MODEL,
        elapsed_sec: float = 0.0,
        error: str = "",
    ) -> AddresseeDecision:
        return AddresseeDecision(
            kind=kind,
            confidence=float(confidence),
            raw=raw,
            probabilities=dict(probabilities or {}),
            source=source,
            elapsed_sec=float(elapsed_sec),
            error=error,
        )

    def _is_closing(self, normalized: str) -> bool:
        if not normalized:
            return False
        candidates = [normalized]
        words = normalized.split()
        # «ну всё, спасибо»: вводные слова не мешают точному совпадению.
        while words and words[0] in _FILLERS:
            words = words[1:]
            candidate = " ".join(words)
            if candidate:
                candidates.append(candidate)
        return any(candidate in self.closing_phrases for candidate in candidates)

    @staticmethod
    def _is_background_marker(normalized: str) -> bool:
        return any(marker in normalized for marker in BACKGROUND_MARKERS)

    def _request(self, text: str, *, last_question: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "state": {
                "dialog_mode": True,
                "last_luna_question": _one_line(last_question)[-300:],
                "utterance": text,
            },
            "questions": {"addressee": QUESTION},
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
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise AddresseeError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AddresseeError(f"сервер недоступен: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise AddresseeError("сервер вернул не JSON") from exc
        if not isinstance(data, dict):
            raise AddresseeError("сервер вернул не объект")
        return data

    def _parse(self, data: dict[str, Any], started: float) -> AddresseeDecision:
        elapsed = time.monotonic() - started
        answers = data.get("answers")
        answer = answers.get("addressee") if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            return self._decision(
                ADDRESSED,
                source=SOURCE_FALLBACK,
                elapsed_sec=elapsed,
                error="в ответе нет answers.addressee",
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

        if raw not in KINDS:
            return self._decision(
                ADDRESSED,
                confidence=confidence,
                raw=raw,
                probabilities=probabilities,
                source=SOURCE_FALLBACK,
                elapsed_sec=elapsed,
                error=f"неизвестный адресат: {raw or 'пусто'}",
            )

        # Низкая уверенность — отвечаем: живой пользователь важнее фона.
        kind = raw if confidence >= self.threshold else ADDRESSED
        return self._decision(
            kind,
            confidence=confidence,
            raw=raw,
            probabilities=probabilities,
            source=SOURCE_MODEL,
            elapsed_sec=elapsed,
        )

    def _finish(self, decision: AddresseeDecision, text: str) -> AddresseeDecision:
        self.last = decision
        self._log(text, decision)
        return decision

    def _log(self, text: str, decision: AddresseeDecision) -> None:
        """Одна строка на решение: по ней потом считается статистика."""
        proposed = (
            f", модель: {decision.raw}" if decision.raw and decision.raw != decision.kind else ""
        )
        error = f", ошибка: {decision.error}" if decision.error else ""
        log.info(
            "Адресат [%s]: %s%s (уверенность %.2f, %s, %.2f с)%s :: %s",
            self.mode,
            decision.kind,
            proposed,
            decision.confidence,
            decision.source,
            decision.elapsed_sec,
            error,
            text[:160],
        )
