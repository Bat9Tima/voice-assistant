"""Куда отнести вопрос: к тексту, к описанию снимка или к новому кадру.

Сам снимок живёт только в памяти до ``vision.followup_ttl_sec``. Дальше
остаётся текст описания в истории разговора, и этот модуль решает, когда
вопрос относится к нему, а когда это обычная текстовая реплика.

Режимы:

``text_only`` — снимок не нужен, работает обычный путь;
``context_only`` — отвечаем по сохранённому описанию, без нового кадра;
``new_capture`` — нужен новый снимок (явная команда зрения);
``ambiguous`` — непонятно, о каком снимке речь: лучше попросить показать экран.

Правила детерминированные: голосовой цикл не должен платить ещё одним
LLM-вызовом за каждую фразу. Спорные случаи честно уходят в ``ambiguous``,
а не угадываются.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .text import normalize

TEXT_ONLY = "text_only"
CONTEXT_ONLY = "context_only"
NEW_CAPTURE = "new_capture"
AMBIGUOUS = "ambiguous"

DEFAULT_CONTEXT_TTL_SEC = 1800.0

# Подсказка для модели, когда отвечаем по сохранённому описанию. Она идёт
# последним обязательным блоком промпта: описание снимка важнее общей памяти,
# но не отменяет правила и манеру общения.
VISUAL_CONTEXT_HINT = (
    "Пользователь спрашивает о последнем снимке экрана. Опирайся на описание "
    "того, что было на экране, из истории выше. Если описания нет или в нём "
    "нет нужной детали, честно скажи, что на этом снимке ты этого не "
    "разглядела, и предложи показать экран ещё раз. Не выдумывай детали "
    "и не говори, что видишь экран прямо сейчас."
)

# «на фото», «на прошлом снимке», «на моём экране» — явная ссылка на кадр.
_VISUAL_REF_RE = re.compile(
    r"\b(?:на|в|по|с)\s+"
    r"(?:(?:этом|этой|том|той|моем|моей|твоем|твоей|прошлом|прошлой)\s+)?"
    r"(?:фото|фотографии|фотке|снимке|скриншоте|скрине|картинке|"
    r"изображении|экране)\b",
    re.UNICODE,
)
_PAST_RE = re.compile(r"\b(?:было|был|была|были|раньше|до этого|только что)\b", re.UNICODE)
# «в левом верхнем углу», «справа», «на фоне» — пространственная деталь.
_POSITION_RE = re.compile(
    r"\b(?:углу|угла|угле|слева|справа|сверху|снизу|наверху|внизу|на\s+фоне)\b",
    re.UNICODE,
)
# «какого цвета кнопка» — цвет спрашивают и про небо, поэтому нужен объект.
_COLOR_RE = re.compile(r"\b(?:цвет|цвета|цветом)\b", re.UNICODE)
_VISUAL_OBJECT_RE = re.compile(
    r"\b(?:кнопк\w*|текст\w*|надпис\w*|окн\w*|иконк\w*|ссылк\w*|пол[еяю]|"
    r"меню|курсор|экран\w*|снимок|снимке|фото|картинк\w*)\b",
    re.UNICODE,
)
# «а там?», «а он кто?» — без указателя это может быть и обычный разговор.
# Личные и пространственные местоимения сами по себе указывают на предмет.
_PRONOMINAL_RE = re.compile(
    r"\b(?:там|тут|здесь|он|она|оно|они|его|ее|их|нем|ней|них|ним|ними)\b",
    re.UNICODE,
)
# «что это?», «это что?» — указательное слово требует вопроса, иначе
# «это интересно» в разговоре уйдёт в зрение без причины.
_DEMONSTRATIVE_RE = re.compile(
    r"\b(?:это|этот|эта|эти|этого|этой|этом|этих)\b",
    re.UNICODE,
)
# Для дейксиса нужен вопрос о предмете. «Как там дела» — разговорная фраза,
# а не вопрос о снимке, поэтому «как» сюда не входит.
_DEICTIC_QUESTION_RE = re.compile(
    r"\b(?:что|где|куда|какой|какая|какое|какие|какого|какому|каком|какую|"
    r"каким|какими|каких|сколько|кто|кого|кому|кем)\b",
    re.UNICODE,
)
_SEE_RE = re.compile(r"\b(?:видела|увидела|разглядела|заметила|показывала)\b", re.UNICODE)

_QUESTION_WORDS = frozenset(
    {
        "что",
        "где",
        "куда",
        "как",
        "какой",
        "какая",
        "какое",
        "какие",
        "какого",
        "какому",
        "каком",
        "какую",
        "каким",
        "какими",
        "каких",
        "сколько",
        "почему",
        "зачем",
        "кто",
        "кого",
        "кому",
        "кем",
        "чей",
        "чья",
        "чье",
        "чьи",
        "чьего",
        "чьему",
        "чьим",
        "чьей",
        "чью",
        "чьих",
    }
)
_QUESTION_VERBS = frozenset(
    {
        "покажи",
        "скажи",
        "опиши",
        "расскажи",
        "прочитай",
        "подскажи",
        "объясни",
        "напомни",
        "уточни",
    }
)
# «а открой…», «ну нажми…» — вводное слово не делает команду вопросом.
_LEAD_FILLERS = frozenset({"а", "ну", "и", "так", "пожалуйста", "давай"})
# Команда важнее кадра: «нажми на кнопку в левом верхнем углу» — не вопрос.
_COMMAND_STARTS = frozenset(
    {
        "открой",
        "закрой",
        "нажми",
        "поставь",
        "перемести",
        "сделай",
        "включи",
        "выключи",
        "запусти",
        "перейди",
        "сверни",
        "разверни",
        "переключи",
        "удали",
        "сохрани",
        "напиши",
        "отправь",
        "убери",
        "подними",
        "опусти",
        "запомни",
        "забудь",
        "запиши",
    }
)


@dataclass(frozen=True)
class VisualDecision:
    """Предложение маршрутизатора. Исполняет его ``main``, не этот модуль."""

    mode: str
    reason: str = ""
    confidence: float = 0.0


class VisualRouter:
    """Правила вокруг последнего описания снимка.

    ``has_context`` и ``context_age`` приходят снаружи: модуль не знает,
    где хранится разговор, и не трогает сессии. Так его можно тестировать
    таблицей фраз без базы и модели.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        section = (cfg or {}).get("vision") or {}
        value = section.get("context_ttl_sec")
        if value is None:
            value = DEFAULT_CONTEXT_TTL_SEC
        self.context_ttl = max(0.0, float(value))

    def decide(
        self,
        text: str,
        *,
        has_context: bool,
        context_age: float | None = None,
        in_dialog: bool = False,
        explicit_capture: bool = False,
    ) -> VisualDecision:
        """Решает, нужен ли снимок для ответа на фразу."""
        raw = (text or "").strip()
        normalized = normalize(raw)
        if not normalized:
            return VisualDecision(TEXT_ONLY, "empty")

        if explicit_capture:
            return VisualDecision(NEW_CAPTURE, "explicit", 1.0)

        words = normalized.split()
        head = next((word for word in words if word not in _LEAD_FILLERS), "")
        if head in _COMMAND_STARTS:
            return VisualDecision(TEXT_ONLY, "command", 0.9)

        question = self._is_question(normalized, raw)
        explicit_ref = bool(_VISUAL_REF_RE.search(normalized))
        past = bool(_PAST_RE.search(normalized))
        position = bool(_POSITION_RE.search(normalized)) or (
            bool(_COLOR_RE.search(normalized)) and bool(_VISUAL_OBJECT_RE.search(normalized))
        )
        pointed = bool(_PRONOMINAL_RE.search(normalized)) or bool(
            _DEMONSTRATIVE_RE.search(normalized)
        )
        # Короткий вопрос с указателем — про снимок; длинная фраза с «там»
        # чаще обычный разговор («что там нового у тебя»).
        deictic = (
            pointed
            and len(words) <= 4
            and (
                bool(_VISUAL_OBJECT_RE.search(normalized))
                or (question and bool(_DEICTIC_QUESTION_RE.search(normalized)))
            )
        )
        seen = bool(_SEE_RE.search(normalized))

        if not has_context:
            # Ссылка на снимок без описания: не отправляем в общий чат,
            # где модель начнёт выдумывать увиденное.
            if explicit_ref or (question and (position or (past and (deictic or seen)))):
                return VisualDecision(AMBIGUOUS, "no_context", 0.8)
            return VisualDecision(TEXT_ONLY, "no_context", 0.6)

        if explicit_ref:
            if question or in_dialog or past:
                return VisualDecision(CONTEXT_ONLY, "explicit_ref", 0.95)
            return VisualDecision(TEXT_ONLY, "explicit_ref_fragment", 0.6)

        # Вне разговора обрывок без вопроса — скорее команда или шум.
        if not question and not in_dialog:
            return VisualDecision(TEXT_ONLY, "not_question", 0.6)
        if not question and not (position or deictic or seen):
            return VisualDecision(TEXT_ONLY, "not_question", 0.6)

        fresh = context_age is None or context_age <= self.context_ttl
        if position:
            return VisualDecision(
                CONTEXT_ONLY if fresh else AMBIGUOUS,
                "position",
                0.8,
            )
        if deictic or seen:
            return VisualDecision(
                CONTEXT_ONLY if fresh else AMBIGUOUS,
                "deictic",
                0.7,
            )
        return VisualDecision(TEXT_ONLY, "general", 0.6)

    @staticmethod
    def _is_question(normalized: str, raw: str) -> bool:
        """Вопрос ли это: по знаку, первому слову или вопросительному слову."""
        if raw.rstrip().endswith("?"):
            return True
        words = normalized.split()
        if not words:
            return False
        if words[0] in _QUESTION_WORDS or words[0] in _QUESTION_VERBS:
            return True
        # «в левом верхнем углу что?» — вопрос не в начале фразы.
        return len(words) <= 8 and any(word in _QUESTION_WORDS for word in words)


__all__ = [
    "AMBIGUOUS",
    "CONTEXT_ONLY",
    "DEFAULT_CONTEXT_TTL_SEC",
    "NEW_CAPTURE",
    "TEXT_ONLY",
    "VISUAL_CONTEXT_HINT",
    "VisualDecision",
    "VisualRouter",
]
