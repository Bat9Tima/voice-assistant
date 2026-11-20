"""Адаптивная манера общения: явные просьбы и автоподстройка.

Слои:

* ядро — ``DIALOG_SYSTEM``, его этот модуль не меняет;
* автоматическая подстройка — мат и длина по профилю речи пользователя;
* устойчивые просьбы — явные команды без слов «сегодня» и «сейчас»;
* ситуативная манера — просьба на сегодня, после смены даты не действует.

Модель не решает, что запомнить. Явный сигнал распознаётся правилами по
дословной фразе, автоподстройка — по порогам и с гистерезисом. Каждое
изменение сохраняется новой версией, поэтому манеру можно посмотреть и
откатить. Явная просьба всегда важнее автоподстройки: если пользователь
сказал «не матерись», автоматика не вернёт мат.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .text import normalize

log = logging.getLogger(__name__)

STABLE = "stable"
MOMENT = "moment"

PROFANITY = "profanity"
LENGTH = "length"
TONE = "tone"
HUMOR = "humor"
IRONY = "irony"

DIMENSIONS = (PROFANITY, LENGTH, TONE, HUMOR, IRONY)
VALUES = {
    PROFANITY: frozenset({"off", "on"}),
    LENGTH: frozenset({"short", "detailed"}),
    TONE: frozenset({"direct", "soft"}),
    HUMOR: frozenset({"off", "on"}),
    IRONY: frozenset({"off", "on"}),
}

CHARACTER_HEADER = (
    "Манера общения (явные пожелания пользователя важнее автоматической "
    "подстройки, а обе — общих правил стиля и прошлых фактов; факты и "
    "безопасность они не отменяют):"
)
STABLE_HEADER = "Устойчиво:"
AUTO_HEADER = "Автоматически, по речи пользователя:"
MOMENT_HEADER = "На сегодня:"

# Автоподстройка включается только по этим измерениям. Шутки, ирония и
# тон пока остаются в тени: их признаки слишком легко спутать с эмоцией.
AUTO_MIN_MESSAGES = 20
AUTO_PROFANITY_MIN_MESSAGES = 5
AUTO_PROFANITY_RATE = 0.15
AUTO_PROFANITY_KEEP_MESSAGES = 2
AUTO_PROFANITY_KEEP_RATE = 0.05
AUTO_LENGTH_SHORT_MAX = 4
AUTO_LENGTH_SHORT_KEEP = 6
AUTO_LENGTH_DETAILED_MIN = 12
AUTO_LENGTH_DETAILED_KEEP = 10

_LABELS = {
    (PROFANITY, "off"): "без мата",
    (PROFANITY, "on"): "можно материться",
    (LENGTH, "short"): "короче",
    (LENGTH, "detailed"): "подробнее",
    (TONE, "direct"): "жёстче и прямее",
    (TONE, "soft"): "мягче",
    (HUMOR, "off"): "без шуток",
    (HUMOR, "on"): "шутки уместны",
    (IRONY, "off"): "без иронии",
    (IRONY, "on"): "ирония уместна",
}

# Формулировки для автоматического слоя: это не приказ, а вывод из речи.
_AUTO_LABELS = {
    (PROFANITY, "on"): "мат уместен",
    (LENGTH, "short"): "отвечаю короче",
    (LENGTH, "detailed"): "отвечаю подробнее",
}

# Явные просьбы. Фразы уже нормализованы: нижний регистр, без «ё» и
# пунктуации. Порядок важен: выключение проверяется раньше включения,
# чтобы «не шути» не сработало как «шути».
_SIGNALS = (
    (
        PROFANITY,
        "off",
        (
            "не матерись",
            "не матюкайся",
            "не ругайся матом",
            "не выражайся матом",
            "не используй мат",
            "не надо мата",
            "без мата",
            "без матерщины",
            "без ругательств",
            "не материться",
            "хватит материться",
            "перестань материться",
            "можешь не материться",
            "ты можешь не материться",
            "можно не материться",
            "не надо материться",
            "не нужно материться",
            "не стоит материться",
        ),
    ),
    (
        PROFANITY,
        "on",
        (
            "можно материться",
            "можешь материться",
            "ты можешь материться",
            "мат можно",
            "мат разрешен",
            "разрешаю мат",
            "разрешаю материться",
            "с матом можно",
            "материться можно",
            "не против мата",
        ),
    ),
    (
        LENGTH,
        "short",
        (
            "короче",
            "покороче",
            "по короче",
            "отвечай короче",
            "говори короче",
            "будь короче",
            "отвечай мне короче",
            "говори мне короче",
            "сделай ответ короче",
            "сделай ответы короче",
            "чуть короче",
            "немного короче",
            "отвечай кратко",
            "говори кратко",
            "будь краткой",
            "будь кратким",
            "будь краток",
            "сократи ответы",
            "сократи ответ",
            "меньше слов",
            "не так подробно",
            "без подробностей",
        ),
    ),
    (
        LENGTH,
        "detailed",
        (
            "отвечай подробнее",
            "говори подробнее",
            "будь подробнее",
            "подробнее отвечай",
            "отвечай мне подробнее",
            "говори мне подробнее",
            "сделай ответ подробнее",
            "сделай ответы подробнее",
            "отвечай развернуто",
            "говори развернуто",
            "развернуто отвечай",
            "больше деталей",
            "не так коротко",
        ),
    ),
    (
        TONE,
        "direct",
        (
            "жестче",
            "пожестче",
            "будь жестче",
            "говори жестче",
            "отвечай жестче",
            "будь со мной жестче",
            "отвечай мне жестче",
            "говори мне жестче",
            "можно жестче",
            "можешь жестче",
            "чуть жестче",
            "немного жестче",
            "прямее",
            "будь прямее",
            "говори прямее",
            "отвечай прямее",
            "без сюсюканья",
            "не сюсюкай",
            "не сюсюкайся",
            "не будь мягкой",
            "не будь таким мягким",
            "не будь такой мягкой",
        ),
    ),
    (
        TONE,
        "soft",
        (
            "мягче",
            "помягче",
            "будь мягче",
            "говори мягче",
            "отвечай мягче",
            "будь со мной мягче",
            "отвечай мне мягче",
            "говори мне мягче",
            "можно мягче",
            "можешь мягче",
            "чуть мягче",
            "немного мягче",
            "поделикатнее",
            "деликатнее",
            "будь деликатнее",
            "будь добрее",
            "не будь резкой",
            "не будь такой резкой",
            "не будь грубой",
            "не груби",
        ),
    ),
    (
        HUMOR,
        "off",
        (
            "не шути",
            "не шути со мной",
            "без шуток",
            "не надо шуток",
            "не надо шутить",
            "не нужно шутить",
            "не стоит шутить",
            "хватит шутить",
            "хватит уже шутить",
            "перестань шутить",
            "перестань уже шутить",
            "не шутить",
            "без юмора",
            "можешь не шутить",
            "ты можешь не шутить",
            "можно не шутить",
        ),
    ),
    (
        HUMOR,
        "on",
        (
            "можно шутить",
            "можешь шутить",
            "ты можешь шутить",
            "шутки можно",
            "разрешаю шутки",
            "с шутками можно",
            "шутить можно",
            "не против шуток",
        ),
    ),
    (
        IRONY,
        "off",
        (
            "без иронии",
            "не иронизируй",
            "не надо иронии",
            "не надо иронизировать",
            "не нужно иронизировать",
            "хватит иронии",
            "хватит уже иронизировать",
            "без сарказма",
            "не сарказмируй",
            "не иронизировать",
            "можешь не иронизировать",
            "ты можешь не иронизировать",
            "можно не иронизировать",
        ),
    ),
    (
        IRONY,
        "on",
        (
            "можно с иронией",
            "ирония уместна",
            "с иронией можно",
            "иронизировать можно",
            "не против иронии",
            "можно иронизировать",
            "можешь иронизировать",
            "ты можешь иронизировать",
        ),
    ),
)

_REPORT_PHRASES = frozenset(
    {
        "как ты со мной общаешься",
        "как ты со мной разговариваешь",
        "как ты сейчас общаешься",
        "какая у тебя манера",
        "какая у тебя манера общения",
        "какой у тебя стиль общения",
        "каким тоном ты со мной говоришь",
        "что ты помнишь о моей манере",
        "что ты знаешь о моей манере",
        "расскажи о своей манере",
    }
)
_REPORT_RE = re.compile(
    r"^(?:как ты (?:сейчас )?(?:со мной )?(?:общаешься|разговариваешь|говоришь)"
    r"|какая (?:сейчас )?у тебя манера(?: общения)?"
    r"|какой (?:сейчас )?у тебя стиль(?: общения)?"
    r"|каким (?:тоном|стилем) ты (?:со мной )?(?:общаешься|говоришь)"
    r"|что ты (?:помнишь|знаешь) о (?:моей |твоей )?манере"
    r"|расскажи (?:о |про )?(?:своей |твоей )?манер[еу])$"
)

_ROLLBACK_PHRASES = frozenset(
    {
        "верни прежнюю манеру",
        "верни прежнюю манеру общения",
        "верни прошлую манеру",
        "верни прошлую манеру общения",
        "верни предыдущую манеру",
        "верни прежний стиль",
        "верни прошлый стиль",
        "откати манеру",
        "отмени последнее изменение",
        "отмени последнюю манеру",
        "верни как было",
    }
)
_RESET_PHRASES = frozenset(
    {
        "верни обычную манеру",
        "верни обычную манеру общения",
        "верни нормальную манеру",
        "вернись к обычной манере",
        "вернись к обычной манере общения",
        "вернись к обычному стилю",
        "сбрось манеру",
        "сбрось манеру общения",
        "сбрось стиль",
        "обычная манера",
        "говори как обычно",
        "общайся как обычно",
        "без особой манеры",
    }
)

# Слова, которые не меняют смысл команды. «Пожалуйста» и обращение можно
# убрать в любом месте, вводные — только в начале.
_POLITE = ("пожалуйста", "прошу тебя", "прошу", "будь добра", "будь добр", "луна")
_FILLERS = (
    "ну",
    "ладно",
    "окей",
    "ок",
    "так",
    "хорошо",
    "давай",
    "давайте",
    "слушай",
    "слышь",
    "эй",
)
_MOMENT_MARKERS = (
    "на этот раз",
    "в этом разговоре",
    "в этом диалоге",
    "до конца разговора",
    "на сегодня",
    "на этот день",
    "сегодня",
    "сейчас",
    "пока",
)
# «Теперь», «впредь» и «всегда» задают не срок, а постоянство. Для
# разбора команды они лишние: «теперь без мата» — та же просьба.
_TENSE = ("с этого момента", "теперь", "впредь", "дальше", "всегда", "постоянно", "отныне")

# «Без мата и шуток» — один список, а не две готовые фразы. Разбираем
# существительные после «без» отдельно, чтобы не ловить «без шуток, я устал».
_WITHOUT = {
    "мата": (PROFANITY, "off"),
    "матерщины": (PROFANITY, "off"),
    "ругательств": (PROFANITY, "off"),
    "шуток": (HUMOR, "off"),
    "юмора": (HUMOR, "off"),
    "иронии": (IRONY, "off"),
    "сарказма": (IRONY, "off"),
}


@dataclass(frozen=True)
class Signal:
    """Одна явная просьба о манере."""

    dimension: str
    value: str
    phrase: str


class CharacterMemory:
    """Хранит устойчивую и ситуативную манеру и разбирает команды.

    ``store`` — объект с методами ``add_character_version``,
    ``latest_character`` и ``character_history``. Обычно это
    :class:`app.session_store.SessionStore`.
    """

    def __init__(self, store: object, *, enabled: bool = True) -> None:
        self.store = store
        self.enabled = bool(enabled)

    @classmethod
    def from_config(cls, store: object, section: dict | None = None) -> CharacterMemory:
        section = section or {}
        return cls(store, enabled=bool(section.get("enabled", True)))

    # ------------------------------------------------------------------ read
    def current(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Действующая манера: явные слои и автоподстройка."""
        moment_now = now or _now()
        row = self.store.latest_character()
        if row is None:
            return {
                "version": 0,
                "created_at": "",
                "source": "",
                "quote": "",
                "summary": "",
                "style": {},
                "auto": {},
                "moment": {},
                "moment_date": "",
            }
        style = _clean_style(row.get("style"))
        auto = _clean_style(row.get("auto"))
        moment = _clean_style(row.get("moment"))
        if str(row.get("moment_date") or "") != _today(moment_now):
            moment = {}
        return {**row, "style": style, "auto": auto, "moment": moment}

    def block(self, *, now: datetime | None = None) -> str:
        """Кусок системного промпта или пустая строка."""
        if not self.enabled:
            return ""
        current = self.current(now=now)
        return _render(current["style"], current["auto"], current["moment"])

    def describe(self, *, now: datetime | None = None) -> str:
        """Прямой ответ на «как ты со мной общаешься»."""
        current = self.current(now=now)
        stable = _labels(current["style"])
        automatic = _auto_labels(
            current["auto"],
            skip={**current["style"], **current["moment"]},
        )
        moment = _labels(current["moment"], skip=current["style"])
        parts: list[str] = []
        if stable:
            parts.append(f"Обычно: {', '.join(stable)}")
        if automatic:
            parts.append(f"Автоматически: {', '.join(automatic)}")
        if moment:
            parts.append(f"Сегодня: {', '.join(moment)}")
        if not parts:
            return "Особых пожеланий к манере нет: общаюсь как обычно, по правилам диалога."
        return ". ".join(parts) + "."

    # ----------------------------------------------------------------- write
    def handle(self, text: str, *, now: datetime | None = None) -> str | None:
        """Разбирает команду. ``None`` — это обычная реплика для модели."""
        if not self.enabled:
            return None
        normalized = normalize(text)
        if not normalized:
            return None
        core = _command_core(normalized)
        if not core:
            return None
        if core in _REPORT_PHRASES or _REPORT_RE.match(core):
            return self.describe(now=now)
        if core in _ROLLBACK_PHRASES:
            return self.rollback(text, now=now)
        if core in _RESET_PHRASES:
            return self.reset(text, now=now)
        signals = _detect_signals(normalized)
        if not signals:
            return None
        return self.apply(text, signals, now=now)

    def apply(
        self,
        text: str,
        signals: list[Signal] | tuple[Signal, ...],
        *,
        now: datetime | None = None,
    ) -> str:
        """Применяет просьбы и сохраняет новую версию."""
        moment_now = now or _now()
        layer = MOMENT if _has_moment_marker(normalize(text)) else STABLE
        current = self.current(now=moment_now)
        style = dict(current["style"])
        moment = dict(current["moment"])
        for signal in signals:
            if layer == MOMENT:
                moment[signal.dimension] = signal.value
            else:
                style[signal.dimension] = signal.value
                # Устойчивая просьба сильнее сегодняшней: иначе «можно
                # материться» не сработает, пока живёт «сегодня без мата».
                moment.pop(signal.dimension, None)
        labels = _labels({signal.dimension: signal.value for signal in signals})
        if style == current["style"] and moment == current["moment"]:
            return f"Уже так и общаюсь: {', '.join(labels)}."
        self._save(
            text,
            style=style,
            auto=current["auto"],
            moment=moment,
            moment_date=_today(moment_now) if moment else "",
            source=layer,
            summary=", ".join(labels),
            now=moment_now,
        )
        if layer == MOMENT:
            return f"Принято на сегодня: {', '.join(labels)}."
        return f"Принято: {', '.join(labels)}."

    def rollback(self, text: str, *, now: datetime | None = None) -> str:
        """Возвращает предыдущую отличающуюся версию манеры."""
        moment_now = now or _now()
        current = self.current(now=moment_now)
        history = self.store.character_history(20)
        target: tuple[dict[str, str], dict[str, str], dict[str, str]] | None = None
        for row in history[1:]:
            style = _clean_style(row.get("style"))
            auto = _clean_style(row.get("auto"))
            moment = _clean_style(row.get("moment"))
            if str(row.get("moment_date") or "") != _today(moment_now):
                moment = {}
            if style != current["style"] or auto != current["auto"] or moment != current["moment"]:
                target = (style, auto, moment)
                break
        if target is None:
            if not current["style"] and not current["auto"] and not current["moment"]:
                return "Прежней манеры нет: я и так общаюсь как обычно."
            target = ({}, {}, {})
        style, auto, moment = target
        self._save(
            text,
            style=style,
            auto=auto,
            moment=moment,
            moment_date=_today(moment_now) if moment else "",
            source="rollback",
            summary="откат",
            now=moment_now,
        )
        labels = (
            _labels(style)
            + _auto_labels(auto, skip={**style, **moment})
            + _labels(moment, skip=style)
        )
        if not labels:
            return "Вернулась к обычной манере."
        return f"Вернула прежнюю манеру: {', '.join(labels)}."

    def reset(self, text: str, *, now: datetime | None = None) -> str:
        """Убирает все адаптации, включая автоподстройку, и возвращает ядро."""
        moment_now = now or _now()
        current = self.current(now=moment_now)
        if not current["style"] and not current["auto"] and not current["moment"]:
            self._clear_observations()
            return "Я и так общаюсь как обычно."
        self._save(
            text,
            style={},
            auto={},
            moment={},
            moment_date="",
            source="reset",
            summary="сброс",
            now=moment_now,
        )
        self._clear_observations()
        return "Вернулась к обычной манере."

    def mirror(self, profile: Any, *, now: datetime | None = None) -> dict[str, str] | None:
        """Применяет автоподстройку по профилю речи.

        Меняет только мат и длину, только при достаточном объёме данных и
        с гистерезисом: включить труднее, чем удержать. Возвращает новый
        автоматический слой или ``None``, если он не изменился.
        """
        if not self.enabled:
            return None
        moment_now = now or _now()
        current = self.current(now=moment_now)
        auto = _mirror_auto(profile, current["auto"])
        if auto == current["auto"]:
            return None
        labels = _auto_labels(auto)
        self._save(
            "",
            style=current["style"],
            auto=auto,
            moment=current["moment"],
            moment_date=_today(moment_now) if current["moment"] else "",
            source="auto",
            summary="авто: " + (", ".join(labels) if labels else "без подстройки"),
            now=moment_now,
        )
        return auto

    def is_reset(self, text: str) -> bool:
        """Является ли реплика командой полного сброса манеры."""
        if not self.enabled:
            return False
        core = _command_core(normalize(text))
        return bool(core) and core in _RESET_PHRASES

    def _clear_observations(self) -> None:
        """Сброс должен забыть наблюдения, иначе автоподстройка вернётся."""
        clear = getattr(self.store, "clear_style_observations", None)
        if not callable(clear):
            return
        try:
            clear()
        except Exception:  # noqa: BLE001
            log.warning("Манера: история стиля не сброшена", exc_info=True)

    def _save(
        self,
        text: str,
        *,
        style: dict[str, str],
        auto: dict[str, str],
        moment: dict[str, str],
        moment_date: str,
        source: str,
        summary: str,
        now: datetime,
    ) -> None:
        self.store.add_character_version(
            created_at=now.isoformat(),
            source=source,
            quote=" ".join(str(text or "").split()),
            style=style,
            auto=auto,
            moment=moment,
            moment_date=moment_date,
            summary=summary,
        )
        log.info("Манера общения: %s (%s)", summary, source)


# ------------------------------------------------------------------ parsing
def detect(text: str) -> list[Signal]:
    """Явные просьбы в тексте. Пустой список — обычная реплика."""
    return _detect_signals(normalize(text))


def _detect_signals(normalized: str) -> list[Signal]:
    core = _command_core(normalized)
    if not core:
        return []
    whole = _without_signals(core)
    if whole:
        return whole
    exact = _exact_signals(core)
    if exact:
        return exact
    segments = _segments(core)
    if len(segments) < 2:
        return []
    found: dict[str, Signal] = {}
    for segment in segments:
        signals = _without_signals(segment) or _exact_signals(segment)
        if not signals:
            # «короче и я устал» — не команда, а фраза с союзом.
            return []
        for signal in signals:
            found.setdefault(signal.dimension, signal)
    return list(found.values())


def _exact_signals(text: str) -> list[Signal]:
    """Сигналы, чья фраза целиком совпала с командой."""
    return [
        Signal(dimension, value, text) for dimension, value, phrases in _SIGNALS if text in phrases
    ]


def _segments(core: str) -> list[str]:
    """Части перечисления: «короче и без мата» — две команды."""
    return [part.strip() for part in re.split(r"\s+и\s+", core) if part.strip()]


def _without_signals(core: str) -> list[Signal]:
    """Разбирает «без мата», «без мата и шуток», «без мата и без шуток»."""
    match = re.fullmatch(r"без\s+(.+)", core)
    if not match:
        return []
    parts = [
        part.strip() for part in re.split(r"\s+и\s+|\s+без\s+", match.group(1)) if part.strip()
    ]
    signals: list[Signal] = []
    for part in parts:
        target = _WITHOUT.get(part)
        if target is None:
            return []
        dimension, value = target
        signals.append(Signal(dimension, value, f"без {part}"))
    return signals


def _command_core(normalized: str) -> str:
    text = _drop_phrases(normalized, (*_POLITE, *_MOMENT_MARKERS, *_TENSE))
    words = text.split()
    while words and words[0] in _FILLERS:
        words.pop(0)
    return " ".join(words)


def _drop_phrases(text: str, phrases: tuple[str, ...]) -> str:
    padded = f" {text} "
    for phrase in sorted(set(phrases), key=len, reverse=True):
        if phrase:
            padded = padded.replace(f" {phrase} ", " ")
    return " ".join(padded.split())


def _has_moment_marker(normalized: str) -> bool:
    padded = f" {normalized} "
    return any(f" {marker} " in padded for marker in _MOMENT_MARKERS)


def _labels(style: dict[str, str], *, skip: dict[str, str] | None = None) -> list[str]:
    labels: list[str] = []
    for dimension in DIMENSIONS:
        value = style.get(dimension)
        if value not in VALUES[dimension]:
            continue
        if skip is not None and skip.get(dimension) == value:
            continue
        labels.append(_LABELS[(dimension, value)])
    return labels


def _auto_labels(style: dict[str, str], *, skip: dict[str, str] | None = None) -> list[str]:
    """Подписи автоматического слоя. Явные значения его перекрывают."""
    labels: list[str] = []
    for dimension in (PROFANITY, LENGTH):
        value = style.get(dimension)
        if (dimension, value) not in _AUTO_LABELS:
            continue
        if skip is not None and dimension in skip:
            continue
        labels.append(_AUTO_LABELS[(dimension, value)])
    return labels


def _render(style: dict[str, str], auto: dict[str, str], moment: dict[str, str]) -> str:
    stable = _labels(style)
    automatic = _auto_labels(auto, skip={**style, **moment})
    today = _labels(moment, skip=style)
    if not stable and not automatic and not today:
        return ""
    lines = [CHARACTER_HEADER]
    if stable:
        lines.append(STABLE_HEADER)
        lines.extend(f"- {label}" for label in stable)
    if automatic:
        lines.append(AUTO_HEADER)
        lines.extend(f"- {label}" for label in automatic)
    if today:
        lines.append(MOMENT_HEADER)
        lines.extend(f"- {label}" for label in today)
    return "\n".join(lines)


def _mirror_auto(profile: Any, current: dict[str, str]) -> dict[str, str]:
    """Решает автоматический слой по профилю речи.

    Пороги асимметричные: чтобы включить мат, нужно 5 матерных сообщений
    и 15% окна; чтобы выключить — меньше двух или меньше 5%. Длина
    включается на краях (≤4 или ≥12 слов), а держится в более широком
    коридоре (≤6 или ≥10), поэтому манера не дёргается на каждой реплике.
    """
    messages = _int_attr(profile, "messages")
    if messages < AUTO_MIN_MESSAGES:
        # Данных мало: не меняем уже принятое решение.
        return dict(current)

    auto = {key: value for key, value in current.items() if key not in (PROFANITY, LENGTH)}

    profanity_messages = _int_attr(profile, "profanity_messages")
    profanity_rate = _float_attr(profile, "profanity_message_rate")
    if current.get(PROFANITY) == "on":
        if (
            profanity_messages >= AUTO_PROFANITY_KEEP_MESSAGES
            and profanity_rate >= AUTO_PROFANITY_KEEP_RATE
        ):
            auto[PROFANITY] = "on"
    elif (
        profanity_messages >= AUTO_PROFANITY_MIN_MESSAGES and profanity_rate >= AUTO_PROFANITY_RATE
    ):
        auto[PROFANITY] = "on"

    avg_words = _float_attr(profile, "avg_words")
    length = current.get(LENGTH)
    if length == "short":
        if avg_words <= AUTO_LENGTH_SHORT_KEEP:
            auto[LENGTH] = "short"
        elif avg_words >= AUTO_LENGTH_DETAILED_MIN:
            auto[LENGTH] = "detailed"
    elif length == "detailed":
        if avg_words >= AUTO_LENGTH_DETAILED_KEEP:
            auto[LENGTH] = "detailed"
        elif avg_words <= AUTO_LENGTH_SHORT_MAX:
            auto[LENGTH] = "short"
    elif avg_words <= AUTO_LENGTH_SHORT_MAX:
        auto[LENGTH] = "short"
    elif avg_words >= AUTO_LENGTH_DETAILED_MIN:
        auto[LENGTH] = "detailed"
    return auto


def _int_attr(obj: Any, name: str) -> int:
    try:
        return max(0, int(getattr(obj, name, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _float_attr(obj: Any, name: str) -> float:
    try:
        return max(0.0, float(getattr(obj, name, 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _clean_style(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    result: dict[str, str] = {}
    for dimension in DIMENSIONS:
        value = str(raw.get(dimension) or "")
        if value in VALUES[dimension]:
            result[dimension] = value
    return result


def _today(moment: datetime) -> str:
    return moment.date().isoformat()


def _now() -> datetime:
    return datetime.now().astimezone()
