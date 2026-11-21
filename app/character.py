"""Адаптивная манера общения: явные просьбы и автоподстройка.

Слои:

* ядро — ``DIALOG_SYSTEM``, его этот модуль не меняет;
* автоматическая подстройка — мат, длина, тон, шутки и ирония по речи;
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

# Автоподстройка: включаем черту только после 20 реплик и с запасом,
# а держим при более слабом сигнале. Пороги асимметричные, чтобы манера
# не дёргалась на одной эмоции.
AUTO_MIN_MESSAGES = 20

AUTO_PROFANITY_MIN_MESSAGES = 5
AUTO_PROFANITY_RATE = 0.15
AUTO_PROFANITY_KEEP_MESSAGES = 2
AUTO_PROFANITY_KEEP_RATE = 0.05

AUTO_LENGTH_SHORT_MAX = 4
AUTO_LENGTH_SHORT_KEEP = 6
AUTO_LENGTH_DETAILED_MIN = 12
AUTO_LENGTH_DETAILED_KEEP = 10

AUTO_HUMOR_MIN_MESSAGES = 4
AUTO_HUMOR_RATE = 0.12
AUTO_HUMOR_KEEP_MESSAGES = 2
AUTO_HUMOR_KEEP_RATE = 0.05

AUTO_IRONY_MIN_MESSAGES = 3
AUTO_IRONY_RATE = 0.08
AUTO_IRONY_KEEP_MESSAGES = 1
AUTO_IRONY_KEEP_RATE = 0.03

AUTO_TONE_DIRECT_WORDS = 5
AUTO_TONE_DIRECT_RATE = 0.02
AUTO_TONE_DIRECT_KEEP_WORDS = 2
AUTO_TONE_DIRECT_KEEP_RATE = 0.01
AUTO_TONE_SOFT_MIN_MESSAGES = 6
AUTO_TONE_SOFT_RATE = 0.20
AUTO_TONE_SOFT_KEEP_MESSAGES = 3
AUTO_TONE_SOFT_KEEP_RATE = 0.10

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
    (TONE, "direct"): "говорю жёстче и прямее",
    (TONE, "soft"): "говорю мягче",
    (HUMOR, "on"): "шутки уместны",
    (IRONY, "on"): "ирония уместна",
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
    def matches(self, text: str) -> bool:
        """Похоже ли на команду манере, без изменения состояния."""
        if not self.enabled:
            return False
        normalized = normalize(text)
        if not normalized:
            return False
        core = _command_core(normalized)
        if not core:
            return False
        if core in _REPORT_PHRASES or _REPORT_RE.match(core):
            return True
        if core in _ROLLBACK_PHRASES or core in _RESET_PHRASES:
            return True
        return bool(_detect_signals(normalized))

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
        # Откат — ручная точка опоры: старое окно не должно пересчитать
        # автослой следующей же репликой.
        self._clear_observations()
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

        Меняет только те черты, для которых хватает сигнала, и с
        гистерезисом: включить труднее, чем удержать. Возвращает новый
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

    def is_rollback(self, text: str) -> bool:
        """Является ли реплика командой отката манеры."""
        if not self.enabled:
            return False
        core = _command_core(normalize(text))
        return bool(core) and core in _ROLLBACK_PHRASES

    def resets_observations(self, text: str) -> bool:
        """Команда, после которой историю стиля нужно начать заново."""
        return self.is_reset(text) or self.is_rollback(text)

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
    for dimension in DIMENSIONS:
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

    Включает только положительные черты: мат, короткую или подробную
    длину, прямой или мягкий тон, шутки и иронию. «Выключение» — это
    отсутствие черты, а не приказ модели. Пороги асимметричные: включить
    труднее, чем удержать, поэтому манера не дёргается на одной реплике.
    """
    messages = _int_attr(profile, "messages")
    if messages < AUTO_MIN_MESSAGES:
        # Данных мало: не меняем уже принятое решение.
        return dict(current)

    auto: dict[str, str] = {}
    if _binary_on(
        profile,
        current.get(PROFANITY) == "on",
        count_attr="profanity_messages",
        rate_attr="profanity_message_rate",
        on_count=AUTO_PROFANITY_MIN_MESSAGES,
        on_rate=AUTO_PROFANITY_RATE,
        keep_count=AUTO_PROFANITY_KEEP_MESSAGES,
        keep_rate=AUTO_PROFANITY_KEEP_RATE,
    ):
        auto[PROFANITY] = "on"

    length = _mirror_length(profile, current.get(LENGTH))
    if length:
        auto[LENGTH] = length

    tone = _mirror_tone(profile, current.get(TONE))
    if tone:
        auto[TONE] = tone

    if _binary_on(
        profile,
        current.get(HUMOR) == "on",
        count_attr="humor_messages",
        rate_attr="humor_rate",
        on_count=AUTO_HUMOR_MIN_MESSAGES,
        on_rate=AUTO_HUMOR_RATE,
        keep_count=AUTO_HUMOR_KEEP_MESSAGES,
        keep_rate=AUTO_HUMOR_KEEP_RATE,
    ):
        auto[HUMOR] = "on"

    if _binary_on(
        profile,
        current.get(IRONY) == "on",
        count_attr="irony_messages",
        rate_attr="irony_rate",
        on_count=AUTO_IRONY_MIN_MESSAGES,
        on_rate=AUTO_IRONY_RATE,
        keep_count=AUTO_IRONY_KEEP_MESSAGES,
        keep_rate=AUTO_IRONY_KEEP_RATE,
    ):
        auto[IRONY] = "on"
    return auto


def _binary_on(
    profile: Any,
    was_on: bool,
    *,
    count_attr: str,
    rate_attr: str,
    on_count: int,
    on_rate: float,
    keep_count: int,
    keep_rate: float,
) -> bool:
    """Порог включения выше порога удержания: так черта не мигает."""
    count = _int_attr(profile, count_attr)
    rate = _float_attr(profile, rate_attr)
    if was_on:
        return count >= keep_count and rate >= keep_rate
    return count >= on_count and rate >= on_rate


def _mirror_length(profile: Any, current: str | None) -> str | None:
    """Короткая или подробная длина с мёртвой зоной между порогами."""
    avg_words = _float_attr(profile, "avg_words")
    if current == "short":
        if avg_words <= AUTO_LENGTH_SHORT_KEEP:
            return "short"
        if avg_words >= AUTO_LENGTH_DETAILED_MIN:
            return "detailed"
    elif current == "detailed":
        if avg_words >= AUTO_LENGTH_DETAILED_KEEP:
            return "detailed"
        if avg_words <= AUTO_LENGTH_SHORT_MAX:
            return "short"
    elif avg_words <= AUTO_LENGTH_SHORT_MAX:
        return "short"
    elif avg_words >= AUTO_LENGTH_DETAILED_MIN:
        return "detailed"
    return None


def _mirror_tone(profile: Any, current: str | None) -> str | None:
    """Прямой тон по грубости, мягкий — по вежливым формулировкам."""
    rough_words = _int_attr(profile, "rough_words")
    rough_rate = _float_attr(profile, "rough_rate")
    if current == "direct":
        direct = (
            rough_words >= AUTO_TONE_DIRECT_KEEP_WORDS and rough_rate >= AUTO_TONE_DIRECT_KEEP_RATE
        )
    else:
        direct = rough_words >= AUTO_TONE_DIRECT_WORDS and rough_rate >= AUTO_TONE_DIRECT_RATE
    if direct:
        return "direct"

    polite_messages = _int_attr(profile, "polite_messages")
    polite_rate = _float_attr(profile, "polite_rate")
    clean = rough_words == 0 and _int_attr(profile, "profanity_messages") == 0
    if current == "soft":
        soft = (
            polite_messages >= AUTO_TONE_SOFT_KEEP_MESSAGES
            and polite_rate >= AUTO_TONE_SOFT_KEEP_RATE
            and clean
        )
    else:
        soft = (
            polite_messages >= AUTO_TONE_SOFT_MIN_MESSAGES
            and polite_rate >= AUTO_TONE_SOFT_RATE
            and clean
        )
    return "soft" if soft else None


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
