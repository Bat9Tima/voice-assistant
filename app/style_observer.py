"""Наблюдатель стиля пользователя и автоподстройка манеры.

Считает измеримые признаки: длину реплик, мат, грубость, вежливость,
шутки, иронию и вопросы. По короткому окну решает, какие черты манеры
включить; это применяет ``CharacterMemory.mirror()``. Текст пользователя
здесь не сохраняется: в базу пишутся только числа.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from .text import normalize

log = logging.getLogger(__name__)

DEFAULT_WINDOW_MESSAGES = 200
# Решения об автоподстройке принимаются по короткому окну: привычка
# должна затухать за десятки реплик, а не за сотни.
DEFAULT_DECISION_MESSAGES = 30
DEFAULT_LOG_EVERY = 10
# Ниже этого числа сообщений профиль слишком шумный для выводов.
MIN_MESSAGES = 20
# Столько сообщений нужно, чтобы уверенность стала максимальной.
CONFIDENCE_MESSAGES = 50

# Корни мата. Границы слов обязательны: «хлеб», «сукно» и «бляха» —
# обычные слова, а не сигналы.
_PROFANITY_RE = re.compile(
    r"\b(?:"
    # Приставки важны: «охуел», «нихуя», «похуист» — тот же корень,
    # но без приставки шаблон их не увидит.
    r"(?:о|а|не|ни|ах|по|на|за|вы|у|от|до|раз|с|про|пере|под|над|об|из)?ху[йяеию]\w*"
    r"|хул[еия]\b"
    r"|пизд\w*"
    r"|бля(?:д\w*|т\w*)?\b"
    # Голое «сук» — это ветка, поэтому нужны формы с окончанием.
    r"|сук(?:а|и|у|ой|ам|ами|ах)\b"
    r"|суч(?:к|ар|он)\w*"
    r"|муд(?:ак|ила|озвон)\w*"
    r"|пид(?:ор|ар|р)\w*"
    r"|педик\w*"
    r"|г[ао]ндон\w*"
    r"|дроч\w*"
    r"|залуп\w*"
    r"|шлюх\w*"
    r"|наху[йя]\w*"
    r"|поху[йя]\w*"
    r"|(?:за|на|у|от|в|раз|до|пере|про|с|под|над|об|из|вы|недо|"
    r"долбо|мудо|съ|въ|объ|разъ|отъ|подъ|надъ|изъ)?еб\w*"
    r")"
)

# Грубость без мата: ругань, но не обязательно нецензурная.
_ROUGH_RE = re.compile(
    r"\b(?:"
    r"говн\w*"
    r"|жоп\w*"
    r"|срат\w*"
    r"|ссан\w*"
    r"|задолбал\w*"
    r"|бесит\w*"
    r"|бешу\w*"
    r"|заткни\w*"
    r"|идиот\w*"
    r"|дурак\w*"
    r"|кретин\w*"
    r"|дебил\w*"
    r"|нахер\w*"
    r"|похер\w*"
    r"|хер(?:ня|ни|ню|ней)\b"
    r"|мраз\w*"
    r"|твар\w*"
    r")"
)

# Смех в распознанной речи: «ахаха», «хех», «лол», «ору» — маркер юмора.
_LAUGH_RE = re.compile(
    r"\b(?:"
    r"(?:а?ха)+"
    r"|(?:хе)+х?"
    r"|(?:хи)+х?"
    r"|(?:хо)+х?"
    r"|лол"
    r"|кек"
    r"|ору"
    r"|ржу"
    r"|ржака"
    r"|ржачно"
    r"|прикол(?:ы|ов|ик|чик|ьн\w*|ист\w*)?\b"
    r"|мем(?:ы|ов|ик|чик|ас\w*)?\b"
    r")\b"
)
_HUMOR_RE = re.compile(r"\b(?:шутк\w*|шучу|смешн\w*|юмор\w*)\b")
_IRONY_RE = re.compile(r"\b(?:ирони\w*|сарказ\w*|саркастич\w*)\b")
# Отрицания не должны включать шутки и иронию: «без шуток, я устал» —
# это серьёзная реплика, а не признак юмора.
_NO_HUMOR_RE = re.compile(
    r"\b(?:без шуток|без юмора|без смеха|не шути|не шутить|не шучу|"
    r"не смешн\w*|это не шутка|не до шуток|не до смеха|хватит шутить)\b"
)
_NO_IRONY_RE = re.compile(
    r"\b(?:без иронии|без сарказма|не ирони\w*|"
    r"это не ирония|не сарказм)\b"
)
# Вежливая, смягчённая речь: просьбы с «пожалуйста», извинения,
# осторожные формулировки. Это сигнал для автоматического tone=soft.
_POLITE_RE = re.compile(
    r"\b(?:"
    r"пожалуйста|спасибо|благодарю|"
    r"извини(?:те)?|прости(?:те)?|"
    r"будь добр(?:а)?|будь так добра|"
    r"если можно|если тебе не сложно|если вам не сложно|"
    r"если хочешь|если тебе удобно|как тебе удобно|как хочешь|"
    r"не спеши|не торопись|не обязательно|по возможности|"
    r"солнышко|милая|дорогая|"
    r"не мог(?:ла)? бы ты"
    r")\b"
)


def _now() -> datetime:
    return datetime.now().astimezone()


@dataclass(frozen=True)
class StyleSample:
    """Признаки одной реплики пользователя. Без исходного текста."""

    words: int = 0
    chars: int = 0
    profanity_words: int = 0
    profanity_messages: int = 0
    rough_words: int = 0
    polite_messages: int = 0
    humor_messages: int = 0
    irony_messages: int = 0
    question: int = 0

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> StyleSample:
        """Строка из ``style_observations``."""
        return cls(
            words=_int(row.get("words")),
            chars=_int(row.get("chars")),
            profanity_words=_int(row.get("profanity_words")),
            profanity_messages=_int(row.get("profanity_messages")),
            rough_words=_int(row.get("rough_words")),
            polite_messages=_int(row.get("polite_messages")),
            humor_messages=_int(row.get("humor_messages")),
            irony_messages=_int(row.get("irony_messages")),
            question=_int(row.get("question")),
        )

    def as_row(self) -> dict[str, int]:
        return {
            "words": self.words,
            "chars": self.chars,
            "profanity_words": self.profanity_words,
            "profanity_messages": self.profanity_messages,
            "rough_words": self.rough_words,
            "polite_messages": self.polite_messages,
            "humor_messages": self.humor_messages,
            "irony_messages": self.irony_messages,
            "question": self.question,
        }


@dataclass
class StyleProfile:
    """Сумма признаков по окну сообщений и производные оценки."""

    messages: int = 0
    words: int = 0
    chars: int = 0
    profanity_words: int = 0
    profanity_messages: int = 0
    rough_words: int = 0
    polite_messages: int = 0
    humor_messages: int = 0
    irony_messages: int = 0
    question_messages: int = 0

    @classmethod
    def from_samples(cls, samples: Iterable[StyleSample]) -> StyleProfile:
        profile = cls()
        for sample in samples:
            profile.add(sample)
        return profile

    def add(self, sample: StyleSample) -> None:
        self.messages += 1
        self.words += sample.words
        self.chars += sample.chars
        self.profanity_words += sample.profanity_words
        self.profanity_messages += sample.profanity_messages
        self.rough_words += sample.rough_words
        self.polite_messages += sample.polite_messages
        self.humor_messages += sample.humor_messages
        self.irony_messages += sample.irony_messages
        self.question_messages += sample.question

    @property
    def avg_words(self) -> float:
        return self.words / self.messages if self.messages else 0.0

    @property
    def profanity_message_rate(self) -> float:
        """Доля сообщений, где встретился мат."""
        return self.profanity_messages / self.messages if self.messages else 0.0

    @property
    def rough_rate(self) -> float:
        """Доля грубых слов среди всех слов."""
        return self.rough_words / self.words if self.words else 0.0

    @property
    def polite_rate(self) -> float:
        return self.polite_messages / self.messages if self.messages else 0.0

    @property
    def humor_rate(self) -> float:
        return self.humor_messages / self.messages if self.messages else 0.0

    @property
    def irony_rate(self) -> float:
        return self.irony_messages / self.messages if self.messages else 0.0

    @property
    def question_rate(self) -> float:
        return self.question_messages / self.messages if self.messages else 0.0

    @property
    def confidence(self) -> float:
        """Насколько профилю можно верить: 0…1 по числу сообщений."""
        return min(1.0, self.messages / CONFIDENCE_MESSAGES)

    def suggestions(self) -> dict[str, str]:
        """Осторожные кандидаты в манеру. Пока только для лога.

        Порогов специально много: лучше промолчать, чем предложить
        зеркалить случайную эмоцию.
        """
        result: dict[str, str] = {}
        if self.messages < MIN_MESSAGES:
            return result
        if self.profanity_messages >= 5 and self.profanity_message_rate >= 0.15:
            result["profanity"] = "on"
        elif self.messages >= CONFIDENCE_MESSAGES and self.profanity_messages == 0:
            result["profanity"] = "off"
        if self.avg_words <= 4:
            result["length"] = "short"
        elif self.avg_words >= 12:
            result["length"] = "detailed"
        if self.humor_rate >= 0.12:
            result["humor"] = "on"
        if self.irony_rate >= 0.08:
            result["irony"] = "on"
        if self.rough_words >= 5 and self.rough_rate >= 0.02:
            result["tone"] = "direct"
        elif (
            self.polite_messages >= 6
            and self.polite_rate >= 0.20
            and self.rough_words == 0
            and self.profanity_messages == 0
        ):
            result["tone"] = "soft"
        return result

    def summary(self) -> str:
        """Одна строка для лога: цифры и осторожное предложение."""
        if not self.messages:
            return "Стиль: данных пока нет."
        line = (
            "Стиль: "
            f"сообщений {self.messages}, "
            f"слов {self.avg_words:.1f}, "
            f"мат {self.profanity_message_rate:.0%}, "
            f"грубость {self.rough_rate:.1%}, "
            f"вежливость {self.polite_rate:.0%}, "
            f"шутки {self.humor_rate:.0%}, "
            f"ирония {self.irony_rate:.0%}, "
            f"вопросы {self.question_rate:.0%}, "
            f"уверенность {self.confidence:.2f}"
        )
        suggestions = self.suggestions()
        if suggestions:
            line += "; кандидаты: " + ", ".join(
                f"{name}={value}" for name, value in suggestions.items()
            )
        return line


class StyleObserver:
    """Копит признаки стиля в окне последних сообщений.

    ``store`` — объект с методами ``add_style_observation`` и
    ``style_observations``. Обычно :class:`app.session_store.SessionStore`.
    Без хранилища наблюдение работает только в памяти.
    """

    def __init__(
        self,
        store: object | None = None,
        *,
        enabled: bool = True,
        window_messages: int = DEFAULT_WINDOW_MESSAGES,
        decision_messages: int = DEFAULT_DECISION_MESSAGES,
        log_every: int = DEFAULT_LOG_EVERY,
    ) -> None:
        self.store = store
        self.enabled = bool(enabled)
        self.window_messages = max(1, int(window_messages))
        self.decision_messages = max(1, min(self.window_messages, int(decision_messages)))
        self.log_every = max(1, int(log_every))
        self._samples: deque[StyleSample] = deque(maxlen=self.window_messages)
        self._recent: deque[StyleSample] = deque(maxlen=self.decision_messages)
        self._profile = StyleProfile()
        self._seen = 0
        if self.enabled:
            self._load()

    @classmethod
    def from_config(
        cls, store: object | None, section: dict[str, Any] | None = None
    ) -> StyleObserver:
        section = section or {}
        return cls(
            store,
            enabled=bool(section.get("enabled", True)),
            window_messages=int(section.get("window_messages") or DEFAULT_WINDOW_MESSAGES),
            decision_messages=int(section.get("decision_messages") or DEFAULT_DECISION_MESSAGES),
            log_every=int(section.get("log_every") or DEFAULT_LOG_EVERY),
        )

    def observe(
        self,
        text: str,
        *,
        session_id: str = "",
        now: datetime | None = None,
    ) -> StyleSample | None:
        """Добавляет реплику в окно и пишет агрегаты."""
        if not self.enabled:
            return None
        sample = analyze(text)
        if not sample.words:
            return None
        self._samples.append(sample)
        self._recent.append(sample)
        self._profile = StyleProfile.from_samples(self._samples)
        self._seen += 1
        self._save(sample, session_id=session_id, now=now or _now())
        if self._seen % self.log_every == 0:
            log.info("%s", self._profile.summary())
        return sample

    def flush(self) -> None:
        """Пишет итог разговора в лог, не дожидаясь кратности ``log_every``."""
        if not self.enabled or not self._seen:
            return
        log.info("%s", self._profile.summary())
        self._seen = 0

    def profile(self) -> StyleProfile:
        """Текущий профиль. Копия: вызывающий не должен править окно."""
        return replace(self._profile)

    def recent_profile(self) -> StyleProfile:
        """Профиль короткого окна для автоподстройки."""
        return StyleProfile.from_samples(self._recent)

    def reset(self) -> None:
        """Забывает историю стиля: сброс манеры начинает профиль заново."""
        self._samples.clear()
        self._recent.clear()
        self._profile = StyleProfile()
        self._seen = 0
        if self.store is None:
            return
        clearer = getattr(self.store, "clear_style_observations", None)
        if not callable(clearer):
            return
        try:
            clearer()
        except Exception:  # noqa: BLE001
            log.warning("Стиль: история наблюдений не сброшена", exc_info=True)

    # ---------------------------------------------------------------- private
    def _load(self) -> None:
        if self.store is None:
            return
        loader = getattr(self.store, "style_observations", None)
        if not callable(loader):
            return
        try:
            rows = loader(self.window_messages)
        except Exception:  # noqa: BLE001
            log.warning("Стиль: история наблюдений недоступна", exc_info=True)
            return
        # Хранилище отдаёт новые сверху; окно наполняем в хронологии.
        for row in reversed(rows):
            sample = StyleSample.from_row(row)
            self._samples.append(sample)
            self._recent.append(sample)
        self._profile = StyleProfile.from_samples(self._samples)

    def _save(
        self,
        sample: StyleSample,
        *,
        session_id: str,
        now: datetime,
    ) -> None:
        if self.store is None:
            return
        writer = getattr(self.store, "add_style_observation", None)
        if not callable(writer):
            return
        try:
            writer(
                created_at=now.isoformat(),
                session_id=session_id,
                **sample.as_row(),
            )
        except Exception:  # noqa: BLE001
            log.warning("Стиль: наблюдение не записано", exc_info=True)


def analyze(text: str) -> StyleSample:
    """Считает признаки одной реплики. Пустая реплика даёт нули."""
    raw = " ".join((text or "").split())
    normalized = normalize(raw)
    words = normalized.split()
    if not words:
        return StyleSample()
    profanity = _PROFANITY_RE.findall(normalized)
    rough = _ROUGH_RE.findall(normalized)
    humor = bool(
        (_LAUGH_RE.search(normalized) or _HUMOR_RE.search(normalized))
        and not _NO_HUMOR_RE.search(normalized)
    )
    irony = bool(_IRONY_RE.search(normalized) and not _NO_IRONY_RE.search(normalized))
    return StyleSample(
        words=len(words),
        chars=len(raw),
        profanity_words=len(profanity),
        profanity_messages=1 if profanity else 0,
        rough_words=len(rough),
        polite_messages=1 if _POLITE_RE.search(normalized) else 0,
        humor_messages=1 if humor else 0,
        irony_messages=1 if irony else 0,
        question=1 if "?" in raw else 0,
    )


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
