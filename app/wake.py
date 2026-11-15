from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vosk

from .text import fuzzy_contains, normalize

log = logging.getLogger(__name__)

# Слова, которыми можно остановить озвучку. Держим их короткими и
# однозначными: в свободном распознавании они редко встречаются в ответах.
DEFAULT_STOP_WORDS = (
    "стоп",
    "хватит",
    "тише",
    "замолчи",
    "перестань",
    "отмена",
    "остановись",
    "помолчи",
)

# Блоки тише этого уровня считаем эхом озвучки или шумом и не отдаём Vosk:
# иначе декодер цепляется за собственный голос ассистента.
BARGE_MIN_LEVEL = 0.002
# До этого уровня усиливаем тихую речь: Vosk плохо распознаёт шёпот.
BARGE_TARGET_LEVEL = 0.08
BARGE_MAX_GAIN = 8.0


def condition_pcm(pcm: bytes, min_level: float, target_level: float) -> bytes:
    """Готовит микрофонный блок к распознаванию перебивания.

    Тихие блоки заменяются тишиной: иначе Vosk цепляется за эхо озвучки.
    Тихая речь усиливается до ``target_level``, громкая остаётся как есть.
    """
    if not pcm:
        return b""
    usable = len(pcm) - (len(pcm) % 2)
    if usable <= 0:
        return b""
    samples = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0
    level = float(np.sqrt(np.mean(samples**2)))
    if level < min_level:
        return b"\x00" * usable
    if level < target_level:
        gain = min(BARGE_MAX_GAIN, target_level / max(level, 1e-6))
        samples = np.clip(samples * gain, -1.0, 1.0)
        return (samples * 32767).astype("<i2").tobytes()
    return pcm[:usable]


def allowed_errors(alias: str) -> int:
    """Сколько опечаток распознавания допускать в ключевой фразе.

    В коротком слове одна ошибка — это уже другое слово: «луна» → «уна»,
    «лупа», «руна». Поэтому такие фразы сравниваем точно.
    """
    if len(alias.split()) > 1:
        # «хей луна» Vosk стабильно слышит как «хэ луна». Более коротким
        # фразам («эй луна») опечатку не прощаем: иначе «и луна» в ответе
        # про космос превращается в ложный wake.
        return 1 if len(alias) >= 8 else 0
    return 0 if len(alias) <= 6 else 1


# Vosk путает «э»/«е» и «й»/«и»: «хей луна» стабильно превращается в
# «хэ луна» или «хэлуна». Перед сравнением сводим эти буквы к одной форме,
# иначе wake-фраза не ловится вообще.
_WAKE_FOLD = str.maketrans({"э": "е", "й": "и"})


def fold_wake_text(text: str) -> str:
    """Приводит текст к форме, устойчивой к заменам «э/е» и «й/и»."""
    return normalize(text).translate(_WAKE_FOLD)


class WakeWordDetector:
    """Офлайн-детектор ключевой фразы на Vosk с ограниченной грамматикой."""

    def __init__(
        self,
        model_dir: Path,
        aliases: list[str],
        sample_rate: int = 16000,
        confusables: list[str] | None = None,
        warm_up: bool = True,
        quiet_level: float = 0.01,
        quiet_window: int = 8,
    ) -> None:
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Модель Vosk не найдена: {model_dir}. Запустите scripts\\setup.ps1"
            )
        self.aliases = [normalize(alias) for alias in aliases if normalize(alias)]
        if not self.aliases:
            raise ValueError("В config.json не заданы wake_aliases")

        self.sample_rate = int(sample_rate)
        # Уровень, ниже которого блок считается тишиной. Wake-фраза должна
        # начинаться после паузы: иначе это обрывок чужой речи или телевизора,
        # а грамматический декодер всё равно выдаст знакомые слова.
        self.quiet_level = max(0.0, float(quiet_level))
        self._quiet: deque[bool] = deque(maxlen=max(3, int(quiet_window)))
        self._mute_until = 0.0
        # Алиас, на котором сработал последний feed: нужен вызывающей
        # стороне, чтобы отличить свою озвучку от команды пользователя.
        self.last_match: str | None = None
        vosk.SetLogLevel(-1)
        self.model = vosk.Model(str(model_dir))
        # Если в грамматике только «луна», декодер обязан услышать её даже
        # в «уна» или «дума». Похожие слова дают ему альтернативу.
        extra = [normalize(word) for word in (confusables or []) if normalize(word)]
        vocabulary = list(dict.fromkeys([*self.aliases, *extra]))
        self.grammar = json.dumps([*vocabulary, "[unk]"], ensure_ascii=False)
        self.recognizer = self._new_recognizer()
        log.info(
            "Детектор ключевого слова готов: %s (похожих слов: %d, порог паузы: %.4f)",
            ", ".join(self.aliases),
            len(extra),
            self.quiet_level,
        )
        if warm_up:
            self._warm_up()

    def _warm_up(self) -> None:
        # Первый проход после загрузки модели Vosk хуже слышит начало фразы:
        # «хей луна» превращается в «луна». Прогреваем распознаватель
        # тишиной, чтобы первая же фраза после запуска не потерялась.
        self.recognizer.AcceptWaveform(b"\x00\x00" * self.sample_rate)
        self.recognizer = self._new_recognizer()

    def _new_recognizer(self):
        recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate, self.grammar)
        recognizer.SetWords(True)
        return recognizer

    def feed(self, pcm: bytes) -> bool:
        if not pcm:
            return False
        self._quiet.append(self._level(pcm) < self.quiet_level)

        if time.monotonic() < self._mute_until:
            return False

        self.last_match = None
        if self.recognizer.AcceptWaveform(pcm):
            raw = json.loads(self.recognizer.Result()).get("text", "")
        else:
            raw = json.loads(self.recognizer.PartialResult()).get("partial", "")

        text = normalize(raw)
        if not text:
            return False
        for alias in self.aliases:
            if fuzzy_contains(text, alias, max_distance=allowed_errors(alias)):
                if not self._after_pause():
                    log.info(
                        "Похоже на «%s», но перед фразой не было паузы — игнорирую",
                        alias,
                    )
                    # Гасим повторы: на непрерывной чужой речи декодер будет
                    # находить фразу в каждом блоке.
                    self._mute_until = time.monotonic() + 2.0
                    return False
                self.last_match = alias
                log.info("Ключевое слово: «%s» (услышано: «%s»)", alias, text)
                return True
        return False

    @staticmethod
    def _level(pcm: bytes) -> float:
        usable = len(pcm) - (len(pcm) % 2)
        if usable <= 0:
            return 0.0
        samples = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(samples**2)))

    def _after_pause(self) -> bool:
        """Перед фразой должна быть пауза: два тихих блока подряд (0,5 с).

        Один тихий блок — это зазор между словами в чужой речи, его мало.
        В начале прослушивания истории ещё нет, поэтому первые блоки
        пропускаем: иначе первая же фраза после запуска не сработает.
        """
        history = list(self._quiet)
        if len(history) < 3:
            return True
        return any(left and right for left, right in zip(history, history[1:], strict=False))

    def reset(self) -> None:
        # Vosk Reset() оставляет часть состояния: после команды следующая
        # фраза может распознаться хуже. Новый recognizer стоит доли мс,
        # зато поток начинается с чистого листа.
        self.recognizer = self._new_recognizer()
        self._quiet.clear()
        self._mute_until = 0.0


@dataclass(frozen=True)
class BargeInMatch:
    """Что услышали во время озвучки: команду остановки или wake-фразу."""

    action: str
    phrase: str


class BargeInDetector:
    """Детектор перебивания на том же Vosk-модели, что и wake-слово.

    Распознаём речь без грамматики: ограниченная грамматика из стоп-слов
    слишком легко «слышит» их в собственной озвучке. Свободное распознавание
    транскрибирует ответ нормальными словами, а «стоп» ищется в конце фразы.
    """

    def __init__(
        self,
        model: object,
        aliases: list[str],
        sample_rate: int = 16000,
        stop_words: list[str] | tuple[str, ...] | None = None,
        min_level: float = BARGE_MIN_LEVEL,
    ) -> None:
        self.aliases = [normalize(alias) for alias in aliases if normalize(alias)]
        self.aliases_folded = [fold_wake_text(alias) for alias in self.aliases]
        words = stop_words or DEFAULT_STOP_WORDS
        self.stop_words = [normalize(word) for word in words if normalize(word)]
        if not self.aliases or not self.stop_words:
            raise ValueError("не заданы wake-алиасы или стоп-слова")

        self.sample_rate = int(sample_rate)
        self.min_level = max(0.0, float(min_level))
        self.model = model
        self.recognizer = self._new_recognizer()
        log.info(
            "Детектор перебивания готов: %s / стоп: %s (порог %.3f)",
            ", ".join(self.aliases),
            ", ".join(self.stop_words),
            self.min_level,
        )

    def _new_recognizer(self):
        recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)
        return recognizer

    def feed(self, pcm: bytes) -> BargeInMatch | None:
        if not pcm:
            return None
        prepared = condition_pcm(pcm, self.min_level, BARGE_TARGET_LEVEL)
        if not prepared:
            return None
        if self.recognizer.AcceptWaveform(prepared):
            raw = json.loads(self.recognizer.Result()).get("text", "")
            if raw:
                # Финальные фразы пишем в лог: по ним видно, что именно
                # услышал детектор, если перебивание не сработало.
                log.info("Перебивание слышит: %r", raw)
        else:
            raw = json.loads(self.recognizer.PartialResult()).get("partial", "")

        text = normalize(raw)
        if not text:
            return None
        log.debug("Перебивание слышит: %r", text)

        # «Стоп» ищем только в конце фразы: если слово стоит в середине,
        # это скорее всего пересказ или собственная озвучка, а не команда.
        # Запас в три слова нужен для «стоп пожалуйста» и «ну всё, стоп».
        for word in text.split()[-3:]:
            if word in self.stop_words:
                return BargeInMatch("stop", word)

        folded = fold_wake_text(text)
        flat = folded.replace(" ", "")
        # Сначала точное совпадение: «эй луна» не должно матчиться как
        # «хей луна» с одной опечаткой и подменять собой название фразы.
        for alias, folded_alias in zip(self.aliases, self.aliases_folded, strict=True):
            if folded_alias in folded or folded_alias.replace(" ", "") in flat:
                return BargeInMatch("wake", alias)

        for alias, folded_alias in zip(self.aliases, self.aliases_folded, strict=True):
            # «хей луна открой ютуб» приходит одним транскриптом; фраза
            # может начинаться не с первого слова, если в буфере остался
            # хвост озвучки. Склеенные варианты («хэлуна») тоже ловим.
            if fuzzy_contains(
                folded,
                folded_alias,
                max_distance=allowed_errors(folded_alias),
                merged=True,
            ):
                return BargeInMatch("wake", alias)
        return None

    def reset(self) -> None:
        self.recognizer = self._new_recognizer()


class OpenWakeWordDetector:
    """Лёгкий детектор одной фразы на openWakeWord (ONNX).

    В отличие от Vosk не распознаёт речь, а ищет акустический паттерн
    конкретной фразы. Поэтому почти не нагружает CPU и не путает похожие
    слова — но и работает только с той фразой, под которую обучена.
    """

    def __init__(self, model_path: Path, threshold: float = 0.5) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Модель openWakeWord не найдена: {model_path}. "
                "Скачайте её в папку моделей (см. README)."
            )
        try:
            import openwakeword
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("openWakeWord не установлен: pip install openwakeword") from exc

        # openWakeWord качает общий feature extractor отдельно от пакета.
        features = [Path(item["model_path"]) for item in openwakeword.FEATURE_MODELS.values()]
        if any(not path.exists() for path in features):
            log.info("Загружаю служебные модели openWakeWord...")
            download_models()

        self.model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
        self.name = next(iter(self.model.models))
        self.threshold = float(threshold)
        log.info(
            "Детектор openWakeWord готов: %s (порог %.2f)",
            self.name,
            self.threshold,
        )

    def feed(self, pcm: bytes) -> bool:
        if not pcm:
            return False
        import numpy as np

        samples = np.frombuffer(pcm, dtype=np.int16)
        scores = self.model.predict(samples)
        score = float(scores.get(self.name, 0.0))
        if score >= self.threshold:
            log.info("Ключевое слово: %s (score %.2f)", self.name, score)
            return True
        return False

    def reset(self) -> None:
        self.model.reset()
