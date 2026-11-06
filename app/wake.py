from __future__ import annotations

import json
import logging
from pathlib import Path

import vosk

from .text import fuzzy_contains, normalize

log = logging.getLogger(__name__)


def allowed_errors(alias: str) -> int:
    """Сколько опечаток распознавания допускать в ключевой фразе.

    В коротком слове одна ошибка — это уже другое слово: «луна» → «уна»,
    «лупа», «руна». Поэтому такие фразы сравниваем точно.
    """
    if len(alias.split()) > 1:
        return 1 if len(alias) >= 10 else 0
    return 0 if len(alias) <= 6 else 1


class WakeWordDetector:
    """Офлайн-детектор ключевой фразы на Vosk с ограниченной грамматикой."""

    def __init__(
        self,
        model_dir: Path,
        aliases: list[str],
        sample_rate: int = 16000,
        confusables: list[str] | None = None,
        warm_up: bool = True,
    ) -> None:
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Модель Vosk не найдена: {model_dir}. Запустите scripts\\setup.ps1"
            )
        self.aliases = [normalize(alias) for alias in aliases if normalize(alias)]
        if not self.aliases:
            raise ValueError("В config.json не заданы wake_aliases")

        self.sample_rate = int(sample_rate)
        vosk.SetLogLevel(-1)
        self.model = vosk.Model(str(model_dir))
        # Если в грамматике только «луна», декодер обязан услышать её даже
        # в «уна» или «дума». Похожие слова дают ему альтернативу.
        extra = [normalize(word) for word in (confusables or []) if normalize(word)]
        vocabulary = list(dict.fromkeys([*self.aliases, *extra]))
        self.grammar = json.dumps([*vocabulary, "[unk]"], ensure_ascii=False)
        self.recognizer = self._new_recognizer()
        log.info(
            "Детектор ключевого слова готов: %s (похожих слов: %d)",
            ", ".join(self.aliases),
            len(extra),
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
        recognizer = vosk.KaldiRecognizer(
            self.model, self.sample_rate, self.grammar)
        recognizer.SetWords(True)
        return recognizer

    def feed(self, pcm: bytes) -> bool:
        if not pcm:
            return False
        if self.recognizer.AcceptWaveform(pcm):
            raw = json.loads(self.recognizer.Result()).get("text", "")
        else:
            raw = json.loads(self.recognizer.PartialResult()).get("partial", "")

        text = normalize(raw)
        if not text:
            return False
        for alias in self.aliases:
            if fuzzy_contains(text, alias, max_distance=allowed_errors(alias)):
                log.info("Ключевое слово: «%s» (услышано: «%s»)", alias, text)
                return True
        return False

    def reset(self) -> None:
        # Vosk Reset() оставляет часть состояния: после команды следующая
        # фраза может распознаться хуже. Новый recognizer стоит доли мс,
        # зато поток начинается с чистого листа.
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
            raise RuntimeError(
                "openWakeWord не установлен: pip install openwakeword"
            ) from exc

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
