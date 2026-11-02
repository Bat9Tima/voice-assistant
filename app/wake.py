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
    ) -> None:
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Модель Vosk не найдена: {model_dir}. Запустите scripts\\setup.ps1"
            )
        self.aliases = [normalize(alias) for alias in aliases if normalize(alias)]
        if not self.aliases:
            raise ValueError("В config.json не заданы wake_aliases")

        vosk.SetLogLevel(-1)
        self.model = vosk.Model(str(model_dir))
        # Если в грамматике только «луна», декодер обязан услышать её даже
        # в «уна» или «дума». Похожие слова дают ему альтернативу.
        extra = [normalize(word) for word in (confusables or []) if normalize(word)]
        vocabulary = list(dict.fromkeys([*self.aliases, *extra]))
        grammar = json.dumps([*vocabulary, "[unk]"], ensure_ascii=False)
        self.recognizer = vosk.KaldiRecognizer(self.model, sample_rate, grammar)
        self.recognizer.SetWords(True)
        log.info(
            "Детектор ключевого слова готов: %s (похожих слов: %d)",
            ", ".join(self.aliases),
            len(extra),
        )

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
        self.recognizer.Reset()
