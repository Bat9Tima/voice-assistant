from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
import uuid
import wave
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import GIGAAM_DIR, VOSK_EN_DIR, WHISPER_DIR
from .text import normalize

log = logging.getLogger(__name__)

# Типичные галлюцинации Whisper на тишине/шуме.
_HALLUCINATIONS = {
    "продолжение следует",
    "субтитры сделал dimatorzok",
    "редактор субтитров а симанков корректор а иванова",
    "спасибо за просмотр",
    "подпишитесь на канал",
    "ставьте лайки",
}

# Слова, после которых начинается поисковый запрос. Нужны, чтобы отличить
# русскую команду от английской фразы, которую GigaAM записал кириллицей.
_RU_COMMAND_WORDS = {
    "найди",
    "поищи",
    "загугли",
    "погугли",
    "ищи",
    "включи",
    "поставь",
    "запусти",
    "покажи",
    "открой",
}
_RU_PLAY_WORDS = {"включи", "поставь", "запусти", "играй", "слушай"}
_RU_SITE_WORDS = {
    "ютуб",
    "ютубе",
    "ютуба",
    "youtube",
    "саундклауд",
    "саундклауде",
    "саундклауда",
    "саундклоуд",
    "soundcloud",
    "саунтлауди",
    "саунтлауде",
    "саундклауди",
    "санклауд",
    "санклауде",
    "гугл",
    "гугле",
    "google",
}
_RU_SITE_PREPOSITIONS = {"на", "в", "у", "для", "on", "in"}
_EN_COMMAND_WORDS = {
    "find",
    "search",
    "look",
    "lookup",
    "google",
    "play",
    "open",
    "show",
    "start",
    "put",
}
_EN_LEADING_FILLERS = {"for", "me", "up"}

_LATIN_RE = re.compile(r"[a-z]", re.IGNORECASE)
_CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
_WORD_KEY_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)
_EN_SITE_RE = re.compile(
    r"\s+(?:on|in)\s+(?:you\s*tube|sound\s*cloud|google)\s*$",
    re.IGNORECASE,
)
_EN_SITE_TAIL_RE = re.compile(r"\s+(?:you\s*tube|sound\s*cloud)\s*$", re.IGNORECASE)

# Ниже этого среднего log-probability Whisper обычно выдаёт транслитерацию
# русской речи, а не настоящий английский текст.
_MIN_ENGLISH_LOGPROB = -0.85
# Запасной путь без Vosk: короткий или неуверенный английский не дописываем.
_MIN_FALLBACK_ENGLISH_LOGPROB = -0.7
# Если после команды почти нет звука, полный проход Whisper не нужен.
_MIN_FALLBACK_TAIL_SEC = 1.0
# Если GigaAM уверена в русском тексте, английскую транслитерацию не трогаем.
_RU_CONFIDENT = 0.85
# Длина паузы, после которой слова Vosk считаются разными английскими фразами.
_SPAN_GAP_SEC = 0.45
_SPAN_PADDING_SEC = 0.25
# Если GigaAM уверена в русском тексте, английским считается только фрагмент,
# который почти не перекрывается русскими словами (то есть попал в «дыру»).
_SPAN_MAX_RU_OVERLAP = 0.2

# Deepgram через ProxyAPI: только batch /v1/audio/transcriptions.
# WebSocket /v1/listen публично не документирован, поэтому стриминга здесь нет.
DEFAULT_DEEPGRAM_URL = "https://api.proxyapi.ru/v1/audio/transcriptions"
DEFAULT_DEEPGRAM_MODEL = "deepgram/nova-3"


@dataclass(frozen=True)
class Word:
    """Слово с временными границами в секундах."""

    text: str
    start: float
    end: float
    probability: float = 1.0


@dataclass(frozen=True)
class EnglishSpan:
    """Распознанный английский фрагмент и его место в аудио."""

    text: str
    start: float
    end: float
    avg_logprob: float = 0.0


@dataclass
class Transcript:
    """Результат одного движка распознавания."""

    text: str
    words: list[Word] = field(default_factory=list)
    language: str | None = None
    confidence: float = 1.0
    segments: list[str] = field(default_factory=list)
    avg_logprob: float | None = None


@dataclass
class TranscriptionResult:
    """Итог параллельного распознавания."""

    text: str
    russian: Transcript
    english: Transcript | None = None
    source: str = "ru"


def _word_key(text: str) -> str:
    return normalize(_WORD_KEY_RE.sub("", text or ""))


def _has_latin(text: str) -> bool:
    return bool(_LATIN_RE.search(text or ""))


def _has_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text or ""))


def _join_words(words: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(part for part in words if part)).strip(" ,.-")


def _clean_english_text(text: str) -> str:
    """Убирает пунктуацию по краям, сохраняя дефисы и точки внутри слов."""
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"^[\s.,!?;:«»\"']+", "", text)
    text = re.sub(r"[\s.,!?;:«»\"']+$", "", text)
    return text


def _strip_leading_command(text: str) -> str:
    parts = text.split()
    while parts and _word_key(parts[0]) in _EN_COMMAND_WORDS:
        parts.pop(0)
        while parts and _word_key(parts[0]) in _EN_LEADING_FILLERS:
            parts.pop(0)
    return _join_words(parts)


def _strip_site_tail(text: str) -> str:
    text = _EN_SITE_RE.sub("", text)
    text = _EN_SITE_TAIL_RE.sub("", text)
    return _clean_english_text(text)


def _gigaam_words(
    tokens: list[str],
    timestamps: list[float],
    log_probs: list[float] | None = None,
) -> list[Word]:
    """GigaAM отдаёт посимвольные токены — собираем из них слова."""
    words: list[Word] = []
    current: list[str] = []
    current_probs: list[float] = []
    start = 0.0
    last = 0.0
    probs = list(log_probs or [])

    def flush(end: float | None = None) -> None:
        nonlocal current, current_probs, start, last
        if current:
            probability = 1.0
            if current_probs:
                probability = float(np.exp(np.mean(current_probs)))
            words.append(
                Word(
                    text="".join(current),
                    start=start,
                    end=end if end is not None else last + 0.1,
                    probability=probability,
                )
            )
            current = []
            current_probs = []

    for index, (token, timestamp) in enumerate(zip(tokens, timestamps, strict=False)):
        token = str(token)
        timestamp = float(timestamp)
        probability = float(probs[index]) if index < len(probs) else 0.0
        if token.isspace():
            # Пробел означает границу слова; его время — удобный конец слова.
            flush(timestamp)
            continue
        if not current:
            start = timestamp
        current.append(token)
        current_probs.append(probability)
        last = timestamp
    flush()
    return words


def _site_prefix(text: str) -> str | None:
    """Возвращает название сайта, слипшееся с английским словом.

    GigaAM может выдать «ютубекадс» вместо «ютубе кэтс». Для склейки важно
    сохранить «ютубе», а хвост заменить английским текстом.
    """
    key = _word_key(text)
    if key in _RU_SITE_WORDS:
        return None
    matches = [site for site in _RU_SITE_WORDS if len(site) >= 4 and key.startswith(site)]
    return max(matches, key=len) if matches else None


def _is_site_word(text: str) -> bool:
    key = _word_key(text)
    return key in _RU_SITE_WORDS or _site_prefix(text) is not None


def _word_text(text: str) -> str:
    return _site_prefix(text) or text


def _preserved_indices(words: list[Word], trigger_index: int) -> set[int]:
    """Индексы слов, которые нельзя заменять английским текстом."""
    preserved = {index for index in range(trigger_index + 1)}
    for index, word in enumerate(words):
        if _word_key(word.text) in _RU_SITE_WORDS:
            preserved.add(index)
            if index > 0 and _word_key(words[index - 1].text) in _RU_SITE_PREPOSITIONS:
                preserved.add(index - 1)
    return preserved


def _textual_merge(ru: Transcript, en: Transcript | None) -> tuple[str, str]:
    """Склейка без таймстемпов — запасной путь и основа unit-тестов."""
    ru_text = _clean_english_text(ru.text)
    if not en:
        return ru_text, "ru"
    en_text = _clean_english_text(en.text)
    if not en_text or not _has_latin(en_text):
        return ru_text, "ru"

    ru_words = list(ru.words)
    trigger_index = next(
        (index for index, word in enumerate(ru_words) if _word_key(word.text) in _RU_COMMAND_WORDS),
        None,
    )
    en_has_trigger = any(_word_key(word) in _EN_COMMAND_WORDS for word in en_text.split())

    # Уверенная русская команда: английский «мусор» без английской команды
    # не дописываем. Настоящая английская речь в «дыре» обрабатывается
    # отдельным путём со spans; сюда попадаем, только если её не нашли.
    if ru.confidence >= _RU_CONFIDENT and not en_has_trigger:
        return ru_text, "ru"

    if trigger_index is None:
        if not ru_text or not _has_cyrillic(ru_text):
            return en_text, "en"
        # Чистая английская команда, которую GigaAM записал кириллицей
        # («оупен ютуб»), уверенно определяется по английскому триггеру.
        if en_has_trigger and (en.avg_logprob is None or en.avg_logprob >= _MIN_ENGLISH_LOGPROB):
            return en_text, "en"
        if en_has_trigger and ru.confidence < _RU_CONFIDENT:
            return en_text, "en"
        if (
            not en_has_trigger
            and ru.confidence < 0.80
            and len(en_text.split()) >= 2
            and (en.avg_logprob is None or en.avg_logprob >= _MIN_ENGLISH_LOGPROB)
        ):
            return en_text, "en"
        return ru_text, "ru"

    query = _clean_english_text(en_text)
    if en_has_trigger:
        query = _strip_leading_command(query)
    elif len(query.split()) < 2 or (
        en.avg_logprob is not None and en.avg_logprob < _MIN_FALLBACK_ENGLISH_LOGPROB
    ):
        # Без Vosk легко принять «Bye-bye» или русскую транслитерацию за
        # английский запрос. Короткий/неуверенный текст не дописываем.
        return ru_text, "ru"

    site_indices = {index for index, word in enumerate(ru_words) if _is_site_word(word.text)}
    for index in list(site_indices):
        if index > 0 and _word_key(ru_words[index - 1].text) in _RU_SITE_PREPOSITIONS:
            site_indices.add(index - 1)
    if site_indices:
        query = _strip_site_tail(query)
    if not query:
        return ru_text, "ru"

    # Сайт сразу после команды — это префикс («найди на ютубе …»),
    # иначе сайт стоит в конце («включи … на саундклауде»).
    site_min = min(site_indices) if site_indices else None
    site_max = max(site_indices) if site_indices else None
    if site_min is not None and site_min <= trigger_index + 2:
        site_at_end = site_max == len(ru_words) - 1
        play_command = _word_key(ru_words[trigger_index].text) in _RU_PLAY_WORDS
        if not (site_at_end and play_command):
            # Сайт-префикс: «найди на ютубе …».
            kept = [_word_text(word.text) for word in ru_words[: site_max + 1]]
            return _join_words([*kept, query]), "mixed"
    if site_min is not None:
        # Сайт-суффикс: «включи … на саундклауде».
        kept = [word.text for word in ru_words[: trigger_index + 1]]
        tail = [_word_text(word.text) for word in ru_words[site_min:]]
        return _join_words([*kept, query, *tail]), "mixed"

    kept = [word.text for word in ru_words[: trigger_index + 1]]
    return _join_words([*kept, query]), "mixed"


def _spans_from_vosk(
    russian_words: list[Word],
    trigger_index: int,
    locator_words: list[Word],
) -> list[EnglishSpan]:
    """Собирает английские интервалы по словам Vosk, исключая русские слова."""
    if not locator_words:
        return []
    preserved = _preserved_indices(russian_words, trigger_index)
    preserved_words = [russian_words[index] for index in sorted(preserved)]

    def overlaps_preserved(word: Word) -> bool:
        return any(word.end >= item.start and word.start <= item.end for item in preserved_words)

    candidates = [
        word
        for word in locator_words
        if _has_latin(word.text) and word.probability >= 0.35 and not overlaps_preserved(word)
    ]
    if not candidates:
        return []

    candidates.sort(key=lambda word: word.start)
    groups: list[list[Word]] = []
    for word in candidates:
        if groups and word.start - groups[-1][-1].end <= _SPAN_GAP_SEC:
            groups[-1].append(word)
        else:
            groups.append([word])

    spans: list[EnglishSpan] = []
    for group in groups:
        start = min(word.start for word in group)
        end = max(word.end for word in group)
        if end - start < 0.15:
            continue
        text = _join_words([word.text for word in group])
        if text:
            spans.append(EnglishSpan(text=text, start=start, end=end))
    spans.sort(key=lambda span: span.start)
    return spans[:2]


def _span_overlap_ratio(span: EnglishSpan, words: list[Word]) -> float:
    """Какая доля английского интервала занята русскими словами."""
    duration = span.end - span.start
    if duration <= 0:
        return 1.0
    overlap = 0.0
    for word in words:
        left = max(span.start, word.start)
        right = min(span.end, word.end)
        if right > left:
            overlap += right - left
    return overlap / duration


def _gap_spans(russian: Transcript, spans: list[EnglishSpan]) -> list[EnglishSpan]:
    """Оставляет английские фрагменты только в промежутках русской речи.

    Vosk на русской речи выдаёт уверенный английский «мусор» с таймстемпами
    поверх русских слов. Если GigaAM уверена в русском тексте, такие фрагменты
    нельзя вставлять в команду — принимаем только английскую речь в «дырах».
    """
    if russian.confidence < _RU_CONFIDENT:
        return spans
    return [
        span for span in spans if _span_overlap_ratio(span, russian.words) <= _SPAN_MAX_RU_OVERLAP
    ]


def _trim_site_words(words: list[Word], spans: list[EnglishSpan]) -> list[Word]:
    """Отрезает от «ютубекадс» английский хвост, оставляя «ютубе»."""
    trimmed: list[Word] = []
    for word in words:
        for span in spans:
            if word.end > span.start and word.start < span.end:
                prefix = _site_prefix(word.text)
                if prefix and word.start < span.start:
                    word = Word(
                        text=prefix,
                        start=word.start,
                        end=min(word.end, span.start),
                        probability=word.probability,
                    )
                break
        trimmed.append(word)
    return trimmed


def _merge_with_spans(ru: Transcript, spans: list[EnglishSpan]) -> str:
    """Вставляет английские фрагменты в русский текст по времени."""
    result: list[str] = []
    span_index = 0
    for word in _trim_site_words(ru.words, spans):
        while span_index < len(spans) and spans[span_index].start < word.start:
            result.append(spans[span_index].text)
            span_index += 1
        if any(word.end > span.start and word.start < span.end for span in spans):
            continue
        result.append(word.text)
    while span_index < len(spans):
        result.append(spans[span_index].text)
        span_index += 1
    return _join_words(result)


def merge_transcripts(primary: Transcript, secondary: Transcript | None) -> str:
    """Публичная обёртка для тестов и внешних вызовов."""
    return _textual_merge(primary, secondary)[0]


class GigaAMTranscriber:
    """Русское распознавание речи через GigaAM v3 и sherpa-onnx (CPU)."""

    def __init__(
        self,
        model_dir: Path = GIGAAM_DIR,
        num_threads: int = 4,
        tail_padding_sec: float = 0.4,
        sample_rate: int = 16000,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.num_threads = max(1, int(num_threads))
        self.tail_padding_sec = max(0.0, float(tail_padding_sec))
        self.sample_rate = int(sample_rate)
        self._recognizer = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._recognizer is not None

    @property
    def english_loaded(self) -> bool:
        """У GigaAM нет английского движка — всегда False."""
        return False

    def _encoder_path(self) -> Path:
        int8 = self.model_dir / "encoder.int8.onnx"
        if int8.exists():
            return int8
        return self.model_dir / "encoder.onnx"

    def _validate(self) -> None:
        required = [
            self._encoder_path(),
            self.model_dir / "decoder.onnx",
            self.model_dir / "joiner.onnx",
            self.model_dir / "tokens.txt",
        ]
        missing = [path.name for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Модель GigaAM не найдена в {self.model_dir}: нет {', '.join(missing)}. "
                "Запустите scripts\\setup.cmd"
            )

    def load(self) -> None:
        if self._recognizer is not None:
            return
        with self._load_lock:
            if self._recognizer is not None:
                return
            self._validate()
            import sherpa_onnx

            log.info("Загружаю GigaAM v3 (%s)...", self.model_dir.name)
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(self._encoder_path()),
                decoder=str(self.model_dir / "decoder.onnx"),
                joiner=str(self.model_dir / "joiner.onnx"),
                tokens=str(self.model_dir / "tokens.txt"),
                model_type="nemo_transducer",
                # GigaAM использует 64 mel-бина, а не 80 как большинство моделей.
                feature_dim=64,
                sample_rate=self.sample_rate,
                num_threads=self.num_threads,
                decoding_method="greedy_search",
                debug=False,
            )
            # Прогрев: первая реальная команда не должна ждать инициализации ONNX.
            self._decode(np.zeros(self.sample_rate // 2, dtype=np.float32))
            log.info("GigaAM готова")

    def _decode(self, audio: np.ndarray) -> Transcript:
        assert self._recognizer is not None
        if self.tail_padding_sec > 0:
            padding = np.zeros(int(self.sample_rate * self.tail_padding_sec), dtype=np.float32)
            audio = np.concatenate([audio, padding])
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, audio)
        self._recognizer.decode_stream(stream)
        result = stream.result
        tokens = list(getattr(result, "tokens", []) or [])
        timestamps = list(getattr(result, "timestamps", []) or [])
        log_probs = list(getattr(result, "ys_log_probs", []) or [])
        confidence = 0.0
        if log_probs:
            confidence = float(np.exp(np.mean(log_probs)))
        return Transcript(
            text=str(result.text or "").strip(),
            words=_gigaam_words(tokens, timestamps, log_probs),
            language="ru",
            confidence=confidence,
        )

    def transcribe_details(self, pcm: bytes) -> Transcript:
        self.load()
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size < 1600:  # меньше 0,1 секунды
            return Transcript(text="", language="ru", confidence=0.0)
        return self._decode(audio)

    def transcribe(self, pcm: bytes) -> str:
        return self.transcribe_details(pcm).text


class WhisperTranscriber:
    """Распознавание через faster-whisper (CPU, int8)."""

    def __init__(
        self,
        model_name: str = "small",
        language: str | None = "ru",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        initial_prompt: str = "",
        num_threads: int = 4,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.num_threads = max(1, int(num_threads))
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel

            log.info("Загружаю Whisper «%s» (это может занять время)...", self.model_name)
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(WHISPER_DIR),
                cpu_threads=self.num_threads,
            )
            # Прогрев, чтобы первая реальная команда не ждала инициализации.
            silence = np.zeros(16000, dtype=np.float32)
            segments, _ = self._model.transcribe(
                silence, language=self.language, beam_size=1, vad_filter=False
            )
            list(segments)
            log.info("Whisper готов")

    def transcribe_details(self, pcm: bytes, beam_size: int | None = None) -> Transcript:
        self.load()
        assert self._model is not None
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size < 1600:  # меньше 0,1 секунды
            return Transcript(text="", language=self.language, avg_logprob=0.0)
        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=beam_size or self.beam_size,
            initial_prompt=self.initial_prompt or None,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        texts: list[str] = []
        logprobs: list[float] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                texts.append(text)
                logprobs.append(float(segment.avg_logprob))
        text = " ".join(texts).strip()
        if text.lower().strip(" .!?") in _HALLUCINATIONS:
            return Transcript(
                text="",
                language=getattr(info, "language", self.language),
                avg_logprob=0.0,
            )
        avg_logprob = float(np.mean(logprobs)) if logprobs else None
        return Transcript(
            text=text,
            language=getattr(info, "language", self.language),
            segments=texts,
            avg_logprob=avg_logprob,
        )

    def transcribe(self, pcm: bytes) -> str:
        return self.transcribe_details(pcm).text


class VoskEnglishLocator:
    """Быстрый локатор английских слов: даёт таймстемпы, но не финальный текст."""

    def __init__(self, model_dir: Path = VOSK_EN_DIR, sample_rate: int = 16000) -> None:
        self.model_dir = Path(model_dir)
        self.sample_rate = int(sample_rate)
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if not self.model_dir.exists():
                raise FileNotFoundError(
                    f"Модель Vosk EN не найдена: {self.model_dir}. Запустите scripts\\setup.cmd"
                )
            import vosk

            vosk.SetLogLevel(-1)
            log.info("Загружаю Vosk EN (%s)...", self.model_dir.name)
            self._model = vosk.Model(str(self.model_dir))
            log.info("Vosk EN готов")

    def transcribe_words(self, pcm: bytes) -> list[Word]:
        self.load()
        assert self._model is not None
        import vosk

        recognizer = vosk.KaldiRecognizer(self._model, self.sample_rate)
        recognizer.SetWords(True)
        recognizer.AcceptWaveform(pcm)
        result = json.loads(recognizer.FinalResult() or "{}")
        words: list[Word] = []
        for item in result.get("result", []):
            text = str(item.get("word") or "").strip()
            if not text:
                continue
            words.append(
                Word(
                    text=text,
                    start=float(item.get("start") or 0.0),
                    end=float(item.get("end") or 0.0),
                    probability=float(item.get("conf") or 0.0),
                )
            )
        return words


class ParallelTranscriber:
    """GigaAM + Vosk EN локатор + Whisper tiny для английских вставок."""

    def __init__(
        self,
        russian: GigaAMTranscriber,
        english: WhisperTranscriber | None = None,
        locator: VoskEnglishLocator | None = None,
        crop_beam_size: int = 5,
    ) -> None:
        self.russian = russian
        self.english = english
        self.locator = locator
        self.crop_beam_size = max(1, int(crop_beam_size))
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="stt")
        self._load_lock = threading.Lock()
        self._loaded = False
        self._english_error: str | None = None
        self._locator_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self.russian.loaded

    @property
    def english_loaded(self) -> bool:
        return bool(self.english and self.english.loaded)

    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            futures = [self._executor.submit(self.russian.load)]
            if self.english is not None:
                futures.append(self._executor.submit(self.english.load))
            if self.locator is not None:
                futures.append(self._executor.submit(self.locator.load))
            russian_future = futures[0]
            russian_future.result()
            for name, future in zip(("english", "locator"), futures[1:], strict=False):
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    if name == "english":
                        self._english_error = str(exc)
                    else:
                        self._locator_error = str(exc)
                    log.warning(
                        "%s распознавание недоступно: %s",
                        "Английское" if name == "english" else "Локатор английского",
                        exc,
                    )
            self._loaded = True

    def _crop_pcm(
        self,
        audio: np.ndarray,
        start: float,
        end: float,
        sample_rate: int = 16000,
    ) -> bytes:
        duration = len(audio) / sample_rate
        left = max(0.0, start - _SPAN_PADDING_SEC)
        right = min(duration, end + _SPAN_PADDING_SEC)
        chunk = audio[int(left * sample_rate) : int(right * sample_rate)]
        return (chunk * 32768.0).astype(np.int16).tobytes()

    def _english_spans(
        self,
        russian: Transcript,
        locator_words: list[Word],
        audio: np.ndarray,
    ) -> list[EnglishSpan]:
        russian_words = list(russian.words)
        trigger_index = next(
            (
                index
                for index, word in enumerate(russian_words)
                if _word_key(word.text) in _RU_COMMAND_WORDS
            ),
            None,
        )
        if trigger_index is None:
            return []
        if not locator_words:
            return []

        candidates = _spans_from_vosk(russian_words, trigger_index, locator_words)
        candidates = _gap_spans(russian, candidates)
        if not candidates:
            return []
        if self.english is None or not self.english.loaded:
            return []

        accepted: list[EnglishSpan] = []
        for span in candidates:
            crop = self._crop_pcm(audio, span.start, span.end)
            if len(crop) < 3200:
                continue
            try:
                result = self.english.transcribe_details(crop, beam_size=self.crop_beam_size)
            except Exception as exc:  # noqa: BLE001
                log.warning("Ошибка распознавания английского фрагмента: %s", exc)
                result = None
            if result is None or not result.text:
                continue
            text = _strip_site_tail(_strip_leading_command(result.text))
            avg = result.avg_logprob if result.avg_logprob is not None else -9.0
            if text and _has_latin(text) and avg >= _MIN_ENGLISH_LOGPROB:
                accepted.append(
                    EnglishSpan(text=text, start=span.start, end=span.end, avg_logprob=avg)
                )
        accepted.sort(key=lambda item: item.start)
        return accepted

    def _full_english(self, pcm: bytes) -> Transcript | None:
        if self.english is None or not self.english.loaded:
            return None
        try:
            return self.english.transcribe_details(pcm)
        except Exception as exc:  # noqa: BLE001
            log.warning("Ошибка английского распознавания: %s", exc)
            return None

    def _locator_words(self, audio: np.ndarray, trigger: Word) -> list[Word]:
        """Ищет английские слова только после команды — так Vosk быстрее."""
        if self.locator is None or not self.locator.loaded:
            return []
        start = max(0.0, float(trigger.end) - 0.15)
        chunk = audio[int(start * 16000) :]
        if chunk.size < 1600:
            return []
        raw = (chunk * 32768.0).astype(np.int16).tobytes()
        try:
            words = self.locator.transcribe_words(raw)
        except Exception as exc:  # noqa: BLE001
            log.warning("Ошибка локатора английского: %s", exc)
            return []
        return [
            Word(
                text=word.text,
                start=word.start + start,
                end=word.end + start,
                probability=word.probability,
            )
            for word in words
        ]

    @staticmethod
    def _russian_result(
        russian: Transcript,
        english: Transcript | None = None,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            text=_clean_english_text(russian.text),
            russian=russian,
            english=english,
            source="ru",
        )

    def transcribe_details(self, pcm: bytes) -> TranscriptionResult:
        self.load()
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        russian = self.russian.transcribe_details(pcm)

        if not russian.text:
            english = self._full_english(pcm)
            text = _clean_english_text(english.text) if english and english.text else ""
            return TranscriptionResult(
                text=text,
                russian=russian,
                english=english,
                source="en" if text else "ru",
            )

        trigger_index = next(
            (
                index
                for index, word in enumerate(russian.words)
                if _word_key(word.text) in _RU_COMMAND_WORDS
            ),
            None,
        )
        if trigger_index is None or self.locator is None:
            # Нет русской команды (или отключён локатор) — нужен полный
            # английский проход: GigaAM мог записать английский кириллицей.
            english = self._full_english(pcm)
            if english is None or not english.text:
                return self._russian_result(russian, english)
            text, source = _textual_merge(russian, english)
            return TranscriptionResult(
                text=text,
                russian=russian,
                english=english,
                source=source,
            )

        locator_words = self._locator_words(audio, russian.words[trigger_index])
        spans = self._english_spans(russian, locator_words, audio)
        if spans:
            merged = _merge_with_spans(russian, spans)
            if merged and _has_latin(merged):
                return TranscriptionResult(
                    text=merged,
                    russian=russian,
                    english=Transcript(
                        text=_join_words([span.text for span in spans]),
                        language="en",
                    ),
                    source="mixed",
                )

        # Если после команды почти нет звука, полный проход Whisper не нужен:
        # Vosk уже не нашёл там английских слов, а Whisper на коротком хвосте
        # часто выдумывает «Bye-bye» или транслитерацию.
        duration = len(audio) / 16000
        if duration - float(russian.words[trigger_index].end) < _MIN_FALLBACK_TAIL_SEC:
            return self._russian_result(russian)

        # Полный проход Whisper — только запасной вариант. Если GigaAM
        # уверена в русском тексте, лишний английский «мусор» не нужен.
        if russian.confidence >= _RU_CONFIDENT:
            return self._russian_result(russian)

        english = self._full_english(pcm)
        if english is None or not english.text:
            return self._russian_result(russian, english)
        text, source = _textual_merge(russian, english)
        return TranscriptionResult(
            text=text,
            russian=russian,
            english=english,
            source=source,
        )

    def transcribe(self, pcm: bytes) -> str:
        return self.transcribe_details(pcm).text


class DeepgramError(RuntimeError):
    """Ошибка облачного распознавания Deepgram."""


class DeepgramTranscriber:
    """Batch-распознавание Deepgram Nova-3 через ProxyAPI.

    API отдаёт только готовый текст: без слов, таймкодов и языка отдельных
    фрагментов. Для команд этого достаточно. Для выравнивания RU/EN нужен
    стриминговый Deepgram напрямую, а не этот прокси.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_DEEPGRAM_URL,
        model: str = DEFAULT_DEEPGRAM_MODEL,
        language: str = "multi",
        language_hints: Iterable[str] = ("ru", "en"),
        api_key: str = "",
        api_key_env: str = "PROXYAPI_KEY",
        timeout: float = 15.0,
        sample_rate: int = 16000,
        fallback: Any = None,
    ) -> None:
        self.base_url = str(base_url)
        self.model = str(model)
        self.language = str(language).strip()
        self.language_hints = [str(hint).strip() for hint in language_hints if str(hint).strip()]
        self.api_key_env = str(api_key_env)
        self.timeout = max(1.0, float(timeout))
        self.sample_rate = int(sample_rate)
        self.fallback = fallback
        self._api_key = self._resolve_key(api_key)
        if not self._api_key:
            log.warning(
                "Deepgram: ключ не задан. Укажите stt.deepgram.api_key или переменную %s",
                self.api_key_env,
            )

    def _resolve_key(self, api_key: str) -> str:
        values = (
            api_key,
            os.environ.get(self.api_key_env),
            os.environ.get("DEEPGRAM_API_KEY"),
        )
        for value in values:
            if value and value.strip():
                return value.strip()
        return ""

    def load(self) -> None:
        """Модель живёт в облаке: локальной загрузки нет."""

    @property
    def loaded(self) -> bool:
        return bool(self._api_key) or self.fallback is not None

    @property
    def english_loaded(self) -> bool:
        """Deepgram multi распознаёт английский без отдельного движка."""
        return True

    def _fallback(self, pcm: bytes, reason: str) -> Transcript:
        if self.fallback is None:
            raise DeepgramError(f"Deepgram: {reason}")
        log.warning("Deepgram: %s. Перехожу на %s", reason, type(self.fallback).__name__)
        return self.fallback.transcribe_details(pcm)

    @staticmethod
    def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        return buffer.getvalue()

    def _build_body(self, wav: bytes) -> tuple[str, bytes]:
        boundary = f"----Luna{uuid.uuid4().hex}"
        fields = [("model", self.model)]
        if self.language:
            fields.append(("language", self.language))
        fields.extend(("language_hint", hint) for hint in self.language_hints)
        chunks: list[bytes] = []
        for name, value in fields:
            chunks.append(f"--{boundary}\r\n".encode())
            chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            chunks.append(f"{value}\r\n".encode())
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(b'Content-Disposition: form-data; name="file"; filename="command.wav"\r\n')
        chunks.append(b"Content-Type: audio/wav\r\n\r\n")
        chunks.append(wav)
        chunks.append(f"\r\n--{boundary}--\r\n".encode())
        return boundary, b"".join(chunks)

    def _request(self, wav: bytes) -> dict[str, Any]:
        boundary, body = self._build_body(wav)
        request = urllib.request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise DeepgramError(f"HTTP {exc.code}: {detail}".strip()) from exc
        except OSError as exc:
            raise DeepgramError(str(exc)) from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeepgramError("ответ не является JSON") from exc
        if not isinstance(payload, dict):
            raise DeepgramError("неожиданный формат ответа")
        return payload

    def transcribe_details(self, pcm: bytes) -> Transcript:
        if not pcm:
            return Transcript(text="", language=self.language or None)
        if not self._api_key:
            return self._fallback(pcm, f"не задан ключ ({self.api_key_env})")
        try:
            payload = self._request(self._wav_bytes(pcm, self.sample_rate))
        except DeepgramError as exc:
            return self._fallback(pcm, str(exc))
        text = " ".join(str(payload.get("text") or "").split())
        if not text:
            return self._fallback(pcm, "пустой ответ")
        # Прокси не отдаёт confidence, поэтому доверие к тексту не размечаем.
        return Transcript(text=text, language=self.language or None, confidence=1.0)

    def transcribe(self, pcm: bytes) -> str:
        return self.transcribe_details(pcm).text


Transcriber = WhisperTranscriber

_GIGAAM_ALIASES = {"gigaam", "giga-am", "sherpa", "sherpa-onnx", "nemo"}
_WHISPER_ALIASES = {"whisper", "faster-whisper", "faster_whisper"}
_DEEPGRAM_ALIASES = {"deepgram", "nova", "nova-3", "nova3"}
_DISABLED_ALIASES = {"", "none", "off", "disabled"}


def create_transcriber(
    cfg: dict[str, Any],
) -> GigaAMTranscriber | WhisperTranscriber | ParallelTranscriber | DeepgramTranscriber:
    """Создаёт движок распознавания по config.json → stt.engine."""
    stt = cfg.get("stt") or {}
    engine = str(stt.get("engine") or "gigaam").strip().lower()

    if engine in _WHISPER_ALIASES:
        return WhisperTranscriber(
            model_name=str(stt.get("model") or "small"),
            language=str(cfg.get("language") or "ru"),
            device=str(stt.get("device") or "cpu"),
            compute_type=str(stt.get("compute_type") or "int8"),
            beam_size=int(stt.get("beam_size") or 5),
            initial_prompt=str(stt.get("initial_prompt") or ""),
            num_threads=int(stt.get("num_threads") or 4),
        )

    if engine in _DEEPGRAM_ALIASES:
        section = stt.get("deepgram") or {}
        fallback = None
        if bool(section.get("fallback", True)):
            fallback = GigaAMTranscriber(
                num_threads=int(stt.get("num_threads") or 4),
                tail_padding_sec=float(stt.get("tail_padding_sec", 0.4)),
            )
        hints = section.get("language_hints")
        if hints is None:
            hints = ("ru", "en")
        elif isinstance(hints, str):
            hints = [part.strip() for part in hints.split(",") if part.strip()]
        return DeepgramTranscriber(
            base_url=str(section.get("base_url") or DEFAULT_DEEPGRAM_URL),
            model=str(section.get("model") or DEFAULT_DEEPGRAM_MODEL),
            language=str(section.get("language") or "multi"),
            language_hints=hints,
            api_key=str(section.get("api_key") or ""),
            api_key_env=str(section.get("api_key_env") or "PROXYAPI_KEY"),
            timeout=float(section.get("timeout_sec") or 15),
            sample_rate=int((cfg.get("audio") or {}).get("sample_rate") or 16000),
            fallback=fallback,
        )

    if engine not in _GIGAAM_ALIASES:
        raise ValueError(
            f"Неизвестный stt.engine: {engine!r}. Допустимо: gigaam, whisper, deepgram"
        )

    russian = GigaAMTranscriber(
        num_threads=int(stt.get("num_threads") or 4),
        tail_padding_sec=float(stt.get("tail_padding_sec", 0.4)),
    )
    if not bool(stt.get("parallel_english", False)):
        return russian

    english_engine = str(stt.get("english_engine") or "whisper").strip().lower()
    if english_engine in _DISABLED_ALIASES:
        return russian
    if english_engine not in _WHISPER_ALIASES:
        raise ValueError(
            f"Неизвестный stt.english_engine: {english_engine!r}. "
            "Сейчас поддерживается только whisper."
        )

    english = WhisperTranscriber(
        model_name=str(stt.get("english_model") or "tiny"),
        # Английский движок всегда декодирует как английский: русские участки
        # нам не нужны, а автоопределение на смешанной речи ошибается.
        language="en",
        device=str(stt.get("english_device") or "cpu"),
        compute_type=str(stt.get("english_compute_type") or "int8"),
        beam_size=int(stt.get("english_beam_size") or 1),
        num_threads=int(stt.get("english_num_threads") or 4),
    )
    locator = None
    if str(stt.get("english_locator") or "vosk").strip().lower() not in _DISABLED_ALIASES:
        locator = VoskEnglishLocator()
    return ParallelTranscriber(
        russian,
        english,
        locator=locator,
        crop_beam_size=int(stt.get("english_crop_beam_size") or 5),
    )
