"""Озвучка ответов: предзаписанные фразы и синтез через ProxyAPI.

Предзаписанные фразы играют с диска за миллисекунды. Динамические ответы
режутся на предложения и синтезируются параллельно: первое предложение
начинает звучать, пока остальные ещё генерируются.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import threading
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .config import DATA_DIR, ROOT

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100
LEAD_SEC = 0.015
TAIL_SEC = 0.060
FADE_SEC = 0.005
RMS_TARGET_DB = -20.0
PEAK_LIMIT = 0.90
DEFAULT_URL = "https://api.proxyapi.ru/v1/audio/speech"
DEFAULT_MODEL = "fish-audio/drama-3"


def normalize_pcm(pcm: bytes) -> np.ndarray | None:
    """Обрезает тишину, сглаживает края и выравнивает громкость."""
    if len(pcm) < 4 or len(pcm) % 2:
        return None
    data = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(data)))
    if peak < 1e-4:
        return None

    threshold = max(peak * 0.02, 0.004)
    loud = np.where(np.abs(data) > threshold)[0]
    if len(loud) == 0:
        return None
    start = max(0, int(loud[0]) - int(LEAD_SEC * SAMPLE_RATE))
    end = min(len(data), int(loud[-1]) + int(TAIL_SEC * SAMPLE_RATE))
    data = data[start:end]

    fade = int(FADE_SEC * SAMPLE_RATE)
    if len(data) > fade * 2:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        data[:fade] *= ramp
        data[-fade:] *= ramp[::-1]

    rms = float(np.sqrt(np.mean(data**2)))
    if rms > 1e-6:
        gain = (10.0 ** (RMS_TARGET_DB / 20.0)) / rms
        scaled_peak = float(np.max(np.abs(data))) * gain
        if scaled_peak > PEAK_LIMIT:
            gain *= PEAK_LIMIT / scaled_peak
        data = data * gain

    return np.clip(data, -1.0, 1.0)


def save_wav(path: str | Path, data: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(target), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(pcm)


def load_wav(path: str | Path) -> np.ndarray | None:
    try:
        with wave.open(str(path), "rb") as file:
            if file.getnchannels() != 1 or file.getsampwidth() != 2:
                return None
            raw = file.readframes(file.getnframes())
    except Exception as exc:  # noqa: BLE001
        log.warning("TTS: не удалось прочитать %s: %s", path, exc)
        return None
    if not raw:
        return None
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


SOURCES_RE = re.compile(r"(?im)^\s*источники\s*:?\s*$")
URL_RE = re.compile(r"https?://\S+")
MARKDOWN_RE = re.compile(r"[*_`#>|]+")
CITATION_RE = re.compile(r"\[\d+\]")


def clean_for_speech(text: str) -> str:
    """Убирает из ответа ссылки, разметку и номера источников."""
    text = text or ""
    text = SOURCES_RE.split(text)[0]
    text = CITATION_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MARKDOWN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text.strip()


SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str, limit: int = 180) -> list[str]:
    """Режет текст на предложения, длинные предложения — по границам слов."""
    chunks: list[str] = []
    for part in SENTENCE_RE.split(text or ""):
        part = part.strip()
        if not part:
            continue
        while len(part) > limit:
            cut = part.rfind(" ", 0, limit)
            if cut < limit // 3:
                cut = limit
            chunks.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            chunks.append(part)

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < 25:
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged


class PhraseBank:
    """Предзаписанные фразы: категория → варианты текста → дубли."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._by_category: dict[str, list[tuple[str, list[Path]]]] = {}
        self._by_id: dict[str, list[Path]] = {}
        self._last_id: dict[str, str] = {}
        self._last_file: dict[str, Path] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        manifest = self.directory / "manifest.json"
        if not manifest.exists():
            log.warning("TTS: нет манифеста фраз %s", manifest)
            return
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("TTS: не удалось прочитать манифест: %s", exc)
            return
        for item in data.get("phrases") or []:
            files = [
                self.directory / str(take.get("file"))
                for take in item.get("takes") or []
                if take.get("file")
            ]
            files = [path for path in files if path.exists()]
            if not files:
                continue
            phrase_id = str(item.get("id") or "")
            category = str(item.get("category") or "")
            if phrase_id:
                self._by_id[phrase_id] = files
            if category:
                self._by_category.setdefault(category, []).append((phrase_id, files))
        log.info(
            "TTS: фраз загружено: %d категорий, %d файлов",
            len(self._by_category),
            sum(len(files) for files in self._by_id.values()),
        )

    def categories(self) -> list[str]:
        return sorted(self._by_category)

    def pick(self, category: str) -> Path | None:
        with self._lock:
            groups = self._by_category.get(category)
            if not groups:
                return None
            last_id = self._last_id.get(category)
            pool = [group for group in groups if group[0] != last_id] or groups
            phrase_id, files = random.choice(pool)
            last_file = self._last_file.get(category)
            choices = [path for path in files if path != last_file] or files
            path = random.choice(choices)
            self._last_id[category] = phrase_id
            self._last_file[category] = path
            return path

    def pick_id(self, phrase_id: str) -> Path | None:
        with self._lock:
            files = self._by_id.get(phrase_id)
            if not files:
                return None
            last_file = self._last_file.get(phrase_id)
            choices = [path for path in files if path != last_file] or files
            path = random.choice(choices)
            self._last_file[phrase_id] = path
            return path


class TTSService:
    def __init__(self, cfg: dict[str, Any]) -> None:
        section = (cfg or {}).get("tts") or {}
        self.enabled = bool(section.get("enabled", False))
        self.base_url = str(section.get("base_url") or DEFAULT_URL)
        self.model = str(section.get("model") or DEFAULT_MODEL)
        self.voice = str(section.get("voice") or "").strip()
        self.api_key = str(section.get("api_key") or "").strip()
        self.timeout = float(section.get("timeout_sec") or 60)
        self.max_chars = int(section.get("max_chars") or 900)
        self.speak_answers = bool(section.get("speak_answers", True))
        self.volume = float(section.get("volume", 1.0))
        self.speed = float(section.get("speed", 1.0))
        self.workers = max(1, int(section.get("workers") or 3))
        self.cache_dir = Path(section.get("cache_dir") or (DATA_DIR / "tts-cache"))
        self.bank = PhraseBank(section.get("phrases_dir") or (ROOT / "assets" / "phrases"))
        if self.enabled:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                log.warning("TTS: не удалось создать кэш %s: %s", self.cache_dir, exc)

        self._speaking = threading.Event()
        self._stop = threading.Event()
        self._play_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.api_key) and bool(self.voice)

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    # ------------------------------------------------------------------ public
    def phrase(self, category: str, phrase_id: str | None = None) -> bool:
        """Играет предзаписанную фразу. Возвращает True, если что-то сыграло."""
        if not self.enabled:
            return False
        path = self.bank.pick_id(phrase_id) if phrase_id else self.bank.pick(category)
        if path is None:
            return False
        data = load_wav(path)
        if data is None:
            return False
        return self._play(data)

    def say(self, text: str) -> bool:
        """Озвучивает произвольный текст: синтез по предложениям параллельно."""
        if not self.available or not self.speak_answers:
            return False
        clean = clean_for_speech(text)
        if not clean:
            return False
        if len(clean) > self.max_chars:
            clean = clean[: self.max_chars].rsplit(" ", 1)[0] + "."
        chunks = split_sentences(clean)
        if not chunks:
            return False

        self._stop.clear()
        played = False
        pool = ThreadPoolExecutor(max_workers=self.workers)
        try:
            futures = [pool.submit(self._synth_cached, chunk) for chunk in chunks]
            for future in futures:
                if self._stop.is_set():
                    break
                try:
                    data = future.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning("TTS: фрагмент не синтезирован: %s", exc)
                    continue
                if data is None:
                    continue
                if self._play(data):
                    played = True
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return played

    def stop(self) -> None:
        self._stop.set()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ synth
    def _cache_path(self, text: str) -> Path:
        key = hashlib.sha256(f"{self.model}|{self.voice}|{text}".encode()).hexdigest()[:32]
        return self.cache_dir / f"{key}.wav"

    def _synth_cached(self, text: str) -> np.ndarray | None:
        path = self._cache_path(text)
        if path.exists():
            cached = load_wav(path)
            if cached is not None:
                return cached
        pcm = self._request(text)
        data = normalize_pcm(pcm)
        if data is None:
            return None
        try:
            save_wav(path, data)
        except Exception as exc:  # noqa: BLE001
            log.debug("TTS: не удалось сохранить кэш: %s", exc)
        return data

    def _request(self, text: str, retries: int = 3) -> bytes:
        payload = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": "pcm",
        }
        if self.speed != 1.0:
            payload["speed"] = self.speed
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        delay = 0.6
        last_error: Exception | None = None
        for attempt in range(retries):
            request = urllib.request.Request(
                self.base_url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            if attempt + 1 < retries:
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"ProxyAPI недоступен: {last_error}")

    # ------------------------------------------------------------------ audio
    def _play(self, data: np.ndarray) -> bool:
        if self._stop.is_set():
            return False
        if self.volume != 1.0:
            data = np.clip(data * self.volume, -1.0, 1.0)
        self._speaking.set()
        try:
            import sounddevice as sd

            with self._play_lock:
                sd.play(data, samplerate=SAMPLE_RATE, blocking=True)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("TTS: не удалось воспроизвести: %s", exc)
            return False
        finally:
            self._speaking.clear()
