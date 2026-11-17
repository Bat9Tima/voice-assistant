"""Озвучка ответов: предзаписанные фразы и синтез через ProxyAPI.

Предзаписанные фразы играют с диска за миллисекунды. Динамические ответы
режутся на предложения и синтезируются параллельно: первое предложение
начинает звучать, пока остальные ещё генерируются.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import random
import re
import threading
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import DATA_DIR, ROOT

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100
# Куски, которыми звук уходит в PortAudio. Меньше — быстрее перебивание,
# больше — меньше вызовов и надёжнее на слабых машинах.
PLAY_CHUNK_SEC = 0.1
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


@dataclass
class _SpeechJob:
    """Очередь озвучки: фрагменты можно добавлять и после постановки в очередь."""

    chunks: list[str] = field(default_factory=list)
    generation: int = 0
    cancelled: threading.Event = field(default_factory=threading.Event)
    # Закрытая задача больше не ждёт новых фрагментов.
    closed: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, chunks: list[str]) -> None:
        with self.lock:
            self.chunks.extend(chunks)

    def snapshot(self) -> list[str]:
        with self.lock:
            return list(self.chunks)

    def seal(self) -> None:
        self.closed.set()

    @property
    def sealed(self) -> bool:
        return self.closed.is_set()


# Граница предложения в потоке: знак конца фразы перед пробелом или перевод
# строки. Без пробела после точки ждём следующий кусок: это может быть
# сокращение или число.
STREAM_BOUNDARY_RE = re.compile(r"[.!?…]+(?=\s)|\n")
STREAM_CHUNK_LIMIT = 180
# Первый фрагмент режем короче: синтез идёт примерно в темпе речи, и на
# длинном первом фрагменте пользователь ждал бы молчания. Запятая в начале
# предложения даёт естественную паузу.
STREAM_FIRST_LIMIT = 70
STREAM_FIRST_MIN = 15
STREAM_FIRST_BREAK_RE = re.compile(r"[,;:—](?=\s)")


class _SpeechStream:
    """Потоковая озвучка: текст добавляется по мере генерации ответа."""

    def __init__(self, service: TTSService, job: _SpeechJob) -> None:
        self._service = service
        self._job = job
        self._lock = threading.Lock()
        self._pending = ""
        self._added = 0
        self._full = False
        self._closed = False

    def feed(self, text: str) -> None:
        """Добавляет кусок текста. Готовые предложения сразу уходят в синтез."""
        if not text:
            return
        chunks: list[str] = []
        with self._lock:
            if self._closed or self._full or self._job.cancelled.is_set():
                return
            self._pending += text
            for sentence in self._take_sentences():
                chunks.extend(self._accept(sentence))
                if self._full:
                    break
            if self._full:
                self._pending = ""
        if chunks:
            self._job.add(chunks)

    def close(self) -> None:
        """Завершает поток: остаток текста уходит в синтез, задача закрывается."""
        chunks: list[str] = []
        with self._lock:
            if self._closed:
                return
            self._closed = True
            rest = self._pending.strip()
            self._pending = ""
            if rest and not self._full and not self._job.cancelled.is_set():
                chunks.extend(self._accept(rest))
        if chunks:
            self._job.add(chunks)
        self._job.seal()

    # ---------------------------------------------------------------- private
    def _take_sentences(self) -> list[str]:
        ready: list[str] = []
        # Самый первый фрагмент — короткий, чтобы озвучка началась раньше.
        cut = self._first_cut()
        if cut:
            ready.append(self._pending[:cut].strip())
            self._pending = self._pending[cut:].lstrip()
        while True:
            match = STREAM_BOUNDARY_RE.search(self._pending)
            if match is None:
                break
            sentence = self._pending[: match.end()].strip()
            self._pending = self._pending[match.end() :].lstrip()
            if sentence:
                ready.append(sentence)
        # Без знаков конца фразы режем по словам, чтобы не ждать конца абзаца.
        while len(self._pending) > STREAM_CHUNK_LIMIT:
            cut = self._pending.rfind(" ", 0, STREAM_CHUNK_LIMIT)
            if cut < STREAM_CHUNK_LIMIT // 2:
                cut = STREAM_CHUNK_LIMIT
            sentence = self._pending[:cut].strip()
            self._pending = self._pending[cut:].lstrip()
            if sentence:
                ready.append(sentence)
        return ready

    def _first_cut(self) -> int:
        """Где оборвать первый фрагмент, пока ничего ещё не сказано."""
        if self._added:
            return 0
        pending = self._pending
        if len(pending) <= STREAM_FIRST_LIMIT:
            return 0
        # Готовое короткое предложение режем по нему, а не по середине.
        sentence = STREAM_BOUNDARY_RE.search(pending)
        if sentence is not None and sentence.end() <= STREAM_FIRST_LIMIT:
            return 0
        window = pending[:STREAM_FIRST_LIMIT]
        for match in STREAM_FIRST_BREAK_RE.finditer(window):
            if match.end() >= STREAM_FIRST_MIN:
                return match.end()
        # Запятой нет — режем по последнему пробелу в окне.
        space = window.rfind(" ")
        if space >= STREAM_FIRST_MIN:
            return space
        return STREAM_FIRST_LIMIT

    def _accept(self, sentence: str) -> list[str]:
        """Чистит предложение и делит его на куски в пределах лимита озвучки."""
        clean = clean_for_speech(sentence)
        if not clean:
            return []
        chunks: list[str] = []
        for chunk in split_sentences(clean):
            limit = self._service.max_chars
            if limit and self._added + len(chunk) > limit:
                self._full = True
                break
            chunks.append(chunk)
            self._added += len(chunk)
        return chunks


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

        self._play_lock = threading.Lock()
        self._job_lock = threading.Lock()
        self._jobs: queue.Queue[_SpeechJob | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._current_job: _SpeechJob | None = None
        self._pending = 0
        self._generation = 0
        self._active_jobs = 0
        self._playing = 0
        self._current_text = ""

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.api_key) and bool(self.voice)

    @property
    def speaking(self) -> bool:
        with self._job_lock:
            return self._active_jobs > 0 or self._playing > 0

    @property
    def busy(self) -> bool:
        """Есть ли что-то в очереди или в динамиках прямо сейчас."""
        with self._job_lock:
            return self._pending > 0 or self._active_jobs > 0 or self._playing > 0

    @property
    def current_text(self) -> str:
        """Фрагмент, который звучит прямо сейчас: для защиты от самопрослушивания."""
        with self._job_lock:
            return self._current_text

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
        """Ставит текст в очередь озвучки и сразу возвращает управление.

        Синтез и проигрывание идут в отдельном потоке, поэтому главный цикл
        продолжает слушать микрофон и может перебить ассистента.
        """
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

        self._ensure_worker()
        with self._job_lock:
            job = _SpeechJob(chunks=chunks, generation=self._generation)
            job.seal()
            self._pending += 1
        self._jobs.put(job)
        return True

    def stream(self) -> _SpeechStream | None:
        """Начинает потоковую озвучку: текст можно добавлять по мере генерации.

        Возвращает ``None``, если озвучка выключена: тогда диалог работает
        обычным путём, одним запросом.
        """
        if not self.available or not self.speak_answers:
            return None
        self._ensure_worker()
        with self._job_lock:
            job = _SpeechJob(generation=self._generation)
            self._pending += 1
        self._jobs.put(job)
        return _SpeechStream(self, job)

    def stop(self) -> None:
        """Обрывает текущую речь и выбрасывает всё, что ещё стоит в очереди."""
        with self._job_lock:
            self._generation += 1
            if self._current_job is not None:
                self._current_job.cancelled.set()
            dropped = 0
            while True:
                try:
                    job = self._jobs.get_nowait()
                except queue.Empty:
                    break
                if job is None:
                    # Служебный маркер завершения работы воркера.
                    self._jobs.put(None)
                    break
                job.cancelled.set()
                dropped += 1
            self._pending = max(0, self._pending - dropped)
            self._current_text = ""
        # PortAudio здесь не трогаем: остановку заметит сам воркер между
        # блоками и закроет свой поток с того же потока, что открыл.

    def wait(self, timeout: float = 5.0) -> bool:
        """Ждёт, пока очередь озвучки опустеет. Нужен тестам и сценариям выхода."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._job_lock:
                pending = self._pending
            if pending == 0 and not self.speaking:
                return True
            time.sleep(0.01)
        return False

    def shutdown(self) -> None:
        self.stop()
        self._jobs.put(None)
        with self._job_lock:
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2)

    # ------------------------------------------------------------------ worker
    def _ensure_worker(self) -> None:
        with self._job_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="tts-worker",
                daemon=True,
            )
            self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                with self._job_lock:
                    stale = job.cancelled.is_set() or job.generation != self._generation
                    if not stale:
                        self._current_job = job
                if stale:
                    continue
                with self._job_lock:
                    self._active_jobs += 1
                try:
                    self._run_job(job)
                finally:
                    with self._job_lock:
                        self._active_jobs = max(0, self._active_jobs - 1)
            finally:
                with self._job_lock:
                    if self._current_job is job:
                        self._current_job = None
                    self._pending = max(0, self._pending - 1)

    def _run_job(self, job: _SpeechJob) -> None:
        pool = ThreadPoolExecutor(max_workers=self.workers)
        futures: dict[int, Future] = {}
        index = 0
        submitted = 0
        try:
            while True:
                if not self._job_alive(job):
                    return
                chunks = job.snapshot()
                while submitted < len(chunks):
                    futures[submitted] = pool.submit(self._synth_cached, chunks[submitted])
                    submitted += 1

                if index >= len(chunks):
                    # Потоковая задача ещё может получить текст; обычная уже
                    # закрыта и просто закончилась.
                    if job.sealed:
                        # Снимок мог быть сделан до последнего добавления:
                        # после seal проверяем очередь ещё раз.
                        if index >= len(job.snapshot()):
                            return
                    else:
                        time.sleep(0.03)
                    continue

                future = futures.pop(index)
                # Синтез идёт по сети: проверяем отмену, не дожидаясь
                # полного ответа ProxyAPI.
                while not future.done():
                    if not self._job_alive(job):
                        return
                    time.sleep(0.05)
                try:
                    data = future.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning("TTS: фрагмент не синтезирован: %s", exc)
                    data = None
                if data is not None:
                    with self._job_lock:
                        self._current_text = chunks[index]
                    if not self._play(data, job):
                        return
                index += 1
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            with self._job_lock:
                self._current_text = ""

    def _job_alive(self, job: _SpeechJob) -> bool:
        with self._job_lock:
            return not job.cancelled.is_set() and job.generation == self._generation

    def _play_alive(self, job: _SpeechJob | None, generation: int) -> bool:
        """Можно ли продолжать фрагмент: отмена джобы или stop() для фразы."""
        if job is not None:
            return self._job_alive(job)
        with self._job_lock:
            return generation == self._generation

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
    def _play(self, data: np.ndarray, job: _SpeechJob | None = None) -> bool:
        """Проигрывает фрагмент, закрывая поток в том же потоке, что открыл.

        ``sd.stop()`` из главного цикла, пока воркер висит в
        ``sd.play(blocking=True)``, закрывает один и тот же поток PortAudio
        из двух потоков — процесс падает с 0xc0000005 в ntdll. Поэтому
        звук пишется кусками: ``stop()`` только поднимает флаг, а поток
        открывает и закрывает сам писатель.
        """
        if job is not None and not self._job_alive(job):
            return False
        if self.volume != 1.0:
            data = np.clip(data * self.volume, -1.0, 1.0)
        try:
            import sounddevice as sd
        except Exception as exc:  # noqa: BLE001
            log.warning("TTS: звук недоступен: %s", exc)
            return False
        with self._play_lock:
            # stop() мог сработать, пока мы ждали предыдущий фрагмент.
            generation = self._generation
            if not self._play_alive(job, generation):
                return False
            with self._job_lock:
                self._playing += 1
            stream = None
            try:
                stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                )
                stream.start()
                step = max(1, int(SAMPLE_RATE * PLAY_CHUNK_SEC))
                for start in range(0, len(data), step):
                    if not self._play_alive(job, generation):
                        # abort выбрасывает уже поставленные в буфер кадры,
                        # поэтому речь замолкает сразу, а не через блок.
                        stream.abort()
                        return False
                    stream.write(data[start : start + step])
                if not self._play_alive(job, generation):
                    stream.abort()
                    return False
                stream.stop()
                return True
            except Exception as exc:  # noqa: BLE001
                log.warning("TTS: не удалось воспроизвести: %s", exc)
                return False
            finally:
                if stream is not None:
                    try:
                        stream.close(True)
                    except Exception:  # noqa: BLE001
                        pass
                with self._job_lock:
                    self._playing = max(0, self._playing - 1)
