from __future__ import annotations

import logging
import queue
import time

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


def rms(pcm: bytes) -> float:
    """Среднеквадратичная громкость фрагмента PCM int16 (0..1)."""
    if not pcm:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    normalized = samples.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(normalized * normalized)))


def list_input_devices() -> list[dict]:
    devices = []
    try:
        default_index = sd.default.device[0]
    except Exception:
        default_index = -1
    for index, info in enumerate(sd.query_devices()):
        if int(info.get("max_input_channels", 0)) <= 0:
            continue
        devices.append(
            {
                "index": index,
                "name": info.get("name", ""),
                "channels": int(info.get("max_input_channels", 0)),
                "default_samplerate": int(info.get("default_samplerate", 0)),
                "default": index == default_index,
            }
        )
    return devices


class Microphone:
    """Потоковый захват микрофона с приведением к 16 кГц mono int16."""

    def __init__(
        self,
        samplerate: int = 16000,
        block_size: int = 4000,
        device: int | str | None = None,
    ) -> None:
        self.target_rate = int(samplerate)
        self.block_size = int(block_size)
        self.device = device
        self.capture_rate = self.target_rate
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._stream: sd.RawInputStream | None = None

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            self._open(self.target_rate)
        except Exception as exc:
            log.warning("Не удалось открыть микрофон на %s Гц: %s", self.target_rate, exc)
            info = sd.query_devices(self.device, "input")
            fallback = int(round(float(info["default_samplerate"])))
            if fallback == self.target_rate:
                raise
            log.info("Пробую частоту устройства %s Гц", fallback)
            self._open(fallback)

    def _open(self, rate: int) -> None:
        self.capture_rate = rate
        self._stream = sd.RawInputStream(
            samplerate=rate,
            blocksize=self.block_size,
            device=self.device,
            dtype="int16",
            channels=1,
            callback=self._callback,
        )
        self._stream.start()
        log.info("Микрофон открыт: %s Гц, устройство=%s", rate, self.device)

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("Статус аудиопотока: %s", status)
        try:
            self._queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def _resample(self, pcm: bytes) -> bytes:
        if self.capture_rate == self.target_rate:
            return pcm
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return pcm
        ratio = self.target_rate / self.capture_rate
        new_length = max(1, int(round(samples.size * ratio)))
        positions = np.linspace(0, samples.size - 1, new_length, dtype=np.float64)
        resampled = np.interp(positions, np.arange(samples.size), samples)
        return resampled.astype(np.int16).tobytes()

    def read(self, timeout: float = 0.5) -> bytes | None:
        if self._stream is None:
            return None
        try:
            chunk = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        return self._resample(chunk)

    def flush(self) -> None:
        """Выбрасывает накопившийся аудиобуфер (после долгой обработки команды)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def measure_noise(self, duration: float = 1.0) -> float:
        """Оценивает уровень фонового шума по медиане RMS."""
        values: list[float] = []
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            chunk = self.read(timeout=0.3)
            if chunk:
                values.append(rms(chunk))
        if not values:
            return 0.0
        return float(np.median(values))

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
                log.info("Микрофон остановлен")
