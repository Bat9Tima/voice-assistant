from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .config import MODELS_DIR

log = logging.getLogger(__name__)

VAD_FILENAME = "silero_vad.onnx"


def _installed_openwakeword_model() -> Path | None:
    """Silero VAD из пакета openwakeword — запасной источник модели."""
    try:
        import openwakeword
    except Exception:  # noqa: BLE001
        return None
    path = Path(openwakeword.__file__).resolve().parent / "resources" / "models" / VAD_FILENAME
    return path if path.is_file() else None


def _ensure_model(target: Path) -> Path | None:
    """Возвращает модель в `target`, при необходимости копируя её из пакета."""
    if target.is_file():
        return target
    source = _installed_openwakeword_model()
    if source is None:
        log.warning("Silero VAD не найден: %s", target)
        return None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        log.info("Скопировал Silero VAD: %s -> %s", source, target)
    except Exception as exc:  # noqa: BLE001
        log.warning("Не удалось скопировать Silero VAD: %s", exc)
        return None
    return target


def ensure_model(model: str | Path | None = None) -> Path | None:
    """Гарантирует наличие модели VAD и возвращает путь к ней."""
    if model is None or not str(model).strip():
        target = MODELS_DIR / VAD_FILENAME
    else:
        target = Path(os.path.expandvars(os.path.expanduser(str(model))))
        if not target.is_absolute():
            target = MODELS_DIR / target
    if target.is_file():
        return target
    if target.name == VAD_FILENAME:
        return _ensure_model(target)
    log.warning("Модель VAD не найдена: %s", target)
    return None


def _model_kind(path: Path) -> str | None:
    """Проверяет интерфейс ONNX-модели до передачи её в sherpa-onnx.

    sherpa-onnx при неподдерживаемом файле завершает процесс, поэтому
    несовместимую модель лучше отсеять заранее и остаться на пороге громкости.
    """
    try:
        import onnxruntime as ort
    except Exception:  # noqa: BLE001
        return "unknown"
    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = [item.name for item in session.get_inputs()]
        outputs = [item.name for item in session.get_outputs()]
    except Exception as exc:  # noqa: BLE001
        log.warning("Не удалось прочитать модель VAD: %s", exc)
        return None
    if inputs == ["input", "sr", "h", "c"] and outputs == ["output", "hn", "cn"]:
        return "v4"
    if inputs == ["input", "state", "sr"] and outputs == ["output", "stateN"]:
        return "v5"
    if inputs == ["x", "h", "c"] and outputs == ["prob", "new_h", "new_c"]:
        return "v4-k2"
    log.warning("Неподдерживаемый формат модели VAD: %s", path)
    return None


class SileroVad:
    """Потоковый Silero VAD: принимает PCM кусками произвольной длины."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._window = int(model.window_size())
        self._pending = np.empty(0, dtype=np.float32)

    @property
    def window_size(self) -> int:
        return self._window

    @classmethod
    def create(
        cls,
        model: str | Path | None = None,
        threshold: float = 0.5,
    ) -> SileroVad | None:
        path = ensure_model(model)
        if path is None:
            return None
        if _model_kind(path) is None:
            return None
        try:
            import sherpa_onnx
        except Exception as exc:  # noqa: BLE001
            log.warning("sherpa-onnx недоступен, VAD выключен: %s", exc)
            return None

        try:
            config = sherpa_onnx.VadModelConfig()
            config.silero_vad = sherpa_onnx.SileroVadModelConfig(
                model=str(path),
                threshold=max(0.01, min(0.99, float(threshold))),
                # Начало и конец речи считаем сами по решениям VAD: так
                # `silence_ms` и `min_speech_ms` остаются общими настройками
                # для VAD и запасного режима по громкости.
                min_silence_duration=0.0,
                min_speech_duration=0.0,
                window_size=512,
                max_speech_duration=20.0,
            )
            config.sample_rate = 16000
            config.num_threads = 1
            config.debug = False
            raw = sherpa_onnx.VadModel.create(config)
        except Exception as exc:  # noqa: BLE001
            log.warning("Silero VAD не загрузился, остаётся порог громкости: %s", exc)
            return None
        if raw is None:
            log.warning("Silero VAD не загрузился, остаётся порог громкости")
            return None
        log.info("Silero VAD включён: %s", path)
        return cls(raw)

    def feed(self, pcm: bytes) -> bool:
        """Возвращает True, если в куске PCM VAD услышал речь."""
        if not pcm:
            return False
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if self._pending.size:
            samples = np.concatenate((self._pending, samples))
        count = samples.size // self._window
        speech = False
        for index in range(count):
            start = index * self._window
            window = samples[start : start + self._window]
            if self._model.is_speech(window):
                speech = True
        self._pending = samples[count * self._window :].copy()
        return speech

    def reset(self) -> None:
        self._model.reset()
        self._pending = np.empty(0, dtype=np.float32)


class EndpointTracker:
    """Определяет начало и конец команды по VAD и порогу громкости.

    Пока VAD молчит, решение принимает порог: так не теряются шёпот и очень
    тихие слова. Как только VAD услышал речь, дальше он главный — громкая
    музыка больше не удерживает запись.
    """

    def __init__(self, silence_ms: float = 800, min_speech_ms: float = 250) -> None:
        self.silence_limit = float(silence_ms)
        self.min_speech = float(min_speech_ms)
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self.vad_ms = 0.0
        self.started = False
        self.vad_active = False

    def update(self, duration_ms: float, vad_speech: bool, loud: bool) -> None:
        if vad_speech:
            self.vad_active = True
            self.vad_ms += duration_ms
        speech = vad_speech if self.vad_active else loud
        if speech:
            self.speech_ms += duration_ms
            self.silence_ms = 0.0
            self.started = True
        elif self.started:
            self.silence_ms += duration_ms

    @property
    def has_speech(self) -> bool:
        return self.speech_ms >= self.min_speech
