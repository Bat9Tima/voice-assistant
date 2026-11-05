# -*- coding: utf-8 -*-
"""Проверка: портит ли beep 880 Гц слово «саундклауда» в живой команде."""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.stt import GigaAMTranscriber  # noqa: E402

AUDIO = Path.home() / "Documents" / "VoiceDataset" / "audio"
NAME = "020_B_wc04.wav"


def load(name: str) -> np.ndarray:
    with wave.open(str(AUDIO / name), "rb") as f:
        return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)


def beep(seconds: float = 0.08, freq: int = 880, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(16000 * seconds)) / 16000
    return (np.sin(2 * np.pi * freq * t) * amp * 32768).astype(np.float32)


def mix(samples: np.ndarray, at: float, gain: float = 1.0) -> np.ndarray:
    out = samples.copy()
    tone = beep() * gain
    start = int(at * 16000)
    end = min(out.size, start + tone.size)
    out[start:end] += tone[: end - start]
    return np.clip(out, -32768, 32767)


def insert(samples: np.ndarray, at: float) -> np.ndarray:
    start = int(at * 16000)
    gap = np.zeros(int(0.12 * 16000), dtype=np.float32)
    return np.concatenate([samples[:start], beep(), gap, samples[start:]])


def main() -> int:
    samples = load(NAME)
    print(f"{NAME}: {samples.size / 16000:.2f} с")
    stt = GigaAMTranscriber(num_threads=2, tail_padding_sec=0.4)
    stt.load()

    def run(label: str, data: np.ndarray) -> None:
        pcm = np.clip(data, -32768, 32767).astype(np.int16).tobytes()
        text = stt.transcribe(pcm)
        print(f"  {label:<28} {text!r}")

    run("чисто", samples)
    for at in (0.9, 1.1, 1.3, 1.5, 1.7):
        run(f"beep врезан {at:.1f} с", insert(samples, at))
    for at in (1.1, 1.3, 1.5):
        run(f"beep подмешан {at:.1f} с", mix(samples, at))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
