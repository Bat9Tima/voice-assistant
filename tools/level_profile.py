# -*- coding: utf-8 -*-
"""Профиль громкости по 100 мс: видно, обрезана ли фраза при записи."""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

AUDIO = Path.home() / "Documents" / "VoiceDataset" / "audio"


def main() -> int:
    names = sys.argv[1:] or ["118_I_wake_2m_01.wav", "119_I_wake_2m_02.wav"]
    for name in names:
        with wave.open(str(AUDIO / name), "rb") as f:
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        print(f"\n{name}: {data.size / 16000:.2f} с")
        step = 1600
        bars = []
        for i in range(0, data.size - step + 1, step):
            chunk = data[i:i + step].astype(np.float32) / 32768.0
            level = 20 * np.log10(max(float(np.sqrt(np.mean(chunk ** 2))), 1e-9))
            bars.append(f"{level:6.1f}")
        for row in range(0, len(bars), 10):
            print(f"  {row * 0.1:4.1f} с  " + " ".join(bars[row:row + 10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
