# -*- coding: utf-8 -*-
"""Диагностика одного файла: что слышит Vosk на разных уровнях громкости."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import vosk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import VOSK_DIR, load_config  # noqa: E402
from app.text import normalize  # noqa: E402

AUDIO = Path.home() / "Documents" / "VoiceDataset" / "audio"


def load(name: str) -> np.ndarray:
    with wave.open(str(AUDIO / name), "rb") as f:
        data = f.readframes(f.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32)


def grammar(cfg: dict) -> str:
    words = [*cfg["wake_aliases"], *(cfg.get("wake_confusables") or [])]
    return json.dumps([*dict.fromkeys(words), "[unk]"], ensure_ascii=False)


def run(samples: np.ndarray, cfg: dict, gain_db: float) -> list[str]:
    boosted = np.clip(samples * (10 ** (gain_db / 20.0)), -32768, 32767)
    pcm = boosted.astype(np.int16).tobytes()
    model = vosk.Model(str(VOSK_DIR))
    rec = vosk.KaldiRecognizer(model, 16000, grammar(cfg))
    rec.SetWords(True)
    seen: list[str] = []
    step = 4000
    for i in range(0, len(pcm) - step + 1, step):
        rec.AcceptWaveform(pcm[i:i + step])
        partial = normalize(json.loads(rec.PartialResult()).get("partial", ""))
        if partial and (not seen or seen[-1] != partial):
            seen.append(partial)
    final = normalize(json.loads(rec.FinalResult()).get("text", ""))
    if final and (not seen or seen[-1] != final):
        seen.append(final)
    return seen


def main() -> int:
    cfg = load_config()
    names = sys.argv[1:] or ["011_A_wake_11.wav", "012_A_wake_12.wav"]
    for name in names:
        samples = load(name)
        level = 20 * np.log10(max(float(np.sqrt((samples / 32768.0) ** 2).mean()), 1e-9))
        print(f"\n{name}  RMS {level:.1f} dBFS")
        for gain in (0, 6, 12, 18):
            seen = run(samples, cfg, gain)
            print(f"  +{gain:>2} dB: {' | '.join(seen[-4:])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
