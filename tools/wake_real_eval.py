# -*- coding: utf-8 -*-
"""Сравнение вариантов алиасов для настоящего WakeWordDetector."""

from __future__ import annotations

import logging
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import VOSK_DIR, load_config  # noqa: E402
from app.wake import WakeWordDetector  # noqa: E402

AUDIO = Path.home() / "Documents" / "VoiceDataset" / "audio"
SIL_1S = b"\x00\x00" * 16000
STEP = 8000

GROUPS = {
    "A обычные": [f"{i:03d}_A_wake_{i:02d}.wav" for i in range(1, 9)],
    "A тихие": [f"{i:03d}_A_wake_{i:02d}.wav" for i in (9, 10)],
    "A 2 м": [f"{i:03d}_A_wake_{i:02d}.wav" for i in (11, 12)],
    "A2 алиас": [f"{i:03d}_A2_alias_{i - 12:02d}.wav" for i in range(13, 17)],
    "B wake+cmd": [f"{i:03d}_B_wc{i - 16:02d}.wav" for i in range(17, 27)],
    "H негативы": [f"{i:03d}_H_neg_{i - 105:02d}.wav" for i in range(106, 118)],
}

VARIANTS = {
    "2 алиаса": ["хей луна", "эй луна"],
    "3 алиаса": ["хей луна", "хэй луна", "эй луна"],
}


def read_pcm(name: str) -> bytes:
    with wave.open(str(AUDIO / name), "rb") as f:
        return f.readframes(f.getnframes())


def feed(det: WakeWordDetector, pcm: bytes) -> bool:
    hit = False
    for i in range(0, len(pcm) - STEP + 1, STEP):
        if det.feed(pcm[i:i + STEP]):
            hit = True
    return hit


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config()
    cache = {n: read_pcm(n) for names in GROUPS.values() for n in names}
    for label, aliases in VARIANTS.items():
        det = WakeWordDetector(
            VOSK_DIR,
            aliases,
            sample_rate=int(cfg["audio"]["sample_rate"]),
            confusables=cfg.get("wake_confusables") or [],
        )
        feed(det, SIL_1S * 2)
        print(f"\n=== {label} ===")
        for group, names in GROUPS.items():
            hits: list[str] = []
            for name in names:
                if feed(det, cache[name]) or feed(det, SIL_1S):
                    hits.append(name)
                    det.reset()
                    feed(det, SIL_1S * 2)
            print(f"{group:<12} {len(hits)}/{len(names)}"
                  + (f"  ложные: {', '.join(hits)}" if group == "H негативы" and hits else "")
                  + (f"  пропуски: {', '.join(n for n in names if n not in hits)}"
                     if len(hits) < len(names) and group != "H негативы" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
