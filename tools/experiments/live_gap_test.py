"""Проверка живого пути без очистки микрофона: сигнал попадает в запись."""

from __future__ import annotations

import csv
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline import Router, expected_intent, intent_ok  # noqa: E402

from app.config import load_config  # noqa: E402
from app.stt import GigaAMTranscriber  # noqa: E402
from app.text import strip_wake_word  # noqa: E402

AUDIO = Path.home() / "Documents" / "VoiceDataset" / "audio"
ALIASES = ["хей луна", "эй луна"]
CUTS = (0.9, 1.1, 1.3, 1.5, 1.7)
RATE = 16000


def make_beep(level_db: float = -20.0, ms: int = 80) -> bytes:
    t = np.arange(int(RATE * ms / 1000)) / RATE
    x = np.sin(2 * np.pi * 880 * t) * (10 ** (level_db / 20)) * 32767
    return x.astype(np.int16).tobytes()


def main() -> int:
    cfg = load_config()
    stt = GigaAMTranscriber(num_threads=2, tail_padding_sec=0.4)
    stt.load()
    labels = list(csv.DictReader(
        open(Path.home() / "Documents" / "VoiceDataset" / "labels.csv",
             encoding="utf-8-sig", newline=""),
        delimiter=";",
    ))
    rows = [r for r in labels if r["block"] == "B"]
    router = Router(cfg)
    beep = make_beep()

    for cut in CUTS:
        for label, insert in (("с сигналом", beep), ("без сигнала", b"")):
            ok = 0
            bad: list[str] = []
            for row in rows:
                with wave.open(str(AUDIO / row["file"]), "rb") as f:
                    pcm = f.readframes(f.getnframes())
                pos = int(cut * RATE) * 2
                stream = pcm[:pos] + insert + pcm[pos:]
                text = stt.transcribe(stream)
                stripped = strip_wake_word(text, ALIASES)
                observed = router.run(stripped)
                want = expected_intent(row["action"])
                if intent_ok(want, observed):
                    ok += 1
                else:
                    bad.append(f"{row['file']}: {text!r} -> {stripped!r}")
            print(f"{cut:.1f} с, {label}: {ok}/{len(rows)}")
            for line in bad[:2]:
                print("   ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
