# -*- coding: utf-8 -*-
"""Подбор wake_confusables для Vosk на записанном датасете.

Проверяем recall на настоящих «хей луна» и ложные срабатывания на H.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import vosk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import VOSK_DIR  # noqa: E402
from app.text import fuzzy_contains, normalize  # noqa: E402

AUDIO = Path.home() / "Documents" / "VoiceDataset" / "audio"
ALIASES = ["хей луна", "эй луна"]

POSITIVES = (
    [f"{i:03d}_A_wake_{i:02d}.wav" for i in range(1, 13)]
    + [f"{i:03d}_A2_alias_{i - 12:02d}.wav" for i in range(13, 17)]
    + [f"{i:03d}_B_wc{i - 16:02d}.wav" for i in range(17, 27)]
)
NEGATIVES = [f"{i:03d}_H_neg_{i - 105:02d}.wav" for i in range(106, 118)]

SETS: dict[str, list[str]] = {
    "пусто": [],
    "слова": [
        "луна", "луну", "лужа", "люда", "юля", "уна", "дума", "руна",
        "лупа", "нуна", "лун", "луна парк", "хей люда", "хей юля",
        "эй лужа", "лей луну", "хей ну на",
    ],
    "фразы": [
        "хей люда", "хей юля", "эй лужа", "лей луну", "хей ну на",
        "хей луна парк", "эй луна парк",
    ],
    "оба": [
        "луна", "луну", "лужа", "люда", "юля", "уна", "дума", "руна",
        "лупа", "нуна", "лун", "хей люда", "хей юля", "эй лужа",
        "лей луну", "хей ну на", "хей луна парк", "эй луна парк",
    ],
}


def read_pcm(name: str) -> bytes:
    with wave.open(str(AUDIO / name), "rb") as f:
        return f.readframes(f.getnframes())


def allowed_errors(alias: str) -> int:
    if len(alias.split()) > 1:
        return 1 if len(alias) >= 10 else 0
    return 0 if len(alias) <= 6 else 1


def main() -> int:
    vosk.SetLogLevel(-1)
    model = vosk.Model(str(VOSK_DIR))
    prepared = [normalize(a) for a in ALIASES]
    limits = [allowed_errors(a) for a in prepared]

    cache = {name: read_pcm(name) for name in POSITIVES + NEGATIVES}

    for label, extra in SETS.items():
        vocabulary = list(dict.fromkeys([*ALIASES, *extra]))
        grammar = json.dumps([*vocabulary, "[unk]"], ensure_ascii=False)
        rec = vosk.KaldiRecognizer(model, 16000, grammar)
        rec.SetWords(True)

        def hit(name: str) -> bool:
            rec.Reset()
            pcm = cache[name]
            step = 4000
            for i in range(0, len(pcm) - step + 1, step):
                if rec.AcceptWaveform(pcm[i:i + step]):
                    raw = json.loads(rec.Result()).get("text", "")
                else:
                    raw = json.loads(rec.PartialResult()).get("partial", "")
                text = normalize(raw)
                if text and any(
                    fuzzy_contains(text, alias, max_distance=limit)
                    for alias, limit in zip(prepared, limits)
                ):
                    return True
            return False

        pos = {name: hit(name) for name in POSITIVES}
        neg = {name: hit(name) for name in NEGATIVES}
        missed = [n for n, v in pos.items() if not v]
        false = [n for n, v in neg.items() if v]
        print(f"\n=== {label} (похожих слов: {len(extra)}) ===")
        print(f"  recall {len(pos) - len(missed)}/{len(pos)}  "
              f"ложные {len(false)}/{len(neg)}")
        if missed:
            print("  пропуски:", ", ".join(missed))
        if false:
            print("  ложные: ", ", ".join(false))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
