# -*- coding: utf-8 -*-
"""Почему конкретный файл не срабатывает: чанки и текст Vosk."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.config import VOSK_DIR, load_config  # noqa: E402
from app.text import fuzzy_contains, normalize  # noqa: E402
from app.wake import WakeWordDetector, allowed_errors  # noqa: E402

AUDIO = Path.home() / "Documents" / "VoiceDataset" / "audio"


def main() -> int:
    cfg = load_config()
    names = sys.argv[1:] or ["122_I_wake_2m_05.wav"]
    det = WakeWordDetector(
        VOSK_DIR, cfg["wake_aliases"], confusables=cfg.get("wake_confusables")
    )
    aliases = det.aliases
    for name in names:
        with wave.open(str(AUDIO / name), "rb") as f:
            pcm = f.readframes(f.getnframes())
        print(f"\n{name}: {len(pcm) / 2 / 16000:.2f} с")
        det.reset()
        for index in range(0, len(pcm) - 8000 + 1, 8000):
            chunk = pcm[index:index + 8000]
            final = det.recognizer.AcceptWaveform(chunk)
            raw = (
                json.loads(det.recognizer.Result()).get("text", "")
                if final
                else json.loads(det.recognizer.PartialResult()).get("partial", "")
            )
            text = normalize(raw)
            hits = [
                alias
                for alias in aliases
                if fuzzy_contains(text, alias, max_distance=allowed_errors(alias))
            ]
            print(f"  {index / 32000:4.1f} с  {raw!r:35} {'HIT ' + ','.join(hits) if hits else ''}")
        det.recognizer.AcceptWaveform(b"\x00\x00" * 16000)
        raw = json.loads(det.recognizer.FinalResult()).get("text", "")
        text = normalize(raw)
        hits = [
            alias
            for alias in aliases
            if fuzzy_contains(text, alias, max_distance=allowed_errors(alias))
        ]
        print(f"  final {raw!r} {'HIT ' + ','.join(hits) if hits else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
