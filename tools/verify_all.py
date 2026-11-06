"""Итоговая проверка после правок: wake + команды на всём датасете.

Wake — настоящий WakeWordDetector из config.json в непрерывном потоке.
Команды — GigaAM + strip_wake_word + ActionRouter (браузер и громкость
заглушены, ничего не выполняется).
"""

from __future__ import annotations

import csv
import json
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline import Router, expected_intent, intent_ok, word_error  # noqa: E402

from app.config import VOSK_DIR, load_config  # noqa: E402
from app.stt import GigaAMTranscriber  # noqa: E402
from app.text import canonicalize, normalize, strip_wake_word  # noqa: E402
from app.wake import WakeWordDetector  # noqa: E402

AUDIO = Path.home() / "Documents" / "VoiceDataset" / "audio"
SIL_1S = b"\x00\x00" * 16000
STEP = 8000
RESULTS = Path(__file__).resolve().parent / "verify_results.json"

GROUPS = {
    "A обычные": [f"{i:03d}_A_wake_{i:02d}.wav" for i in range(1, 9)],
    "A тихие": [f"{i:03d}_A_wake_{i:02d}.wav" for i in (9, 10)],
    "A 2 м": [f"{i:03d}_A_wake_{i:02d}.wav" for i in (11, 12)],
    "A2 алиас": [f"{i:03d}_A2_alias_{i - 12:02d}.wav" for i in range(13, 17)],
    "B wake+cmd": [f"{i:03d}_B_wc{i - 16:02d}.wav" for i in range(17, 27)],
    "H негативы": [f"{i:03d}_H_neg_{i - 105:02d}.wav" for i in range(106, 118)],
    "I 2 м": [f"{i:03d}_I_wake_2m_{i - 117:02d}.wav" for i in range(118, 124)],
    "L негативы 2": [f"{i:03d}_L_neg_{i - 147:02d}.wav" for i in range(148, 154)],
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
    cfg = load_config()
    aliases = cfg["wake_aliases"]
    labels = list(csv.DictReader(
        open(Path.home() / "Documents" / "VoiceDataset" / "labels.csv",
             encoding="utf-8-sig", newline=""),
        delimiter=";",
    ))
    labels = [r for r in labels if (AUDIO / r["file"]).exists()]
    cache = {r["file"]: read_pcm(r["file"]) for r in labels}
    for block, label in (("M", "M условия"), ("N", "N негативы")):
        names = [r["file"] for r in labels if r["block"] == block]
        if names:
            GROUPS[label] = names

    # ---------------------------------------------------------------- wake
    det = WakeWordDetector(
        VOSK_DIR,
        aliases,
        sample_rate=int(cfg["audio"]["sample_rate"]),
        confusables=cfg.get("wake_confusables") or [],
    )
    feed(det, SIL_1S * 2)
    wake: dict[str, list[str]] = {}
    for group, names in GROUPS.items():
        names = [name for name in names if name in cache]
        if not names:
            continue
        hits: list[str] = []
        for name in names:
            if feed(det, cache[name]) or feed(det, SIL_1S):
                hits.append(name)
                det.reset()
                feed(det, SIL_1S * 2)
        wake[group] = hits

    # ------------------------------------------------------------- команды
    stt = GigaAMTranscriber(
        num_threads=int((cfg.get("stt") or {}).get("num_threads") or 2),
        tail_padding_sec=float((cfg.get("stt") or {}).get("tail_padding_sec", 0.4)),
    )
    stt.load()
    router = Router(cfg)

    blocks: dict[str, list[dict]] = defaultdict(list)
    decode_times: list[float] = []
    for row in labels:
        if row["kind"] not in ("command", "wake_command"):
            continue
        t0 = time.perf_counter()
        text = stt.transcribe(cache[row["file"]])
        decode_times.append(time.perf_counter() - t0)
        stripped = canonicalize(strip_wake_word(text, aliases))
        want_text = strip_wake_word(row["text"], aliases) if row["kind"] == "wake_command" else row["text"]
        want_text = canonicalize(want_text)
        observed = router.run(stripped)
        blocks[row["block"]].append({
            "file": row["file"],
            "ok": intent_ok(expected_intent(row["action"]), observed),
            "exact": normalize(want_text) == normalize(stripped),
            "wer": word_error(want_text, stripped),
            "text": text,
            "stripped": stripped,
        })

    print("WAKE (настоящий Vosk + confusables)")
    total_hits = 0
    total = 0
    for group, names in GROUPS.items():
        hits = wake.get(group)
        if hits is None:
            continue
        names = [name for name in names if name in cache]
        print(f"  {group:<12} {len(hits)}/{len(names)}")
        if group.startswith(("H", "L", "N")):
            if hits:
                print("    ложные:", ", ".join(hits))
        else:
            total_hits += len(hits)
            total += len(names)
    print(f"  итого настоящих: {total_hits}/{total}")

    print()
    print("КОМАНДЫ")
    header = f"{'блок':<5}{'N':>4}{'действие':>10}{'текст':>8}{'WER':>7}"
    print(header)
    print("-" * len(header))
    for block in sorted(blocks):
        items = blocks.get(block, [])
        if not items:
            continue
        print(f"{block:<5}{len(items):>4}"
              f"{sum(i['ok'] for i in items) / len(items) * 100:>9.0f}%"
              f"{sum(i['exact'] for i in items) / len(items) * 100:>7.0f}%"
              f"{np.mean([i['wer'] for i in items]):>7.2f}")
    all_items = [i for items in blocks.values() for i in items]
    print(f"{'все':<5}{len(all_items):>4}"
          f"{sum(i['ok'] for i in all_items) / len(all_items) * 100:>9.0f}%"
          f"{sum(i['exact'] for i in all_items) / len(all_items) * 100:>7.0f}%"
          f"{np.mean([i['wer'] for i in all_items]):>7.2f}")
    print(f"  среднее время распознавания: {np.mean(decode_times):.2f} с")

    bad = [i for i in all_items if not i["ok"]]
    if bad:
        print()
        print(f"Ошибки действий: {len(bad)}")
        for i in bad:
            print(f"  {i['file']:<18} {i['text']!r} -> {i['stripped']!r}")

    RESULTS.write_text(json.dumps({
        "wake": {g: hits for g, hits in wake.items()},
        "commands": {b: items for b, items in blocks.items()},
        "decode_sec": round(float(np.mean(decode_times)), 3),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСырые результаты: {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
