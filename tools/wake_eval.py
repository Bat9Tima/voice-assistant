#!/usr/bin/env python
"""Оценка настоящего Vosk wake word на записанном датасете.

Считает FRR (пропуски настоящих «хей луна») и FAR (ложные срабатывания на
негативах), с разбивкой по блокам, условиям и расстоянию. Ничего не меняет,
только читает labels.csv и audio/.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import wave
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import VOSK_DIR, load_config  # noqa: E402
from app.wake import WakeWordDetector  # noqa: E402

DEFAULT_DATASET = Path.home() / "Documents" / "VoiceDataset"
STEP = 8000  # 0.5 с при 16 кГц
LEAD_SILENCE = b"\x00\x00" * 16000
TAIL_SILENCE = b"\x00\x00" * 16000

CONDITION_ORDER = {
    "silence": 0,
    "fan": 1,
    "music": 2,
    "tv": 3,
    "street": 4,
    "unknown": 99,
}


def read_pcm(path: Path) -> bytes | None:
    with wave.open(str(path), "rb") as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, 16000):
            print(f"  ! пропуск {path.name}: не mono/16 бит/16 кГц")
            return None
        return f.readframes(f.getnframes())


def feed(det: WakeWordDetector, pcm: bytes) -> bool:
    """Прогоняет аудио непрерывным потоком шагами по 0.5 с.

    Хвостовой кусок короче шага не подаётся: датасетные файлы всегда
    длиннее 0.5 с, а недостающие байты компенсирует TAIL_SILENCE.
    """
    hit = False
    for i in range(0, len(pcm) - STEP + 1, STEP):
        if det.feed(pcm[i:i + STEP]):
            hit = True
    return hit


def detect(det: WakeWordDetector, pcm: bytes) -> bool:
    det.reset()
    hit = feed(det, LEAD_SILENCE)
    hit = feed(det, pcm) or hit
    hit = feed(det, TAIL_SILENCE) or hit
    det.reset()
    return hit


def distance_label(value: object) -> str:
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return "? см"
    return f"{number:g} см"


def distance_key(label: str) -> float:
    try:
        return float(label.split()[0])
    except (ValueError, IndexError):
        return 9999.0


def rate(hits: int, total: int) -> str:
    if total == 0:
        return "0/0"
    return f"{hits}/{total} ({hits / total * 100:.0f}%)"


def group_report(
    title: str,
    items: list[dict],
    key_fn,
    sort_key=None,
) -> None:
    if not items:
        return
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[str(key_fn(item))].append(item)
    print(f"  {title}:")
    keys = sorted(groups, key=sort_key) if sort_key else sorted(groups)
    for key in keys:
        rows = groups[key]
        hits = sum(1 for item in rows if item["hit"])
        print(f"    {key:<16} {rate(hits, len(rows))}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценка Vosk wake word")
    parser.add_argument("--labels", type=Path,
                        default=DEFAULT_DATASET / "labels.csv")
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    labels_path = args.labels.expanduser().resolve()
    audio = (args.audio or labels_path.parent / "audio").expanduser().resolve()
    return labels_path, audio


def evaluate(
    rows: list[dict],
    audio: Path,
    det: WakeWordDetector,
    labels_path: Path | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Прогоняет разметку через детектор, возвращает (позитивы, негативы, json)."""
    feed(det, LEAD_SILENCE * 2)

    positives: list[dict] = []
    negatives: list[dict] = []
    for row in rows:
        kind = str(row.get("kind") or "")
        if kind in {"wake", "wake_alias", "wake_command"}:
            target = positives
        elif kind == "negative":
            target = negatives
        else:
            continue
        pcm = read_pcm(audio / row["file"])
        if pcm is None:
            continue
        target.append({
            "file": row["file"],
            "block": row.get("block") or "?",
            "condition": row.get("condition") or "unknown",
            "distance": distance_label(row.get("distance_cm")),
            "level": row.get("level") or "?",
            "hit": detect(det, pcm),
        })

    misses = [item for item in positives if not item["hit"]]
    false = [item for item in negatives if item["hit"]]
    payload = {
        "labels": str(labels_path) if labels_path
        else str(audio.parent / "labels.csv"),
        "positives": len(positives),
        "negatives": len(negatives),
        "misses": [item["file"] for item in misses],
        "false_accepts": [item["file"] for item in false],
        "frr": len(misses) / len(positives) if positives else None,
        "far": len(false) / len(negatives) if negatives else None,
    }
    return positives, negatives, payload


def print_report(
    labels_path: Path,
    positives: list[dict],
    negatives: list[dict],
    payload: dict,
) -> None:
    misses = [item for item in positives if not item["hit"]]
    false = [item for item in negatives if item["hit"]]

    print("Vosk wake evaluation")
    print(f"  датасет: {labels_path.parent}")
    print(f"  позитивы: {len(positives)} · негативы: {len(negatives)}")
    print()
    print(f"FRR: {rate(len(misses), len(positives))}")
    group_report("по блокам", positives, lambda i: i["block"])
    group_report("по условиям", positives, lambda i: i["condition"],
                 lambda k: (CONDITION_ORDER.get(k, 50), k))
    group_report("по дистанции", positives, lambda i: i["distance"],
                 distance_key)
    if misses:
        print("  пропуски:")
        for item in misses:
            print(f"    {item['file']}: {item['block']}, "
                  f"{item['condition']}, {item['distance']}, {item['level']}")

    print()
    print(f"FAR: {rate(len(false), len(negatives))}")
    group_report("по блокам", negatives, lambda i: i["block"])
    group_report("по условиям", negatives, lambda i: i["condition"],
                 lambda k: (CONDITION_ORDER.get(k, 50), k))
    if false:
        print("  ложные:")
        for item in false:
            print(f"    {item['file']}: {item['block']}, "
                  f"{item['condition']}, {item['distance']}")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labels_path, audio = resolve_paths(args)
    if not labels_path.exists():
        print(f"Нет файла разметки: {labels_path}")
        return 2
    if not audio.exists():
        print(f"Нет папки с аудио: {audio}")
        return 2

    with open(labels_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    rows = [r for r in rows if (audio / r["file"]).exists()]
    if not rows:
        print("В разметке нет файлов, которые есть в audio/.")
        return 2

    logging.basicConfig(level=logging.WARNING)
    cfg = load_config()
    det = WakeWordDetector(
        VOSK_DIR,
        cfg["wake_aliases"],
        sample_rate=int(cfg["audio"]["sample_rate"]),
        confusables=cfg.get("wake_confusables") or [],
    )
    positives, negatives, payload = evaluate(
        rows, audio, det, labels_path=labels_path)
    print_report(labels_path, positives, negatives, payload)

    if args.json:
        write_json(args.json, payload)
        print(f"\nJSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
