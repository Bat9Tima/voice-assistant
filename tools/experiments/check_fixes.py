# -*- coding: utf-8 -*-
"""Проверка текстовых правок на уже распознанных текстах baseline.

STT детерминирован, поэтому можно взять сохранённые тексты и прогнать
новый strip_wake_word + ActionRouter без повторного распознавания.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from app.text import normalize, strip_wake_word  # noqa: E402
from baseline import Router, expected_intent, intent_ok, word_error  # noqa: E402

ALIASES = ["хей луна", "хэй луна", "эй луна"]
DATA = json.loads((ROOT / "tools" / "baseline_results.json").read_text(encoding="utf-8"))
LABELS = list(csv.DictReader(
    open(Path.home() / "Documents" / "VoiceDataset" / "labels.csv",
         encoding="utf-8-sig", newline=""),
    delimiter=";",
))
BY = {r["file"]: r for r in LABELS}


def main() -> int:
    cfg = load_config()
    router = Router(cfg)
    blocks: dict[str, list[dict]] = defaultdict(list)
    failures: list[str] = []

    for row in DATA["results"]:
        label = BY[row["file"]]
        if label["kind"] not in ("command", "wake_command"):
            continue
        text = strip_wake_word(row["text"], ALIASES)
        want_text = strip_wake_word(label["text"], ALIASES) if label["kind"] in {
            "wake", "wake_alias", "wake_command"
        } else label["text"]
        observed = router.run(text)
        want = expected_intent(label["action"])
        ok = intent_ok(want, observed)
        exact = normalize(want_text) == normalize(text)
        blocks[row["block"]].append({
            "file": row["file"],
            "ok": ok,
            "exact": exact,
            "wer": word_error(want_text, text),
        })
        if not ok:
            failures.append(
                f"{row['file']:<18} ждали {label['action']:<24} "
                f"услышали {text!r} -> {observed}"
            )

    header = f"{'блок':<5}{'N':>4}{'действие':>10}{'текст':>8}{'WER':>7}"
    print(header)
    print("-" * len(header))
    for block in "BCDEFGH":
        items = blocks.get(block, [])
        if not items:
            continue
        print(f"{block:<5}{len(items):>4}"
              f"{sum(i['ok'] for i in items) / len(items) * 100:>9.0f}%"
              f"{sum(i['exact'] for i in items) / len(items) * 100:>7.0f}%"
              f"{np.mean([i['wer'] for i in items]):>7.2f}")

    all_items = [
        i for block, items in blocks.items() if block != "H" for i in items
    ]
    print(f"{'все':<5}{len(all_items):>4}"
          f"{sum(i['ok'] for i in all_items) / len(all_items) * 100:>9.0f}%"
          f"{sum(i['exact'] for i in all_items) / len(all_items) * 100:>7.0f}%"
          f"{np.mean([i['wer'] for i in all_items]):>7.2f}")

    print()
    print(f"Ошибки действий: {len(failures)}")
    for line in failures:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
