# -*- coding: utf-8 -*-
"""Печатает детали baseline: блок B, ошибки команд, негативы H."""

import csv
import json
from pathlib import Path

data = json.loads(Path("tools/baseline_results.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader(
    open(Path.home() / "Documents" / "VoiceDataset" / "labels.csv",
         encoding="utf-8-sig", newline=""),
    delimiter=";",
))
by = {r["file"]: r for r in rows}

print("=== B block ===")
for r in data["results"]:
    if r["block"] == "B":
        print(f"{r['file']:<18} want {r['expected']:<22} "
              f"text={r['text']!r:<32} stripped={r['stripped']!r:<28} "
              f"obs={r['observed']}")

print()
print("=== command failures ===")
for r in data["results"]:
    if r["kind"] in ("command", "wake_command") and not r["ok"]:
        print(f"{r['file']:<18} want {r['expected']:<22} "
              f"text={r['text']!r:<36} stripped={r['stripped']!r:<32} "
              f"obs={r['observed']}")

print()
print("=== H block ===")
for r in data["results"]:
    if r["block"] == "H":
        print(f"{r['file']:<18} kind={r['kind']:<8} "
              f"label={by[r['file']]['text']!r:<16} "
              f"text={r['text']!r:<24} stripped={r['stripped']!r}")
