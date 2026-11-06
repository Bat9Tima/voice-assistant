"""Собирает проверенные синтетические позитивы в единый manifest.

Читает `qc_files.csv` из подкаталогов `data/positives_tts`, отбрасывает
клипы без «луна»/«хей», с клиппингом и выходом за пределы длительности
или уровня. Итог — `manifest.csv` с относительными путями для обучения.

Пример:
    python wake_model/scripts/collect_positives.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from app.text import normalize  # noqa: E402

HEY = {"хей", "хэй", "эй", "хэ", "хе", "ей", "ай", "а"}


def classify(text: str) -> tuple[bool, bool, bool]:
    tokens = normalize(text).split()
    has_luna = any("лун" in token for token in tokens)
    has_hey = any(
        token in HEY or token.startswith(("хей", "хэй")) for token in tokens)
    return has_luna, has_hey, tokens == ["хей", "луна"]


def read_qc(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=ROOT / "data" / "positives_tts")
    parser.add_argument("--min-duration", type=float, default=0.70)
    parser.add_argument("--max-duration", type=float, default=2.50)
    parser.add_argument("--min-peak", type=float, default=-6.0)
    parser.add_argument("--max-peak", type=float, default=-1.0)
    parser.add_argument("--asr-exempt", action="append", default=["eugene"],
                        help="голоса, для которых GigaAM не фильтр "
                             "(по умолчанию eugene: его речь ASR слышит плохо)")
    args = parser.parse_args()

    root: Path = args.root
    exempt = set(args.asr_exempt)
    qc_files = sorted(root.glob("*/qc_files.csv"))
    if not qc_files:
        print(f"Нет qc_files.csv в {root}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for qc_path in qc_files:
        source = qc_path.parent.name
        for row in read_qc(qc_path):
            duration = float(row["duration_s"])
            peak = float(row["peak_db"])
            clipped = int(row.get("clipped") or 0)
            has_luna, has_hey, exact = classify(row.get("text", ""))
            reasons = []
            if not (args.min_duration <= duration <= args.max_duration):
                reasons.append("длительность")
            if not (args.min_peak <= peak <= args.max_peak):
                reasons.append("уровень")
            if clipped:
                reasons.append("клиппинг")
            if not has_luna:
                reasons.append("нет луна")
            if not has_hey:
                reasons.append("нет хей")
            if reasons and row.get("voice") in exempt:
                reasons = [reason for reason in reasons
                           if reason not in ("нет луна", "нет хей")]
            rows.append({
                "path": f"{source}/{row['file']}",
                "source": source,
                "voice": row.get("voice", "?"),
                "variant": row.get("variant", ""),
                "text": row.get("text", ""),
                "duration_s": f"{duration:.3f}",
                "exact": int(exact),
                "kept": int(not reasons),
                "reason": ",".join(reasons),
            })

    manifest = root / "manifest.csv"
    fields = ["path", "source", "voice", "variant", "text", "duration_s",
              "exact", "kept", "reason"]
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    kept = [row for row in rows if row["kept"]]
    print(f"Клипов: {len(rows)}, оставлено: {len(kept)}")
    for source, count in sorted(Counter(r["source"] for r in kept).items()):
        exact = sum(r["exact"] for r in kept if r["source"] == source)
        print(f"  {source}: {count}, точно «хей луна» {exact}")
    for voice, count in sorted(Counter(r["voice"] for r in kept).items()):
        print(f"  {voice}: {count}")
    dropped = Counter(r["reason"] for r in rows if not r["kept"])
    if dropped:
        print("Отброшено:")
        for reason, count in sorted(dropped.items()):
            print(f"  {reason}: {count}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
