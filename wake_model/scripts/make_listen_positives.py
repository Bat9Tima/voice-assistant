"""Страница для выборочного прослушивания синтетических позитивов.

Берёт `manifest.csv` после QC и выбирает по два клипа на голос: один
точный «хей луна» и один с другим вариантом распознавания, если есть.
Нужна для ручного контроля этапа 1.

Пример:
    python wake_model/scripts/make_listen_positives.py
"""

from __future__ import annotations

import argparse
import csv
import html
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return [row for row in csv.DictReader(f, delimiter=";")
                if row["kept"] == "1"]


def choose(rows: list[dict], rng: random.Random) -> list[dict]:
    exact = [row for row in rows if row["exact"] == "1"]
    other = [row for row in rows if row["exact"] != "1"]
    picked = []
    if exact:
        picked.append(rng.choice(exact))
    if other:
        picked.append(rng.choice(other))
    if len(picked) < 2:
        rest = [row for row in rows if row not in picked]
        picked.extend(rng.sample(rest, min(2 - len(picked), len(rest))))
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=ROOT / "data" / "positives_tts")
    parser.add_argument("--per-voice", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = read_manifest(args.root / "manifest.csv")
    by_voice: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_voice[row["voice"]].append(row)

    rng = random.Random(args.seed)
    cards = []
    for voice in sorted(by_voice):
        picked = choose(by_voice[voice], rng)
        for row in picked[:args.per_voice]:
            meta = [row["source"], row["variant"], f"{row['duration_s']} с"]
            if row.get("text"):
                meta.append(f"ASR: «{row['text']}»")
            cards.append(
                '<div class="card">'
                f'<div class="title">{html.escape(voice)}</div>'
                f'<audio controls preload="metadata" '
                f'src="{html.escape(row["path"])}"></audio>'
                f'<div class="meta">{html.escape(" · ".join(meta))}</div>'
                "</div>")

    page = f"""<!doctype html>
<html lang="ru">
<meta charset="utf-8">
<title>Синтетические позитивы — выборочный контроль</title>
<style>
  body {{ font-family: Segoe UI, sans-serif; background:#14161a; color:#e8e8e8;
         max-width: 900px; margin: 24px auto; padding: 0 16px; }}
  h1 {{ font-size: 22px; font-weight: 600; }}
  p {{ color:#a8adb5; line-height: 1.5; }}
  .card {{ background:#1e222a; border:1px solid #2c323d; border-radius:10px;
           padding:14px 16px; margin:12px 0; }}
  .title {{ font-weight:600; margin-bottom:8px; }}
  audio {{ width:100%; }}
  .meta {{ color:#8f96a3; font-size:12px; margin-top:6px; }}
  .hint {{ background:#1b2330; border-left:3px solid #4b7bec; padding:10px 14px; }}
</style>
<h1>Позитивы этапа 1: выборочный контроль</h1>
<p class="hint">По два клипа на голос. Проверить: слышно ли «хей луна»,
нет ли «кей/фей», проглатывания слогов, хрипа и обрывов. Всего
{len(by_voice)} голосов, {len(cards)} клипов.</p>
{''.join(cards)}
</html>
"""
    path = args.root / "listen_samples.html"
    path.write_text(page, encoding="utf-8")
    print(f"Голосов: {len(by_voice)}, клипов: {len(cards)}")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
