"""Автоматическая проверка синтезированных клипов перед прослушиванием.

Считает длительность, пик, клиппинг и распознаёт каждый клип GigaAM.
Клип считается проблемным, если в нём не слышно «луна», есть клиппинг
или длительность/уровень выходят за пределы этапа 0.

Пример:
    python wake_model/scripts/check_qc.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from app.config import load_config as load_app_config  # noqa: E402
from app.stt import create_transcriber  # noqa: E402
from app.text import normalize  # noqa: E402

HEY = {"хей", "хэй", "эй", "хэ", "хе", "ей", "ай", "а"}
MIN_DURATION = 0.70
MAX_DURATION = 2.50
MIN_PEAK = -6.0
MAX_PEAK = -1.0


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: ожидается 16-битный PCM")
        channels = f.getnchannels()
        rate = f.getframerate()
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return pcm, rate


def db(value: float) -> float:
    return 20.0 * float(np.log10(max(value, 1e-9)))


def load_metadata(out: Path) -> dict[str, dict]:
    meta = out / "metadata.csv"
    if not meta.exists():
        return {}
    with open(meta, encoding="utf-8", newline="") as f:
        return {row["file"]: row for row in csv.DictReader(f, delimiter=";")}


def classify(text: str) -> tuple[bool, bool, bool]:
    """Возвращает (есть «луна», есть «хей/эй», точное совпадение)."""
    tokens = normalize(text).split()
    has_luna = any(token.startswith("лун") for token in tokens)
    has_hey = any(token in HEY for token in tokens)
    return has_luna, has_hey, tokens == ["хей", "луна"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=ROOT / "data" / "piper_qc")
    parser.add_argument("--no-stt", action="store_true",
                        help="только формат и уровни, без GigaAM")
    args = parser.parse_args()

    out: Path = args.dir
    wavs = sorted(p for p in out.glob("*.wav") if not p.name.startswith("_"))
    if not wavs:
        print(f"Нет WAV в {out}", file=sys.stderr)
        return 2

    metadata = load_metadata(out)
    rows: list[dict] = []
    problems: list[str] = []

    transcriber = None
    if not args.no_stt:
        app_cfg = load_app_config()
        transcriber = create_transcriber(app_cfg)
        transcriber.load()

    for wav in wavs:
        pcm, rate = read_wav(wav)
        duration = len(pcm) / rate
        peak = float(np.abs(pcm.astype(np.float32)).max())
        peak_db = db(peak / 32768.0)
        rms_db = db(float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) / 32768.0)
        clipped = int(np.count_nonzero(np.abs(pcm.astype(np.int32)) >= 32767))
        text = ""
        if transcriber is not None:
            text = transcriber.transcribe(pcm.astype(np.int16).tobytes())
        has_luna, has_hey, exact = classify(text)

        reasons = []
        if not (MIN_DURATION <= duration <= MAX_DURATION):
            reasons.append(f"длительность {duration:.2f} с")
        if not (MIN_PEAK <= peak_db <= MAX_PEAK):
            reasons.append(f"пик {peak_db:.1f} dBFS")
        if clipped:
            reasons.append(f"клиппинг {clipped} отсчётов")
        if transcriber is not None and not has_luna:
            reasons.append(f"нет «луна»: «{text}»")
        if reasons:
            problems.append(f"{wav.name}: " + ", ".join(reasons))

        rows.append({
            "file": wav.name,
            "voice": metadata.get(wav.name, {}).get("voice", "?"),
            "duration": duration,
            "peak_db": peak_db,
            "rms_db": rms_db,
            "clipped": clipped,
            "text": text,
            "has_luna": has_luna,
            "has_hey": has_hey,
            "exact": exact,
            "params": metadata.get(wav.name, {}),
        })

    durations = [row["duration"] for row in rows]
    peaks = [row["peak_db"] for row in rows]
    variants = Counter(row["text"] for row in rows)
    lines = [
        "Синтез QC",
        f"  клипов: {len(rows)}",
        f"  голосов: {len({row['voice'] for row in rows})}",
        f"  длительность: {min(durations):.2f}–{max(durations):.2f} с "
        f"(среднее {statistics.mean(durations):.2f})",
        f"  пик: {min(peaks):.1f}…{max(peaks):.1f} dBFS",
    ]
    if transcriber is not None:
        exact = sum(row["exact"] for row in rows)
        luna = sum(row["has_luna"] for row in rows)
        hey = sum(row["has_hey"] for row in rows)
        lines += [
            "",
            "GigaAM",
            f"  точно «хей луна»: {exact}/{len(rows)}",
            f"  есть «луна»: {luna}/{len(rows)}",
            f"  есть «хей/эй/ай»: {hey}/{len(rows)}",
            "  варианты:",
        ]
        for text, count in variants.most_common():
            lines.append(f"    {count:3d}  «{text or '—'}»")

    by_voice = Counter(row["voice"] for row in rows)
    lines += ["", "По голосам"]
    for voice, count in sorted(by_voice.items()):
        subset = [row for row in rows if row["voice"] == voice]
        exact_count = sum(row["exact"] for row in subset)
        luna_count = sum(row["has_luna"] for row in subset)
        lines.append(
            f"  {voice}: {count}, точно «хей луна» {exact_count}, "
            f"есть «луна» {luna_count}")

    params = [row["params"] for row in rows if row["params"]]
    if params:
        lines += ["", "Параметры синтеза"]
        for key in ("length_scale", "noise_scale", "noise_w"):
            if all(key in row for row in params):
                values = [float(row[key]) for row in params]
                lines.append(f"  {key}: {min(values):.2f}–{max(values):.2f}")
        if all("variant" in row for row in params):
            counts = Counter(row["variant"] for row in params)
            lines.append("  варианты: " + ", ".join(
                f"{name} {count}" for name, count in sorted(counts.items())))

    lines += ["", "Проблемы"]
    lines += [f"  {problem}" for problem in problems] or ["  нет"]

    report = "\n".join(lines) + "\n"
    path = out / "qc_report.txt"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Отчёт: {path}")

    csv_path = out / "qc_files.csv"
    fields = [
        "file", "voice", "duration_s", "peak_db", "rms_db", "text",
        "has_luna", "has_hey", "exact", "variant",
        "length_scale", "noise_scale", "noise_w",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            record = {
                "file": row["file"],
                "voice": row["voice"],
                "duration_s": f"{row['duration']:.3f}",
                "peak_db": f"{row['peak_db']:.1f}",
                "rms_db": f"{row['rms_db']:.1f}",
                "text": row["text"],
                "has_luna": int(row["has_luna"]),
                "has_hey": int(row["has_hey"]),
                "exact": int(row["exact"]),
            }
            record.update({key: row["params"].get(key, "") for key in fields[9:]})
            writer.writerow(record)
    print(f"По файлам: {csv_path}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
