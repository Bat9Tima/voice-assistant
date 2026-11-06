#!/usr/bin/env python
"""QC датасета: формат, уровни, обрезка, дубли, соответствие разметке.

Только читает audio/ и labels.csv. Отчёт пишет рядом с датасетом:
qc_report.txt. Ничего не перезаписывает и не удаляет.
"""

from __future__ import annotations

import csv
import hashlib
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import record  # noqa: E402

BASE = Path.home() / "Documents" / "VoiceDataset"
AUDIO = BASE / "audio"
REPORT = BASE / "qc_report.txt"

LEVEL_RANGES = record.LEVEL_RANGES
FRAME = int(0.02 * record.RATE)
# Старые A–L не содержат этих колонок; пусто = запись в тишине, собой, в комнате.
DEFAULTS = {
    "condition": record.DEFAULT_CONDITION,
    "speaker": record.DEFAULT_SPEAKER,
    "room": record.DEFAULT_ROOM,
}


def dbfs(value: float) -> float:
    return 20.0 * np.log10(max(float(value), 1e-9))


def same_label_value(key: str, actual: str, wanted: str) -> bool:
    """Совпадает ли значение разметки с планом (пустое = default для A–L)."""
    if actual.strip() == wanted:
        return True
    return not actual.strip() and wanted == DEFAULTS.get(key)


def read_wav(path: Path) -> tuple[np.ndarray, dict]:
    with wave.open(str(path), "rb") as f:
        info = {
            "channels": f.getnchannels(),
            "width": f.getsampwidth(),
            "rate": f.getframerate(),
            "frames": f.getnframes(),
        }
        raw = f.readframes(info["frames"])
    return np.frombuffer(raw, dtype=np.int16), info


def frame_levels(samples: np.ndarray) -> np.ndarray:
    count = samples.size // FRAME
    if count == 0:
        return np.zeros(0)
    blocks = samples[: count * FRAME].astype(np.float32) / 32768.0
    blocks = blocks.reshape(count, FRAME)
    return np.sqrt((blocks ** 2).mean(axis=1))


def speech_span(levels: np.ndarray) -> tuple[int, int] | None:
    if levels.size < 5:
        return None
    loud = float(np.percentile(levels, 95))
    noise = float(np.percentile(levels, 10))
    threshold = max(noise * 1.5, loud * 0.04, 0.0015)
    mask = levels > threshold
    if not mask.any() or mask.sum() * 0.02 < 0.15:
        return None
    first = int(np.argmax(mask))
    last = levels.size - 1 - int(np.argmax(mask[::-1]))
    return first, last


def fingerprint(samples: np.ndarray) -> np.ndarray:
    """Грубый отпечаток: 16 полос × 40 долей времени."""
    x = samples.astype(np.float32) / 32768.0
    hop = int(0.01 * record.RATE)
    win = int(0.025 * record.RATE)
    if x.size < win:
        return np.zeros(16 * 40, dtype=np.float32)
    count = 1 + (x.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(count)[:, None]
    frames = x[idx] * np.hanning(win)[None, :]
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(win, 1.0 / record.RATE)
    edges = np.geomspace(100, 7000, 17)
    bands = np.zeros((count, 16), dtype=np.float32)
    for b in range(16):
        mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if mask.any():
            bands[:, b] = spec[:, mask].mean(axis=1)
    bands = np.log10(bands + 1e-10)
    # 40 долей времени
    src = np.linspace(0, count - 1, 40)
    out = np.zeros((40, 16), dtype=np.float32)
    for i, pos in enumerate(src):
        lo = int(np.floor(pos))
        hi = min(lo + 1, count - 1)
        w = pos - lo
        out[i] = bands[lo] * (1 - w) + bands[hi] * w
    vec = out - out.mean()
    norm = float(np.linalg.norm(vec))
    return (vec / norm).ravel() if norm > 0 else vec.ravel()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def main() -> int:
    rows = list(csv.DictReader(
        open(BASE / "labels.csv", encoding="utf-8-sig", newline=""),
        delimiter=";",
    ))
    plan = {int(r["index"]): r for r in record.build_full_plan()}

    issues: dict[str, list[str]] = defaultdict(list)
    stats: list[dict] = []
    prints: list[tuple[str, np.ndarray, float]] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    labels_mismatch: list[str] = []

    on_disk = {p.name for p in AUDIO.glob("*.wav")}
    known = {r["file"] for r in rows}
    for extra in sorted(on_disk - known):
        issues["лишние файлы"].append(extra)
    for missing in sorted(known - on_disk):
        issues["нет файла"].append(missing)

    for row in rows:
        index = int(row["index"])
        name = row["file"]
        path = AUDIO / name
        if not path.exists():
            continue

        expected = plan.get(index)
        if expected:
            for key in ("file", "block", "kind", "text", "action", "level",
                        "rate", "distance_cm", "condition", "speaker", "room",
                        "session", "take", "max_sec", "trim"):
                # В старых labels.csv новых колонок ещё нет — не считаем это ошибкой.
                if key not in row:
                    continue
                actual = str(row.get(key, ""))
                wanted = str(expected.get(key, ""))
                # Старые строки A–L писались до появления condition/speaker/room:
                # пустое значение там означает условие плана по умолчанию.
                if same_label_value(key, actual, wanted):
                    continue
                if actual != wanted:
                    labels_mismatch.append(
                        f"{name}: {key} = {row.get(key)!r}, "
                        f"в плане {expected.get(key)!r}"
                    )

        samples, info = read_wav(path)
        hashes[hashlib.sha1(path.read_bytes()).hexdigest()].append(name)

        dur = samples.size / record.RATE
        peak = dbfs(float(np.abs(samples).max()) / 32768.0) if samples.size else -120.0
        rms = dbfs(float(np.sqrt(np.mean(
            (samples.astype(np.float32) / 32768.0) ** 2
        )))) if samples.size else -120.0
        dc = float(np.mean(samples.astype(np.float32) / 32768.0)) if samples.size else 0.0
        clipped = int(np.sum(np.abs(samples.astype(np.int32)) >= 32767))
        levels = frame_levels(samples)
        span = speech_span(levels)

        item = {
            "index": index,
            "file": name,
            "block": row["block"],
            "level": row["level"],
            "condition": row.get("condition") or "unknown",
            "distance": row.get("distance_cm") or "",
            "dur": dur,
            "peak": peak,
            "rms": rms,
            "snr": None,
            "lead": None,
            "tail": None,
            "speech": None,
            "flags": [],
        }

        if (info["channels"], info["width"], info["rate"]) != (1, 2, 16000):
            issues["формат WAV"].append(
                f"{name}: {info['channels']} ch, {info['width'] * 8} бит, "
                f"{info['rate']} Гц"
            )
        if info["frames"] == 0:
            issues["пустой файл"].append(name)
            stats.append(item)
            continue

        if clipped:
            issues["клиппинг"].append(f"{name}: {clipped} сэмплов на пределе")
        if abs(dc) > 0.01:
            issues["постоянная составляющая"].append(f"{name}: DC {dc:+.3f}")

        if span is None:
            if str(row.get("trim", "1")) == "1":
                issues["речь не найдена"].append(name)
        else:
            first, last = span
            lead = first * FRAME / record.RATE
            tail = (levels.size - 1 - last) * FRAME / record.RATE
            nonzero = levels[levels > 1e-6]
            noise = float(np.percentile(nonzero, 15)) if nonzero.size else 0.0
            loudest = float(levels.max()) if levels.size else 0.0
            speech_frames = levels[levels > loudest * 0.1]
            speech_ref = (float(np.median(speech_frames))
                          if speech_frames.size >= 3 else loudest)
            snr = dbfs(speech_ref) - dbfs(noise)
            item.update(snr=snr, lead=lead, tail=tail,
                        speech=(last - first + 1) * FRAME / record.RATE)

            # Края: сравниваем громкость первых/последних 0.3 с с речью.
            edge = int(0.3 * record.RATE)
            if samples.size > 2 * edge and str(row.get("trim", "1")) == "1":
                head_db = dbfs(float(np.sqrt(np.mean(
                    (samples[:edge].astype(np.float32) / 32768.0) ** 2
                ))))
                tail_db = dbfs(float(np.sqrt(np.mean(
                    (samples[-edge:].astype(np.float32) / 32768.0) ** 2
                ))))
                if head_db > dbfs(speech_ref) - 12:
                    issues["громкое начало"].append(
                        f"{name}: первые 0.3 с {head_db:.1f} dBFS при речи "
                        f"{dbfs(speech_ref):.1f}"
                    )
                if tail_db > dbfs(speech_ref) - 12:
                    issues["громкий конец"].append(
                        f"{name}: последние 0.3 с {tail_db:.1f} dBFS при речи "
                        f"{dbfs(speech_ref):.1f}"
                    )
            if snr < 8.0:
                issues["низкий SNR"].append(f"{name}: {snr:.1f} дБ")

        low, high = LEVEL_RANGES.get(row["level"], (-60.0, 0.0))
        if peak < low - 2:
            issues["тише целевого уровня"].append(
                f"{name}: пик {peak:.1f} dBFS, цель {low:.0f}…{high:.0f}"
            )
        if peak > high + 2:
            issues["громче целевого уровня"].append(
                f"{name}: пик {peak:.1f} dBFS, цель {low:.0f}…{high:.0f}"
            )

        try:
            labeled_peak = float(row["peak_db"])
            if abs(labeled_peak - peak) > 1.0:
                issues["пик не совпал с разметкой"].append(
                    f"{name}: в CSV {labeled_peak:.1f}, в файле {peak:.1f}"
                )
        except (TypeError, ValueError):
            pass
        try:
            labeled_dur = float(row["dur_s"])
            if abs(labeled_dur - dur) > 0.05:
                issues["длительность не совпала"].append(
                    f"{name}: в CSV {labeled_dur:.2f}, в файле {dur:.2f}"
                )
        except (TypeError, ValueError):
            pass

        stats.append(item)
        prints.append((name, fingerprint(samples), peak))

    for row in rows:
        if str(row.get("block", "")) not in {"M", "N", "O"}:
            continue
        for key in ("condition", "speaker", "room"):
            if not str(row.get(key) or "").strip():
                issues["нет метаданных"].append(f"{row['file']}: {key}")

    for _digest, names in hashes.items():
        if len(names) > 1:
            issues["одинаковые файлы"].append(", ".join(names))

    # Дубли по отпечатку
    similar = []
    for i in range(len(prints)):
        for j in range(i + 1, len(prints)):
            score = cosine(prints[i][1], prints[j][1])
            if score > 0.985:
                similar.append((score, prints[i][0], prints[j][0]))
    similar.sort(reverse=True)
    for score, a, b in similar[:10]:
        issues["похожие дубли"].append(f"{a} ~ {b}: {score:.3f}")

    # Длительность: файл не должен быть длиннее лимита записи + 1 с обрезки
    for row in rows:
        item = next((i for i in stats if i["index"] == int(row["index"])), None)
        if item is None:
            continue
        try:
            limit = float(row["max_sec"]) + 1.05
        except (TypeError, ValueError):
            continue
        if item["dur"] > limit:
            issues["длиннее лимита"].append(
                f"{item['file']}: {item['dur']:.2f} с при лимите {limit:.2f}"
            )
        if item["dur"] < 0.5:
            issues["слишком короткий"].append(
                f"{item['file']}: {item['dur']:.2f} с"
            )

    # ------------------------------------------------------------ отчёт
    lines: list[str] = []
    add = lines.append
    add(f"QC датасета: {BASE}")
    add(f"Дублей в разметке: {len(rows)} · файлов в audio: {len(on_disk)}")
    add(f"Статусы: {dict(Counter(r['status'] for r in rows))}")
    add("")

    add("По блокам")
    add(f"{'блок':<5}{'N':>4}{'пик медиана':>14}{'SNR медиана':>14}"
        f"{'длит. медиана':>16}{'проблем':>10}")
    by_block: dict[str, list[dict]] = defaultdict(list)
    for item in stats:
        by_block[item["block"]].append(item)
    for block in sorted(by_block):
        items = by_block[block]
        peaks = [i["peak"] for i in items]
        snrs = [i["snr"] for i in items if i["snr"] is not None]
        durs = [i["dur"] for i in items]
        bad = sum(1 for i in items if i["flags"]) or 0
        add(f"{block:<5}{len(items):>4}{np.median(peaks):>13.1f}"
            f"{(np.median(snrs) if snrs else float('nan')):>13.1f}"
            f"{np.median(durs):>15.2f}{bad:>10}")
    add("")

    cond_items = [i for i in stats if i["block"] in {"M", "N", "O"}]
    if cond_items:
        add("По условиям (M/N/O)")
        add(f"{'блок':<5}{'условие':<10}{'дист.':>7}{'N':>4}"
            f"{'пик':>9}{'SNR':>8}")
        grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for item in cond_items:
            grouped[(item["block"], item["condition"],
                     str(item["distance"]))].append(item)
        for key in sorted(grouped):
            items = grouped[key]
            peaks = [i["peak"] for i in items]
            snrs = [i["snr"] for i in items if i["snr"] is not None]
            add(f"{key[0]:<5}{key[1]:<10}{key[2]:>7}{len(items):>4}"
                f"{np.median(peaks):>9.1f}"
                f"{(np.median(snrs) if snrs else float('nan')):>8.1f}")
        add("")

    add("Проблемы")
    if not issues and not labels_mismatch:
        add("  нет")
    for kind in sorted(issues):
        add(f"  {kind}: {len(issues[kind])}")
        for text in issues[kind]:
            add(f"    - {text}")
    if labels_mismatch:
        add(f"  разметка не совпала с планом: {len(labels_mismatch)}")
        for text in labels_mismatch:
            add(f"    - {text}")
    add("")

    # Файлы на перезапись: любые замечания, кроме мягких
    hard = {"нет файла", "пустой файл", "формат WAV", "клиппинг",
            "речь не найдена", "громкое начало", "громкий конец",
            "одинаковые файлы", "лишние файлы", "длиннее лимита",
            "слишком короткий"}
    redo = sorted({
        line.split(":")[0].strip()
        for kind in hard
        for line in issues.get(kind, [])
    })
    add("Перезаписать в первую очередь")
    add("  " + (", ".join(redo) if redo else "нечего"))

    text = "\n".join(lines) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    print(f"Отчёт: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
