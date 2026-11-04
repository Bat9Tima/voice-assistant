#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Baseline на записанном датасете: wake-слово и команды.

Ничего не выполняет: браузер и системная громкость заменены заглушками.
Модели те же, что в ассистенте: openWakeWord hey_loona + GigaAM v3.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import wave
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.actions import ActionRouter  # noqa: E402
from app.config import MODELS_DIR, load_config  # noqa: E402
from app.stt import GigaAMTranscriber  # noqa: E402
from app.text import levenshtein, normalize, strip_wake_word  # noqa: E402

BASE = Path.home() / "Documents" / "VoiceDataset"
AUDIO = BASE / "audio"
RESULTS = Path(__file__).resolve().parent / "baseline_results.json"
WAKE_MODEL = MODELS_DIR / "openwakeword" / "hey_loona.onnx"
THRESHOLDS = (0.5, 0.45, 0.4, 0.35)


def read_pcm(name: str) -> bytes:
    with wave.open(str(AUDIO / name), "rb") as f:
        return f.readframes(f.getnframes())


class FakeBrowser:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, action: str, timeout: int = 90, **args) -> dict:
        self.calls.append((action, args))
        return {"ok": True, "data": {}}


class FakeVolume:
    def __init__(self) -> None:
        self.level = 0.5
        self.muted = False

    def GetMasterVolumeLevelScalar(self) -> float:
        return self.level

    def SetMasterVolumeLevelScalar(self, value, context) -> None:
        self.level = float(value)

    def GetMute(self) -> int:
        return int(self.muted)

    def SetMute(self, value, context) -> None:
        self.muted = bool(value)


class Router:
    """ActionRouter с записью намерения вместо побочных эффектов."""

    def __init__(self, cfg: dict) -> None:
        self.browser = FakeBrowser()
        self.volume = FakeVolume()
        self.intent: tuple | None = None
        self.router = ActionRouter(cfg, self.browser)
        self.router._volume = self.volume

        original_change = self.router._change_volume
        original_set = self.router._set_volume
        original_mute = self.router._set_mute

        def change(delta):
            self.intent = ("volume_delta", round(delta * 100))
            return original_change(delta)

        def set_volume(level):
            self.intent = ("volume_set", round(level * 100))
            return original_set(level)

        def set_mute(mute):
            self.intent = ("mute", bool(mute))
            return original_mute(mute)

        self.router._change_volume = change
        self.router._set_volume = set_volume
        self.router._set_mute = set_mute

    def run(self, text: str) -> tuple:
        self.browser.calls.clear()
        self.intent = None
        self.router.handle(text)
        if self.intent is not None:
            return self.intent
        if self.browser.calls:
            action, args = self.browser.calls[-1]
            return ("browser", action, args)
        return ("none",)


def expected_intent(action: str) -> tuple:
    if not action:
        return ("none",)
    parts = action.split(":")
    if parts[0] == "volume":
        if len(parts) == 2:
            value = parts[1]
            if value.startswith(("+", "-")):
                return ("volume_delta", int(value))
            return ("volume_set", int(value))
        if len(parts) == 3:
            return ("browser", "soundcloud_volume",
                    {"mode": "set", "value": int(parts[2])})
    if action == "mute":
        return ("mute", True)
    if action in {"newtab", "closetab"}:
        return ("browser", action, {})
    if action == "media:next":
        return ("browser", "next", {})
    if action == "media:prev":
        return ("browser", "prev", {})
    if action == "goto:youtube":
        return ("browser", "goto", {"site": "youtube"})
    if action.startswith("youtube:"):
        return ("browser", "youtube", {"query": action.split(":", 1)[1]})
    return ("unknown", action)


def intent_ok(expected: tuple, observed: tuple) -> bool:
    if expected[0] != observed[0]:
        return False
    if expected[0] in {"volume_set", "volume_delta", "mute"}:
        return expected == observed
    if expected[0] == "none":
        return True
    if expected[0] != "browser":
        return False
    if expected[1] != observed[1]:
        return False

    kind = expected[1]
    if kind in {"newtab", "closetab", "next", "prev"}:
        return True
    args = observed[2] if len(observed) > 2 else {}
    if kind == "goto":
        url = str(args.get("url", "")).lower()
        return expected[2]["site"] in url
    if kind == "youtube":
        want = normalize(expected[2]["query"]).replace(" ", "")
        got = normalize(str(args.get("query", ""))).replace(" ", "")
        if not got:
            return False
        return SequenceMatcher(None, want, got).ratio() >= 0.8
    if kind == "soundcloud_volume":
        return (args.get("mode") == "set"
                and int(args.get("value", -1)) == expected[2]["value"])
    return False


def word_error(expected: str, observed: str) -> float:
    want = normalize(expected).split()
    got = normalize(observed).split()
    if not want:
        return 0.0 if not got else 1.0
    return levenshtein(" ".join(want), " ".join(got)) / len(" ".join(want))


def wake_scores(names: list[str]) -> dict[str, list[float]]:
    """Максимальный score по 8 фазам потока (шаг 10 мс).

    openWakeWord режет поток на окна по 80 мс, и score зависит от того,
    как фраза легла на границу окна. В живом захвате эта фаза случайна,
    поэтому одного прогона мало.
    """
    from openwakeword.model import Model

    step = 1280  # 80 мс
    phases = range(0, step, 160)  # 8 фаз по 10 мс
    lead = np.zeros(16000, dtype=np.int16)  # 1 с тишины, как простой ассистента
    scores: dict[str, list[float]] = {}
    for name in names:
        pcm = np.frombuffer(read_pcm(name), dtype=np.int16)
        stream = np.concatenate([lead, pcm])
        values = []
        for phase in phases:
            model = Model(wakeword_models=[str(WAKE_MODEL)],
                          inference_framework="onnx")
            key = next(iter(model.models))
            best = 0.0
            for start in range(phase, stream.size - step + 1, step):
                chunk = model.predict(stream[start:start + step])
                best = max(best, float(chunk.get(key, 0.0)))
            values.append(best)
        scores[name] = values
    return scores


def main() -> int:
    rows = list(csv.DictReader(
        open(BASE / "labels.csv", encoding="utf-8-sig", newline=""),
        delimiter=";",
    ))
    by_file = {r["file"]: r for r in rows}
    names = [r["file"] for r in rows]
    cfg = load_config()
    aliases = cfg.get("wake_aliases") or ["хей луна"]

    print("Загружаю openWakeWord...")
    t0 = time.perf_counter()
    scores = wake_scores(names)
    wake_time = time.perf_counter() - t0

    print("Загружаю GigaAM v3...")
    stt = GigaAMTranscriber(
        num_threads=int((cfg.get("stt") or {}).get("num_threads") or 2),
        tail_padding_sec=float((cfg.get("stt") or {}).get("tail_padding_sec", 0.4)),
    )
    stt.load()
    router = Router(cfg)

    results = []
    for row in rows:
        name = row["file"]
        pcm = read_pcm(name)
        t0 = time.perf_counter()
        text = stt.transcribe(pcm)
        decode = time.perf_counter() - t0
        stripped = strip_wake_word(text, aliases)
        expected = row["action"]
        observed = router.run(stripped)
        want = expected_intent(expected)
        results.append({
            "index": int(row["index"]),
            "file": name,
            "block": row["block"],
            "kind": row["kind"],
            "level": row["level"],
            "expected": expected,
            "expected_text": row["text"],
            "text": text,
            "stripped": stripped,
            "observed": observed,
            "ok": intent_ok(want, observed),
            "wer": word_error(row["text"], stripped),
            "decode_sec": decode,
            "wake": scores.get(name, [0.0]),
        })

    # ------------------------------------------------------------------ wake
    def group(pred) -> list[dict]:
        return [r for r in results if pred(by_file[r["file"]])]

    wake = group(lambda r: r["kind"] == "wake")
    alias = group(lambda r: r["kind"] == "wake_alias")
    wake_cmd = group(lambda r: r["kind"] == "wake_command")
    negatives = group(lambda r: r["kind"] == "negative")

    print()
    print(f"WAKE: {WAKE_MODEL.name}, файлов {len(results)}, "
          f"прогон {wake_time:.1f} с, фаз на файл 8")
    print()
    print("Вероятность срабатывания, % (среднее по фазам):")
    header = (f"{'порог':>6} | {'A 1-8':>7} {'A тихо':>7} {'A 2м':>6} "
              f"{'A2':>6} {'B':>6} {'H ложн':>7}")
    print(header)
    print("-" * len(header))

    def rate(items: list[dict], threshold: float) -> float:
        values = [v for r in items for v in r["wake"]]
        if not values:
            return 0.0
        return sum(1 for v in values if v >= threshold) / len(values) * 100

    for threshold in THRESHOLDS:
        normal = [r for r in wake if int(r["index"]) <= 8]
        quiet = [r for r in wake if 9 <= int(r["index"]) <= 10]
        far = [r for r in wake if int(r["index"]) >= 11]
        print(f"{threshold:>6.2f} | {rate(normal, threshold):>6.0f}% "
              f"{rate(quiet, threshold):>6.0f}% {rate(far, threshold):>5.0f}% "
              f"{rate(alias, threshold):>5.0f}% {rate(wake_cmd, threshold):>5.0f}% "
              f"{rate(negatives, threshold):>6.0f}%")

    misses = [r for r in wake if max(r["wake"]) < 0.4]
    if misses:
        print()
        print("Пропуски wake (даже лучшая фаза < 0.40):")
        for r in misses:
            print(f"  {r['file']:<24} max {max(r['wake']):.3f}  "
                  f"({by_file[r['file']]['level']})")
    false = [r for r in negatives if max(r["wake"]) >= 0.4]
    if false:
        print()
        print("Ложные срабатывания H (лучшая фаза ≥ 0.40):")
        for r in false:
            print(f"  {r['file']:<24} max {max(r['wake']):.3f}  "
                  f"«{by_file[r['file']]['text']}»")

    # -------------------------------------------------------------- команды
    print()
    print("КОМАНДЫ")
    header = (f"{'блок':<5}{'N':>4}{'действие':>10}{'текст':>8}"
              f"{'WER':>7}{'ср. сек':>10}")
    print(header)
    print("-" * len(header))

    blocks: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        blocks[r["block"]].append(r)
    for block in "BCDEFGH":
        items = blocks.get(block, [])
        if not items:
            continue
        ok = sum(1 for r in items if r["ok"])
        exact = sum(1 for r in items if normalize(r["expected_text"]) ==
                    normalize(r["stripped"]))
        wer = float(np.mean([r["wer"] for r in items]))
        sec = float(np.mean([r["decode_sec"] for r in items]))
        print(f"{block:<5}{len(items):>4}{ok / len(items) * 100:>9.0f}%"
              f"{exact / len(items) * 100:>7.0f}%{wer:>7.2f}{sec:>10.2f}")

    all_cmd = [r for r in results if r["kind"] in ("command", "wake_command")]
    ok = sum(1 for r in all_cmd if r["ok"])
    print(f"{'все':<5}{len(all_cmd):>4}{ok / len(all_cmd) * 100:>9.0f}%"
          f"{sum(1 for r in all_cmd if normalize(r['expected_text']) == normalize(r['stripped'])) / len(all_cmd) * 100:>7.0f}%"
          f"{np.mean([r['wer'] for r in all_cmd]):>7.2f}"
          f"{np.mean([r['decode_sec'] for r in all_cmd]):>10.2f}")

    bad = [r for r in all_cmd if not r["ok"]]
    print()
    print(f"Ошибки действий: {len(bad)} из {len(all_cmd)}")
    for r in bad[:40]:
        print(f"  {r['file']:<24} ждали {r['expected']:<26} "
              f"услышали «{r['stripped']}»")

    neg_actions = [r for r in negatives if r["observed"] != ("none",)]
    print()
    print(f"Ложные действия на негативах: {len(neg_actions)} из {len(negatives)}")
    for r in neg_actions:
        print(f"  {r['file']:<24} «{r['stripped']}» -> {r['observed']}")

    payload = {
        "wake_model": WAKE_MODEL.name,
        "thresholds": list(THRESHOLDS),
        "wake_seconds": round(wake_time, 2),
        "results": results,
    }
    RESULTS.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print()
    print(f"Сырые результаты: {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
