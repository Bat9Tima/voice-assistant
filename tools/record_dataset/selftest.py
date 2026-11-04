#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Самотесты рекордера. Не трогают датасет: пишут только во временную папку."""

from __future__ import annotations

import math
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import record  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILED.append(name)


def synth(seconds, speech_start, speech_end, amp=0.1, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    total = int(seconds * record.RATE)
    signal = (rng.normal(0, noise, total) * 32768).astype(np.int16)
    a = int(speech_start * record.RATE)
    b = int(speech_end * record.RATE)
    if b > a:
        signal[a:b] = (rng.normal(0, amp, b - a) * 32768).astype(np.int16)
    return signal


# ------------------------------------------------------------------- review
def test_review() -> None:
    def run(keys):
        iterator = iter(keys)
        plays = []

        def read():
            try:
                return next(iterator)
            except StopIteration:
                return "space"

        result = record.review(read, lambda: plays.append(1))
        return result, len(plays)

    cases = [
        (["space"], "space", 0),
        (["redo"], "redo", 0),
        (["skip"], "skip", 0),
        (["quit"], "quit", 0),
        (["play", "space"], "space", 1),
        (["play", "play", "space"], "space", 2),
        (["play", "redo"], "redo", 1),
        (["play", "play", "quit"], "quit", 2),
        (["junk", "space"], "space", 0),
    ]
    for keys, expected, plays in cases:
        result, count = run(keys)
        check(
            f"review {keys} -> {expected}",
            result == expected and count == plays,
            f"got {result}, plays={count}",
        )


# --------------------------------------------------------------------- клавиши
def test_keys() -> None:
    cases = [
        (" ", "space"),
        ("\r", "space"),
        ("\n", "space"),
        ("p", "play"),
        ("P", "play"),
        ("з", "play"),
        ("З", "play"),
        ("r", "redo"),
        ("R", "redo"),
        ("к", "redo"),
        ("К", "redo"),
        ("s", "skip"),
        ("S", "skip"),
        ("ы", "skip"),
        ("Ы", "skip"),
        ("q", "quit"),
        ("Q", "quit"),
        ("й", "quit"),
        ("Й", "quit"),
        ("\x1b", "quit"),
    ]
    for ch, expected in cases:
        got = record.action_for(ch)
        check(f"key {ch!r} -> {expected}", got == expected, f"got {got!r}")

    for ch in ("a", "ф", "1", "", "\x00"):
        got = record.action_for(ch)
        check(f"key {ch!r} игнорируется", got is None, f"got {got!r}")

    # P и R в русской раскладке не должны попадать в «принять».
    check("key: з не space", record.action_for("з") != "space")
    check("key: к не space", record.action_for("к") != "space")

    # Фильтр: на экране приёма P/R/S/Q не должны приниматься.
    def key_from(keys):
        iterator = iter(keys)
        return lambda: next(iterator)

    result = record.wait_key(
        "", allowed=("space", "redo", "skip", "quit"), reader=key_from("pзs")
    )
    check("wait_key: P не принимает", result == "skip", f"got {result!r}")

    result = record.wait_key(
        "", allowed=("space", "redo", "skip", "quit"), reader=key_from("  ")
    )
    check("wait_key: пробел принимает", result == "space", f"got {result!r}")

    result = record.wait_key(
        "", allowed=("space", "redo", "skip", "quit"), reader=key_from("q")
    )
    check("wait_key: Q выходит", result == "quit", f"got {result!r}")

    result = record.wait_key(
        "", allowed=("space", "redo", "skip", "quit"), reader=key_from("й")
    )
    check("wait_key: й (рус) выходит", result == "quit", f"got {result!r}")

    result = record.wait_key(
        "", allowed=("space",), reader=key_from("ззз ")
    )
    check("wait_key: P×3 не принимает", result == "space", f"got {result!r}")


# --------------------------------------------------------------------- trim
def test_trim() -> None:
    signal = synth(3.0, 0.8, 2.0)
    out = record.trim_speech(signal, noise=0.002)
    check("trim: речь найдена", out is not None)
    if out is None:
        return

    duration = out.size / record.RATE
    check("trim: длительность 2.2 с", abs(duration - 2.2) < 0.06, f"{duration:.2f}")

    head = out[: int(0.45 * record.RATE)].astype(np.float32) / 32768.0
    check("trim: тишина в начале", float(np.abs(head).max()) < 0.02,
          f"{float(np.abs(head).max()):.3f}")

    tail = out[-int(0.45 * record.RATE):].astype(np.float32) / 32768.0
    check("trim: тишина в конце", float(np.abs(tail).max()) < 0.02,
          f"{float(np.abs(tail).max()):.3f}")

    silence = synth(2.0, 0, 0, noise=0.002)
    check("trim: чистая тишина -> None", record.trim_speech(silence, 0.002) is None)

    quiet = synth(3.0, 1.0, 2.0, amp=0.02, noise=0.006)
    out = record.trim_speech(quiet, noise=0.006)
    check("trim: тихая речь найдена",
          out is not None and out.size / record.RATE > 1.0)

    at_start = synth(2.0, 0.0, 1.0)
    out = record.trim_speech(at_start, noise=0.002)
    # 0.5 с искусственного lead-in + 1.0 с речи + 0.5 с хвоста
    check("trim: lead-in добавляется",
          out is not None and abs(out.size / record.RATE - 2.0) < 0.06,
          f"{0.0 if out is None else out.size / record.RATE:.2f}")
    if out is not None:
        head = out[: int(0.45 * record.RATE)].astype(np.float32) / 32768.0
        check("trim: lead-in это тишина", float(np.abs(head).max()) == 0.0,
              f"{float(np.abs(head).max()):.3f}")


# ---------------------------------------------------------------------- wav
def test_wav() -> None:
    path = Path(tempfile.mkdtemp()) / "one_second.wav"
    record.save_wav(path, (np.ones(record.RATE, dtype=np.int16) * 1000))
    with wave.open(str(path), "rb") as f:
        check("wav: моно", f.getnchannels() == 1)
        check("wav: 16 бит", f.getsampwidth() == 2)
        check("wav: 16 кГц", f.getframerate() == 16000)
        check("wav: 1 секунда", f.getnframes() == record.RATE)


# --------------------------------------------------------------------- misc
def test_misc() -> None:
    check("dbfs 1.0 = 0", abs(record.dbfs(1.0)) < 1e-9)
    check("dbfs 0.5 = -6.02", abs(record.dbfs(0.5) + 6.0206) < 0.01)

    check("verdict normal ok", record.verdict("normal", -15) == "в диапазоне")
    check("verdict normal тихо", "тише" in record.verdict("normal", -30))
    check("verdict normal громко", "громче" in record.verdict("normal", -3))


# --------------------------------------------------------------------- план
def test_plan() -> None:
    rows = record.build_plan()
    check("план: 117 дублей", len(rows) == 117)
    check("план: имена уникальны", len({r["file"] for r in rows}) == 117)

    sessions: dict[str, int] = {}
    for row in rows:
        sessions[row["session"]] = sessions.get(row["session"], 0) + 1
    check("план: сессии 33/40/44",
          sessions == {"s1": 33, "s2": 40, "s3": 44}, str(sessions))

    check("план: у команд есть action",
          all(r["action"] for r in rows
              if r["kind"] in ("command", "wake_command")))

    raw = [r for r in rows if r["trim"] == "0"]
    check("план: фон без обрезки", len(raw) == 1 and raw[0]["block"] == "H")


def main() -> int:
    print("review")
    test_review()
    print("keys")
    test_keys()
    print("trim")
    test_trim()
    print("wav")
    test_wav()
    print("misc")
    test_misc()
    print("plan")
    test_plan()

    print()
    if FAILED:
        print(f"Провалено: {len(FAILED)}")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print("Все тесты пройдены.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
