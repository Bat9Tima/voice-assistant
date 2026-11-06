"""Генерация синтетических позитивов через Supertonic 3.

Supertonic 3 — локальный ONNX-TTS на 31 язык, включая русский. В открытом
пакете 10 готовых голосов (F1–F5, M1–M5), темп задаётся параметром speed.
Модель скачивается в кэш Hugging Face при первом запуске.

Пример:
    python wake_model/scripts/synthesize_supertonic.py --count 30 \
        --out wake_model/data/supertonic_probe --clean
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import wave
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RATE = 44100


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("data", "features", "checkpoints"):
        cfg["paths"][key] = ROOT / cfg["paths"][key]
    return cfg


def write_wav(path: Path, pcm: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.astype(np.int16).tobytes())


def db(value: float) -> float:
    return 20.0 * float(np.log10(max(value, 1e-9)))


def trim_speech(pcm: np.ndarray, rate: int, threshold_db: float,
                padding: float) -> np.ndarray:
    amplitude = np.abs(pcm.astype(np.float32))
    peak = float(amplitude.max())
    if peak <= 0:
        return pcm
    threshold = peak * 10.0 ** (threshold_db / 20.0)
    voiced = np.flatnonzero(amplitude > threshold)
    if voiced.size == 0:
        return pcm
    pad = int(rate * padding)
    start = max(0, int(voiced[0]) - pad)
    end = min(len(pcm), int(voiced[-1]) + pad + 1)
    return pcm[start:end]


def normalize(pcm: np.ndarray, target_db: float) -> np.ndarray:
    peak = float(np.abs(pcm.astype(np.float32)).max())
    if peak <= 0:
        return pcm
    gain = 10.0 ** (target_db / 20.0) / (peak / 32768.0)
    return np.clip(pcm.astype(np.float32) * gain, -32768, 32767).astype(np.int16)


def main() -> int:
    cfg = load_config()
    tonic = cfg["supertonic"]
    synth = cfg["synthesis"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=cfg["paths"]["data"] / "supertonic_probe")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--text", default=cfg["phrase"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--voice", action="append", choices=tonic["voices"])
    parser.add_argument("--variant", action="append", choices=tonic["variants"])
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for wav in out.glob("*.wav"):
            wav.unlink()
        meta = out / "metadata.csv"
        if meta.exists():
            meta.unlink()

    from supertonic import TTS

    voices = args.voice or tonic["voices"]
    variants = args.variant or tonic["variants"]
    allowed = tonic.get("voice_variants", {})
    combos = [
        (voice, variant)
        for variant in variants
        for voice in voices
        if variant in allowed.get(voice, tonic["variants"])
    ]
    if not combos:
        print("Нет допустимых сочетаний голос/вариант", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    rate = int(cfg["audio"]["sample_rate"])
    tts = TTS(model=tonic["model"], auto_download=True)
    styles = {voice: tts.get_voice_style(voice) for voice in voices}
    rows: list[dict] = []

    for index in range(args.count):
        voice, variant = combos[index % len(combos)]
        speed = rng.uniform(*tonic["speed"][variant])
        audio, _ = tts.synthesize(
            args.text, voice_style=styles[voice], lang="ru", speed=speed,
            total_steps=tonic["total_steps"])
        pcm = np.asarray(audio, dtype=np.float32).reshape(-1) * 32768.0
        pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
        pcm = resample_poly(pcm.astype(np.float32), rate, SOURCE_RATE)
        pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
        pcm = trim_speech(pcm, rate, synth["trim_threshold_db"],
                          synth["trim_padding"])
        pcm = normalize(pcm, synth["target_peak_db"])
        pcm = np.concatenate([
            np.zeros(int(rate * synth["lead_silence"]), dtype=np.int16),
            pcm,
            np.zeros(int(rate * synth["tail_silence"]), dtype=np.int16),
        ])

        name = f"{voice}_{variant}_{index + 1:03d}.wav"
        write_wav(out / name, pcm, rate)
        peak_db = db(float(np.abs(pcm.astype(np.float32)).max()) / 32768.0)
        rms_db = db(float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) / 32768.0)
        rows.append({
            "file": name,
            "voice": voice,
            "variant": variant,
            "text": args.text,
            "speed": f"{speed:.3f}",
            "duration_s": f"{len(pcm) / rate:.3f}",
            "peak_db": f"{peak_db:.1f}",
            "rms_db": f"{rms_db:.1f}",
        })
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{args.count}")

    with open(out / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Готово: {len(rows)} клипов → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
