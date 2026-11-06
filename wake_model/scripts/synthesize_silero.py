"""Генерация синтетических позитивов через Silero TTS v5.

Silero даёт 5 русских голосов и SSML: можно менять темп и вставлять паузу
между «хей» и «луна». Модель скачивается torch.hub в кэш пользователя,
поэтому кириллица в пути проекта ей не мешает.

Пример:
    python wake_model/scripts/synthesize_silero.py --count 20 \
        --out wake_model/data/silero_probe
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import wave
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import torch
import yaml
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
PIPER_RATE = 48000


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("data", "features", "checkpoints"):
        cfg["paths"][key] = ROOT / cfg["paths"][key]
    return cfg


def load_model(version: str):
    model, _ = torch.hub.load(
        "snakers4/silero-models", model="silero_tts",
        language="ru", speaker=version, trust_repo=True)
    return model


def ssml_for(variant: str, text: str, break_ms: int) -> tuple[str | None, str | None]:
    """Возвращает (text, ssml_text) для apply_tts."""
    clean = escape(text)
    if variant == "slow":
        return None, f'<speak><prosody rate="slow">{clean}</prosody></speak>'
    if variant == "fast":
        return None, f'<speak><prosody rate="fast">{clean}</prosody></speak>'
    if variant == "break":
        words = text.split()
        if len(words) < 2:
            return text, None
        first = escape(words[0])
        rest = escape(" ".join(words[1:]))
        return None, f'<speak>{first}<break time="{break_ms}ms"/>{rest}</speak>'
    return text, None


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
    silero = cfg["silero"]
    synth = cfg["synthesis"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=cfg["paths"]["data"] / "silero_probe")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--text", default=cfg["phrase"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--speaker", action="append", choices=silero["voices"])
    parser.add_argument("--variant", action="append", choices=silero["variants"])
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for wav in out.glob("*.wav"):
            wav.unlink()
        meta = out / "metadata.csv"
        if meta.exists():
            meta.unlink()

    voices = args.speaker or silero["voices"]
    variants = args.variant or silero["variants"]
    combos = [(voice, variant) for variant in variants for voice in voices]
    rng = random.Random(args.seed)

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = load_model(silero["model"])
    rate = int(cfg["audio"]["sample_rate"])
    rows: list[dict] = []

    for index in range(args.count):
        voice, variant = combos[index % len(combos)]
        break_ms = rng.randint(*silero["break_ms"])
        text, ssml = ssml_for(variant, args.text, break_ms)
        if ssml:
            audio = model.apply_tts(ssml_text=ssml, speaker=voice,
                                    sample_rate=PIPER_RATE)
        else:
            audio = model.apply_tts(text=text, speaker=voice,
                                    sample_rate=PIPER_RATE)
        pcm = resample_poly(audio.numpy().astype(np.float32), rate, PIPER_RATE)
        pcm = np.clip(pcm * 32768.0, -32768, 32767).astype(np.int16)
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
            "break_ms": break_ms if variant == "break" else "",
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
