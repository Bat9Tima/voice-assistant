"""Генерация синтетических позитивов через Silero TTS v5.

Два профиля из config.yaml: `silero` (v5_ru) и `silero_cis` (v5_cis_base).
Silero даёт SSML: можно менять темп и вставлять паузу между «хей» и
«луна». Модель скачивается torch.hub в кэш пользователя, поэтому
кириллица в пути проекта ей не мешает.

Пример:
    python wake_model/scripts/synthesize_silero.py --count 20 \
        --out wake_model/data/stage0/silero_probe
    python wake_model/scripts/synthesize_silero.py --profile cis \
        --count 16 --out wake_model/data/stage0/silero_cis_smoke
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def ssml_for(variant: str, text: str, break_ms: int,
             rate_value: float, pitch_value: float
             ) -> tuple[str | None, str | None]:
    """Возвращает (text, ssml_text) для apply_tts."""
    clean = escape(text)
    inner = clean
    if variant == "break":
        words = text.split()
        if len(words) < 2:
            return text, None
        first = escape(words[0])
        rest = escape(" ".join(words[1:]))
        inner = f'{first}<break time="{break_ms}ms"/>{rest}'
    attrs = []
    if abs(rate_value - 1.0) > 1e-4:
        attrs.append(f'rate="{rate_value * 100:.0f}%"')
    if abs(pitch_value) > 1e-4:
        sign = "+" if pitch_value > 0 else "-"
        attrs.append(f'pitch="{sign}{abs(pitch_value) * 100:.0f}%"')
    if not attrs and variant != "break":
        return text, None
    prosody = f'<prosody {" ".join(attrs)}>' if attrs else ""
    close = "</prosody>" if attrs else ""
    return None, f"<speak>{prosody}{inner}{close}</speak>"


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


def render_clip(model, voice: str, variant: str, text: str, break_ms: int,
                rate_value: float, pitch_value: float, rate_hz: int,
                synth: dict) -> np.ndarray:
    """Синтез одного клипа с обрезкой, нормализацией и паузами."""
    plain, ssml = ssml_for(variant, text, break_ms, rate_value, pitch_value)
    if ssml:
        audio = model.apply_tts(ssml_text=ssml, speaker=voice,
                                sample_rate=PIPER_RATE)
    else:
        audio = model.apply_tts(text=plain, speaker=voice,
                                sample_rate=PIPER_RATE)
    pcm = resample_poly(audio.numpy().astype(np.float32), rate_hz, PIPER_RATE)
    pcm = np.clip(pcm * 32768.0, -32768, 32767).astype(np.int16)
    pcm = trim_speech(pcm, rate_hz, synth["trim_threshold_db"],
                      synth["trim_padding"])
    pcm = normalize(pcm, synth["target_peak_db"])
    return np.concatenate([
        np.zeros(int(rate_hz * synth["lead_silence"]), dtype=np.int16),
        pcm,
        np.zeros(int(rate_hz * synth["tail_silence"]), dtype=np.int16),
    ])


def build_grid(profile: dict, variant: str,
               rng: random.Random) -> list[tuple[float, float, int]]:
    """Детерминированная сетка SSML-параметров для одного варианта."""
    grid = profile["grid"]
    rates = grid["rate"].get(variant) or grid["rate"]["plain"]
    pitches = grid["pitch"]
    breaks = grid.get("break_ms", [0]) if variant == "break" else [0]
    combos = [(rate, pitch, brk) for brk in breaks
              for rate in rates for pitch in pitches]
    rng.shuffle(combos)
    return combos


def existing_hashes(root: Path, skip: Path) -> set[str]:
    """Хеши уже сгенерированных волн, чтобы не плодить байт-копии."""
    seen: set[str] = set()
    for path in root.glob("*/*.wav"):
        if path.parent == skip:
            continue
        seen.add(hashlib.md5(path.read_bytes()).hexdigest())
    return seen


def run_grid(cfg: dict, profile: dict, out: Path, voices: list[str],
             variants: list[str], target: int, max_attempts: int,
             seed: int, text: str) -> int:
    """Добирает уникальные волны по сетке до target на голос."""
    synth = cfg["synthesis"]
    rate_hz = int(cfg["audio"]["sample_rate"])
    seen = existing_hashes(out.parent, out)
    print(f"Известных волн до генерации: {len(seen)}")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = load_model(profile["model"])
    rng = random.Random(seed)
    rows: list[dict] = []

    for voice in voices:
        combos: list[tuple[str, float, float, int]] = []
        for variant in variants:
            for rate_value, pitch_value, break_ms in build_grid(
                    profile, variant, rng):
                combos.append((variant, rate_value, pitch_value, break_ms))
        rng.shuffle(combos)
        written = 0
        attempts = 0
        for variant, rate_value, pitch_value, break_ms in combos:
            if written >= target or attempts >= max_attempts:
                break
            attempts += 1
            pcm = render_clip(model, voice, variant, text, break_ms,
                              rate_value, pitch_value, rate_hz, synth)
            key = hashlib.md5(pcm.tobytes()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            written += 1
            name = f"{voice}_{variant}_{written:04d}.wav"
            write_wav(out / name, pcm, rate_hz)
            peak_db = db(float(np.abs(pcm.astype(np.float32)).max()) / 32768.0)
            rms_db = db(
                float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) / 32768.0
            )
            rows.append({
                "file": name,
                "voice": voice,
                "variant": variant,
                "rate_pct": f"{rate_value * 100:.0f}",
                "pitch_pct": f"{pitch_value * 100:+.1f}",
                "text": text,
                "break_ms": break_ms if variant == "break" else "",
                "duration_s": f"{len(pcm) / rate_hz:.3f}",
                "peak_db": f"{peak_db:.1f}",
                "rms_db": f"{rms_db:.1f}",
            })
        print(f"  {voice}: новых волн {written}, попыток {attempts}")

    fields = ["file", "voice", "variant", "rate_pct", "pitch_pct", "text",
              "break_ms", "duration_s", "peak_db", "rms_db"]
    with open(out / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Готово: {len(rows)} новых волн → {out}")
    return 0


def main() -> int:
    cfg = load_config()
    synth = cfg["synthesis"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["silero", "cis"],
                        default="silero",
                        help="silero = v5_ru, cis = v5_cis_base")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--text", default=cfg["phrase"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--grid", action="store_true",
                        help="добирать уникальные волны по сетке параметров")
    parser.add_argument("--target", type=int, default=None,
                        help="сколько новых волн на голос (grid)")
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="предел попыток синтеза на голос (grid)")
    args, _ = parser.parse_known_args()
    profile = cfg["silero"] if args.profile == "silero" else cfg["silero_cis"]
    allowed_variants = list(profile["variants"])
    for name in profile.get("grid", {}).get("variants", []):
        if name not in allowed_variants:
            allowed_variants.append(name)
    parser.add_argument("--speaker", action="append",
                        choices=profile["voices"])
    parser.add_argument("--variant", action="append",
                        choices=allowed_variants)
    args = parser.parse_args()

    if args.out is not None:
        out: Path = args.out
    elif args.grid:
        folder = "silero_grid" if args.profile == "silero" else "silero_cis_grid"
        out = cfg["paths"]["data"] / "positives_tts" / folder
    else:
        out = (
            cfg["paths"]["data"] / "stage0" /
            ("silero_probe" if args.profile == "silero" else "silero_cis_probe")
        )
    out.mkdir(parents=True, exist_ok=True)
    if args.grid and not args.clean and any(out.glob("*.wav")):
        print(f"В {out} уже есть WAV; для grid-прогона нужен --clean",
              file=sys.stderr)
        return 2
    if args.clean:
        for wav in out.glob("*.wav"):
            wav.unlink()
        meta = out / "metadata.csv"
        if meta.exists():
            meta.unlink()

    voices = args.speaker or profile["voices"]
    grid_cfg = profile.get("grid", {})
    variants = args.variant or (
        grid_cfg.get("variants", profile["variants"])
        if args.grid else profile["variants"]
    )
    if args.grid:
        target = args.target or grid_cfg.get("target", 80)
        max_attempts = args.max_attempts or max(target * 4, 100)
        return run_grid(cfg, profile, out, voices, variants, target,
                        max_attempts, args.seed, args.text)

    combos = [(voice, variant) for variant in variants for voice in voices]
    break_range = profile.get("break_ms", [150, 400])
    pitch_range = profile.get("pitch", [0.0, 0.0])
    rng = random.Random(args.seed)

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = load_model(profile["model"])
    rate = int(cfg["audio"]["sample_rate"])
    rows: list[dict] = []

    for index in range(args.count):
        voice, variant = combos[index % len(combos)]
        break_ms = rng.randint(*break_range)
        rate_value = rng.uniform(*profile["rate"].get(variant, [1.0, 1.0]))
        pitch_value = rng.uniform(*pitch_range)
        pcm = render_clip(model, voice, variant, args.text, break_ms,
                          rate_value, pitch_value, rate, synth)

        name = f"{voice}_{variant}_{index + 1:03d}.wav"
        write_wav(out / name, pcm, rate)
        peak_db = db(float(np.abs(pcm.astype(np.float32)).max()) / 32768.0)
        rms_db = db(float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) / 32768.0)
        rows.append({
            "file": name,
            "voice": voice,
            "variant": variant,
            "rate_pct": f"{rate_value * 100:.0f}",
            "pitch_pct": f"{pitch_value * 100:+.1f}",
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
