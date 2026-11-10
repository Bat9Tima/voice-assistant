"""Генерация клипов Piper для QC и будущих синтетических позитивов.

Piper запускается отдельным процессом: бинарник 2023.11.14-2 не умеет
работать с кириллическими путями к exe/модели, поэтому инструменты
вынесены в untracked-папку VoiceAssistantTools в корне проекта, а корень
проекта должен оставаться ASCII. Путь к готовому WAV может быть любым.

Пример:
    python wake_model/scripts/synthesize_piper.py --count 100
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import random
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    piper = os.environ.get("PIPER_EXE") or cfg["paths"]["piper"]
    voices = os.environ.get("PIPER_VOICES") or cfg["paths"]["voices"]
    # В config.yaml пути к инструментам относительны корня проекта:
    # абсолютный путь с кириллицей ломает piper.exe. Переопределения
    # PIPER_EXE/PIPER_VOICES могут быть абсолютными — их не трогаем.
    piper_path = Path(os.path.expandvars(piper))
    voices_path = Path(os.path.expandvars(voices))
    if not piper_path.is_absolute():
        piper_path = PROJECT / piper_path
    if not voices_path.is_absolute():
        voices_path = PROJECT / voices_path
    cfg["paths"]["piper"] = piper_path
    cfg["paths"]["voices"] = voices_path
    for key in ("data", "features", "checkpoints"):
        cfg["paths"][key] = ROOT / cfg["paths"][key]
    return cfg


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


def run_piper(piper: Path, model: Path, text: str, out: Path,
              length_scale: float, noise_scale: float, noise_w: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(piper),
        "-m", str(model),
        "-f", str(out),
        "-q",
        "--sentence_silence", "0",
        "--length_scale", f"{length_scale:.3f}",
        "--noise_scale", f"{noise_scale:.3f}",
        "--noise_w", f"{noise_w:.3f}",
    ]
    result = subprocess.run(
        cmd, input=text.encode("utf-8"), capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Piper завершился с кодом {result.returncode}: "
            f"{result.stderr.decode('utf-8', 'replace')[-500:]}")


def pause_variant(phrase: str) -> str:
    words = phrase.split()
    if len(words) < 2:
        return phrase
    return f"{words[0]}, {' '.join(words[1:])}"


def voice_short(name: str) -> str:
    return name.removeprefix("ru_RU-").removesuffix("-medium")


def make_html(out: Path, rows: list[dict], phrase: str,
              per_voice: int = 5) -> Path:
    by_voice: dict[str, list[dict]] = {}
    for row in rows:
        by_voice.setdefault(row["voice"], []).append(row)

    picked: list[dict] = []
    for _voice, items in by_voice.items():
        if len(items) <= per_voice:
            chosen = items
        else:
            positions = np.linspace(0, len(items) - 1, per_voice).round().astype(int)
            chosen = [items[i] for i in dict.fromkeys(positions)]
        picked.extend(chosen)

    cards = []
    for row in picked:
        cards.append(
            '<div class="card">'
            f'<div class="title">{html.escape(voice_short(row["voice"]))}'
            f' · {html.escape(row["text"])}</div>'
            f'<audio controls preload="metadata" src="{html.escape(row["file"])}"></audio>'
            f'<div class="meta">length_scale {row["length_scale"]} · '
            f'noise_scale {row["noise_scale"]} · noise_w {row["noise_w"]} · '
            f'{row["duration_s"]} с · пик {row["peak_db"]} dBFS</div>'
            '</div>'
        )

    page = f"""<!doctype html>
<html lang="ru">
<meta charset="utf-8">
<title>Piper QC — «{html.escape(phrase)}»</title>
<style>
  body {{ font-family: Segoe UI, sans-serif; background:#14161a; color:#e8e8e8;
         max-width: 860px; margin: 24px auto; padding: 0 16px; }}
  h1 {{ font-size: 20px; font-weight: 600; }}
  p {{ color:#a8adb5; }}
  .card {{ background:#1e222a; border:1px solid #2c323d; border-radius:10px;
           padding:14px 16px; margin:12px 0; }}
  .title {{ font-weight:600; margin-bottom:8px; }}
  audio {{ width:100%; }}
  .meta {{ color:#8f96a3; font-size:12px; margin-top:6px; }}
</style>
<h1>Piper QC: {len(picked)} клипов «{html.escape(phrase)}»</h1>
<p>Слушать по 5 клипов на голос. Оценить: слышно ли «хей», не глотается ли
«й», естественна ли пауза между словами, нет ли хрипа и обрывов.</p>
{''.join(cards)}
</html>
"""
    path = out / "listen.html"
    path.write_text(page, encoding="utf-8")
    return path


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=cfg["paths"]["data"] / "stage0" / "piper_qc")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true",
                        help="удалить старые WAV и metadata.csv в --out")
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()

    synth = cfg["synthesis"]
    piper = cfg["paths"]["piper"]
    voices_dir = cfg["paths"]["voices"]
    voices: list[str] = list(cfg["voices"])
    phrase: str = cfg["phrase"]

    if not piper.is_file():
        print(f"Нет Piper: {piper}", file=sys.stderr)
        return 2
    if not str(voices_dir).isascii():
        print("Путь к голосам должен быть ASCII: piper.exe не понимает "
              f"кириллицу ({voices_dir})", file=sys.stderr)
        return 2
    if not voices:
        print("В config.yaml пуст список voices", file=sys.stderr)
        return 2

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for wav in out.glob("*.wav"):
            wav.unlink()
        meta = out / "metadata.csv"
        if meta.exists():
            meta.unlink()

    rng = random.Random(args.seed)
    rows: list[dict] = []
    rate = int(cfg["audio"]["sample_rate"])

    for index in range(args.count):
        voice = voices[index % len(voices)]
        model = voices_dir / voice / f"{voice}.onnx"
        if not model.is_file():
            print(f"Нет модели голоса: {model}", file=sys.stderr)
            return 2

        text = pause_variant(phrase) if rng.random() < synth["pause_probability"] else phrase
        length_scale = rng.uniform(*synth["length_scale"])
        noise_scale = rng.uniform(*synth["noise_scale"])
        noise_w = rng.uniform(*synth["noise_w"])

        raw = out / "_raw.wav"
        run_piper(piper, model, text, raw, length_scale, noise_scale, noise_w)
        pcm, raw_rate = read_wav(raw)
        raw.unlink(missing_ok=True)
        if raw_rate != rate:
            pcm = resample_poly(pcm.astype(np.float32), rate, raw_rate)
            pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
        pcm = trim_speech(pcm, rate, synth["trim_threshold_db"],
                          synth["trim_padding"])
        pcm = normalize(pcm, synth["target_peak_db"])
        lead = np.zeros(int(rate * synth["lead_silence"]), dtype=np.int16)
        tail = np.zeros(int(rate * synth["tail_silence"]), dtype=np.int16)
        pcm = np.concatenate([lead, pcm, tail])

        name = f"{voice_short(voice)}_{index // len(voices) + 1:03d}.wav"
        write_wav(out / name, pcm, rate)

        peak_db = db(float(np.abs(pcm.astype(np.float32)).max()) / 32768.0)
        rms_db = db(float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) / 32768.0)
        rows.append({
            "file": name,
            "voice": voice,
            "text": text,
            "length_scale": f"{length_scale:.3f}",
            "noise_scale": f"{noise_scale:.3f}",
            "noise_w": f"{noise_w:.3f}",
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
    if not args.no_html:
        page = make_html(out, rows, phrase)
        print(f"Страница для прослушивания: {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
