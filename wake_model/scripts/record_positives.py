"""Запись реальных позитивов «хей луна» для этапа 2.

Инструмент ведёт сессию «диктор × микрофон» по матрице
дистанция × фон × уровень голоса, пишет raw WAV в нативном формате
микрофона, даёт прослушать дубль и принять/переснять/забраковать.
Принятые файлы никогда не перезаписываются; брак остаётся в журнале.

Режимы:
    --list-devices          показать входы и частоты
    --setup                 новая сессия (нужны --speaker и --mic)
    --resume                продолжить последнюю незавершённую сессию
    --review [ПУТЬ]         прослушать manifest и забраковать лишнее
    --finalize              raw → 16 кГц, обрезка, QC, manifest, страница

Пример:
    python wake_model/scripts/record_positives.py --list-devices
    python wake_model/scripts/record_positives.py --setup --speaker anna --mic laptop
    python wake_model/scripts/record_positives.py --finalize
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import queue
import random
import re
import sys
import time
import wave
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from app.text import normalize  # noqa: E402

try:
    import msvcrt
except ImportError:  # pragma: no cover - Windows only
    msvcrt = None

RATE = 16000
HEY = {"хей", "хэй", "эй", "хэ", "хе", "ей", "ай", "а"}

FIELDS = [
    "file", "speaker", "mic", "distance_m", "condition", "level", "phrase",
    "take", "duration_s", "peak_db", "rms_db", "clipping", "noise_floor_db",
    "has_luna", "has_hey", "exact", "text", "timestamp", "notes", "kept",
]

QC_FIELDS = [
    "file", "speaker", "mic", "distance_m", "condition", "level", "phrase",
    "take", "duration_s", "speech_s", "peak_db", "rms_db", "clipping",
    "noise_floor_db", "has_luna", "has_hey", "exact", "text", "timestamp",
    "notes", "kept", "reason", "warnings",
]

DEFAULTS = {
    "phrase": "хей луна",
    "extra_phrase": "эй луна",
    "extra_takes": 2,
    "window_s": 3.0,
    "countdown_s": 2.0,
    "calibration_s": 5.0,
    "takes_per_cell": 3,
    "distances_m": [0.5, 1.0, 1.5, 2.0],
    "conditions": ["quiet", "music", "tv"],
    "levels": ["norm", "sub"],
    "trim_threshold_db": -40.0,
    "trim_padding": 0.03,
    "lead_silence": 0.30,
    "tail_silence": 0.30,
    "min_speech_s": 0.70,
    "max_speech_s": 2.50,
    "target_min": 60,
    "target_max": 100,
}

CONDITION_RU = {"quiet": "тишина", "music": "музыка", "tv": "телевизор"}
LEVEL_RU = {"norm": "обычный", "sub": "тихий"}


# --------------------------------------------------------------------------
# Конфигурация и план
# --------------------------------------------------------------------------

def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def recording_config(cfg: dict) -> dict:
    result = dict(DEFAULTS)
    result.update(cfg.get("recording") or {})
    return result


def data_root(cfg: dict) -> Path:
    return ROOT / cfg["paths"]["data"] / "positives_real"


def token(value: str, name: str) -> str:
    """Короткий ASCII-id для имени файла и manifest."""
    clean = re.sub(r"[^a-z0-9_-]+", "-", str(value).strip().lower())
    clean = clean.strip("-_")
    if not clean:
        raise ValueError(f"Пустой {name}: нужно латинское имя (например, anna)")
    return clean


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_plan(speaker: str, mic: str, rec: dict, takes_per_cell: int,
               with_extra: bool, seed: int) -> list[dict]:
    """Детерминированный план сессии: фон — внешний блок."""
    rng = random.Random(f"{speaker}|{mic}|{seed}")
    plan: list[dict] = []
    for condition in rec["conditions"]:
        cells = [(float(d), level) for d in rec["distances_m"]
                 for level in rec["levels"]]
        rng.shuffle(cells)
        for distance, level in cells:
            code = f"d{int(round(distance * 10)):02d}"
            plan.append({
                "id": f"{condition}_{code}_{level}",
                "phrase": rec["phrase"],
                "distance_m": distance,
                "condition": condition,
                "level": level,
                "target": int(takes_per_cell),
                "accepted": 0,
                "attempts": 0,
                "status": "pending",
            })
    if with_extra:
        for distance in rec["distances_m"]:
            distance = float(distance)
            code = f"d{int(round(distance * 10)):02d}"
            plan.append({
                "id": f"ey_quiet_{code}_norm",
                "phrase": rec["extra_phrase"],
                "distance_m": distance,
                "condition": "quiet",
                "level": "norm",
                "target": int(rec["extra_takes"]),
                "accepted": 0,
                "attempts": 0,
                "status": "pending",
            })
    return plan


def make_take_name(speaker: str, mic: str, cell: dict, take: int,
                   main_phrase: str) -> str:
    distance = f"{int(round(float(cell['distance_m']) * 10)):02d}"
    prefix = "" if cell["phrase"] == main_phrase else "ey_"
    return (f"{speaker}_{mic}_{prefix}d{distance}_{cell['condition']}_"
            f"{cell['level']}_{take:03d}.wav")


def next_take(root: Path, session: dict, cell: dict) -> tuple[int, str]:
    used = [int(t["take"]) for t in session["takes"]
            if t["cell"] == cell["id"]]
    take = max(used, default=0) + 1
    while True:
        name = make_take_name(session["speaker"], session["mic"], cell,
                              take, session["settings"]["phrase"])
        if not (root / "raw" / name).exists():
            return take, name
        take += 1


def total_accepted(session: dict) -> int:
    return sum(1 for take in session["takes"] if take["status"] == "accepted")


def refresh_cells(session: dict) -> None:
    """Пересчитывает прогресс ячеек по журналу дублей."""
    takes = session.get("takes", [])
    for cell in session["plan"]:
        accepted = sum(1 for take in takes
                       if take["cell"] == cell["id"]
                       and take["status"] == "accepted")
        attempts = sum(1 for take in takes if take["cell"] == cell["id"])
        cell["accepted"] = accepted
        cell["attempts"] = attempts
        cell["status"] = "done" if accepted >= cell["target"] else "pending"


def next_cell(session: dict) -> dict | None:
    for cell in session["plan"]:
        if cell["accepted"] < cell["target"]:
            return cell
    return None


# --------------------------------------------------------------------------
# Журнал сессии
# --------------------------------------------------------------------------

def sessions_dir(root: Path) -> Path:
    return root / "sessions"


def session_file(root: Path, speaker: str, mic: str) -> Path:
    return sessions_dir(root) / f"{speaker}_{mic}.json"


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("plan"), list):
        refresh_cells(data)
    return data


def save_session(path: Path, session: dict) -> None:
    session["updated"] = iso_now()
    save_json(path, session)


def archive_session(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.stem}.{stamp}.bak.json")
    counter = 1
    while target.exists():
        target = path.with_name(f"{path.stem}.{stamp}-{counter}.bak.json")
        counter += 1
    path.replace(target)
    return target


def find_latest_session(root: Path) -> Path | None:
    """Последняя активная сессия; архивные `.bak.json` не трогаем."""
    candidates = [p for p in sessions_dir(root).glob("*.json")
                  if "." not in p.stem]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def accepted_hashes(root: Path, current: dict | None = None) -> set[str]:
    """Все принятые PCM-хеши, включая архивные журналы."""
    hashes: set[str] = set()

    def collect(session: dict) -> None:
        for take in session.get("takes", []):
            if take.get("status") == "accepted" and take.get("pcm_md5"):
                hashes.add(str(take["pcm_md5"]))

    if current is not None:
        collect(current)
    for path in sessions_dir(root).glob("*.json"):
        try:
            collect(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return hashes


def register_take(session: dict, cell: dict, take: int, rel_file: str,
                  status: str, metrics: dict, notes: str = "",
                  warnings: list[str] | None = None) -> None:
    session["takes"].append({
        "cell": cell["id"],
        "take": take,
        "file": rel_file,
        "status": status,
        "phrase": cell["phrase"],
        "distance_m": cell["distance_m"],
        "condition": cell["condition"],
        "level": cell["level"],
        "duration_s": round(float(metrics["duration_s"]), 3),
        "speech_s": round(float(metrics["speech_s"]), 3),
        "peak_db": round(float(metrics["peak_db"]), 1),
        "rms_db": round(float(metrics["rms_db"]), 1),
        "clipping": int(metrics["clipped"]),
        "pcm_md5": metrics["pcm_md5"],
        "noise_floor_db": session.get("calibration", {}).get("noise_floor_db"),
        "timestamp": iso_now(),
        "notes": notes,
        "warnings": warnings or [],
    })
    refresh_cells(session)


# --------------------------------------------------------------------------
# WAV, метрики, обработка
# --------------------------------------------------------------------------

def db(value: float) -> float:
    return 20.0 * float(np.log10(max(value, 1e-9)))


def write_wav(path: Path, pcm: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.astype(np.int16).tobytes())


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


def resample_audio(pcm: np.ndarray, source: int, target: int) -> np.ndarray:
    if source == target or pcm.size == 0:
        return pcm.astype(np.int16)
    from scipy.signal import resample_poly

    divisor = math.gcd(int(source), int(target))
    up = int(target) // divisor
    down = int(source) // divisor
    result = resample_poly(pcm.astype(np.float32), up, down)
    return np.clip(np.round(result), -32768, 32767).astype(np.int16)


def trim_speech(pcm: np.ndarray, rate: int, threshold_db: float,
                padding: float) -> np.ndarray:
    """Обрезка по энергии: та же логика, что в synthesize_silero."""
    amplitude = np.abs(pcm.astype(np.float32))
    peak = float(amplitude.max()) if amplitude.size else 0.0
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


def pad_speech(pcm: np.ndarray, rate: int, lead: float,
               tail: float) -> np.ndarray:
    return np.concatenate([
        np.zeros(int(rate * lead), dtype=np.int16),
        pcm.astype(np.int16),
        np.zeros(int(rate * tail), dtype=np.int16),
    ])


def pcm_metrics(pcm: np.ndarray, rate: int, threshold_db: float,
                padding: float) -> dict:
    if pcm.size == 0:
        return {
            "duration_s": 0.0, "speech_s": 0.0, "peak_db": -180.0,
            "rms_db": -180.0, "clipped": 0, "pcm_md5": "",
        }
    samples = pcm.astype(np.float32) / 32768.0
    peak = float(np.abs(samples).max())
    rms = float(np.sqrt(np.mean(samples * samples)))
    speech = trim_speech(pcm, rate, threshold_db, padding)
    return {
        "duration_s": len(pcm) / rate,
        "speech_s": len(speech) / rate,
        "peak_db": db(peak),
        "rms_db": db(rms),
        "clipped": int(np.count_nonzero(np.abs(pcm.astype(np.int32)) >= 32767)),
        "pcm_md5": hashlib.md5(pcm.astype(np.int16).tobytes()).hexdigest(),
    }


def classify(text: str) -> tuple[bool, bool, bool]:
    tokens = normalize(text).split()
    has_luna = any("лун" in token for token in tokens)
    has_hey = any(
        token in HEY or token.startswith(("хей", "хэй")) for token in tokens)
    return has_luna, has_hey, tokens == ["хей", "луна"]


def evaluate_take(metrics: dict, text: str | None, duplicate: bool,
                  rec: dict) -> list[str]:
    warnings: list[str] = []
    if metrics["peak_db"] < -18.0:
        warnings.append(f"тихо: пик {metrics['peak_db']:.1f} dBFS")
    if metrics["peak_db"] > -1.0 or metrics["clipped"]:
        warnings.append(
            f"громко/клиппинг: пик {metrics['peak_db']:.1f} dBFS, "
            f"{metrics['clipped']} отсчётов")
    if not (rec["min_speech_s"] <= metrics["speech_s"] <= rec["max_speech_s"]):
        warnings.append(
            f"длительность речи {metrics['speech_s']:.2f} с "
            f"(норма {rec['min_speech_s']:.1f}–{rec['max_speech_s']:.1f})")
    if duplicate:
        warnings.append("точный повтор уже принятого дубля")
    if text is not None:
        has_luna, has_hey, _ = classify(text)
        if not has_luna:
            warnings.append(f"ASR не слышит «луна»: «{text or '—'}»")
        elif not has_hey:
            warnings.append(f"ASR не слышит «хей»: «{text}»")
    return warnings


# --------------------------------------------------------------------------
# Аудио: устройства, запись, плейбек, клавиши
# --------------------------------------------------------------------------

def list_input_devices() -> list[dict]:
    import sounddevice as sd

    try:
        default_index = int(sd.default.device[0])
    except Exception:
        default_index = -1
    result = []
    for index, info in enumerate(sd.query_devices()):
        channels = int(info.get("max_input_channels") or 0)
        if channels <= 0:
            continue
        result.append({
            "index": index,
            "name": str(info.get("name") or ""),
            "channels": channels,
            "rate": int(round(float(info.get("default_samplerate") or 0))),
            "default": index == default_index,
        })
    return result


def resolve_device(value: str | int | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int) or str(value).isdigit():
        return int(value)
    devices = list_input_devices()
    needle = str(value).strip().lower()
    matches = [d for d in devices if needle in d["name"].lower()]
    if len(matches) == 1:
        return int(matches[0]["index"])
    if not matches:
        raise ValueError(f"Микрофон не найден: {value}")
    names = ", ".join(f"{d['index']}: {d['name']}" for d in matches)
    raise ValueError(f"Несколько микрофонов подходят: {names}")


def beep(freq: int = 880, ms: int = 120) -> None:
    try:
        import winsound
        winsound.Beep(int(freq), int(ms))
    except Exception:
        print("\a", end="", flush=True)


def capture_take(device: int | None, seconds: float, countdown: float,
                 rate: int | None = None, announce: bool = True
                 ) -> tuple[np.ndarray, int]:
    """Пишет ровно `seconds` секунд; поток открыт до сигнала старта."""
    import sounddevice as sd

    info = sd.query_devices(device, "input")
    native = rate or int(round(float(info.get("default_samplerate") or 48000)))
    max_channels = max(1, int(info.get("max_input_channels") or 1))
    rates = []
    for value in (native, RATE, 48000, 44100):
        value = int(value)
        if value > 0 and value not in rates:
            rates.append(value)
    channels_list = [1]
    if max_channels >= 2:
        channels_list.append(2)

    errors = []
    for sample_rate in rates:
        for channels in channels_list:
            audio: queue.Queue[bytes] = queue.Queue()

            def callback(indata, frames, time_info, status,
                         sink: queue.Queue[bytes] = audio) -> None:
                sink.put(bytes(indata))

            stream = None
            try:
                stream = sd.RawInputStream(
                    samplerate=sample_rate,
                    blocksize=max(1, int(sample_rate * 0.05)),
                    device=device,
                    dtype="int16",
                    channels=channels,
                    callback=callback,
                )
                stream.start()
                time.sleep(0.15)
                while not audio.empty():
                    audio.get_nowait()
                if announce and countdown > 0:
                    for remaining in range(int(math.ceil(countdown)), 0, -1):
                        print(f"  {remaining}…", flush=True)
                        beep(660, 90)
                        time.sleep(1.0)
                if announce:
                    beep(990, 160)
                    # Маркер старта не должен попасть в запись: даём
                    # динамику дозвучать и вычищаем очередь.
                    time.sleep(0.12)
                while not audio.empty():
                    audio.get_nowait()

                needed = max(1, int(seconds * sample_rate))
                chunks: list[bytes] = []
                got = 0
                deadline = time.monotonic() + seconds + 1.0
                while got < needed and time.monotonic() < deadline:
                    try:
                        chunk = audio.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    chunks.append(chunk)
                    got += len(chunk) // 2
                pcm = np.frombuffer(b"".join(chunks), dtype=np.int16)
                if pcm.size == 0:
                    raise RuntimeError("не пришло ни одного отсчёта")
                if channels > 1:
                    usable = pcm.size - pcm.size % channels
                    pcm = (pcm[:usable].reshape(-1, channels)
                           .mean(axis=1).astype(np.int16))
                return pcm[:needed].copy(), sample_rate
            except Exception as exc:
                errors.append(
                    f"{sample_rate} Гц / {channels} кан.: {exc}")
            finally:
                if stream is not None:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:
                        pass
    raise RuntimeError("Не удалось открыть микрофон:\n  " + "\n  ".join(errors))


def play_wav(path: Path) -> None:
    if sys.platform != "win32":
        print("(плейбек доступен только на Windows)")
        return
    try:
        import winsound
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
    except Exception as exc:
        print(f"(не удалось проиграть: {exc})")


def drain_keys() -> None:
    if msvcrt is None:
        return
    while msvcrt.kbhit():
        msvcrt.getwch()


def read_key() -> str:
    if msvcrt is not None:
        char = msvcrt.getwch()
        if char in ("\x00", "\xe0"):
            msvcrt.getwch()
            return ""
        return char.lower()
    return input("> ").strip().lower()[:1]


def ask_comment() -> str:
    try:
        return input("Комментарий к браку: ").strip()
    except EOFError:
        return ""


def review_take(path: Path, metrics: dict, warnings: list[str]
                ) -> tuple[str, str]:
    """Плейбек и решение: (accepted|rejected|quit, комментарий)."""
    while True:
        play_wav(path)
        print(f"  пик {metrics['peak_db']:.1f} dBFS, "
              f"речь {metrics['speech_s']:.2f} с")
        if warnings:
            for warning in warnings:
                print(f"  ! {warning}")
        print("  [Пробел] принять  [R] переснять  [X] брак  "
              "[P] ещё раз  [Q] выход")
        drain_keys()
        key = read_key()
        if key in (" ", "\r", "\n"):
            return "accepted", ""
        if key in ("r", "к"):
            return "rejected", "переснято"
        if key in ("x", "ч"):
            return "rejected", ask_comment() or "брак"
        if key in ("p", "з"):
            continue
        if key in ("q", "й"):
            return "quit", "выход без решения"


# --------------------------------------------------------------------------
# Интерактивная сессия
# --------------------------------------------------------------------------

class AsrHints:
    """Ленивая GigaAM-подсказка; сбой ASR не мешает записи."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._transcriber = None
        self._failed = False

    def text(self, pcm: np.ndarray, rate: int = RATE) -> str | None:
        if not self.enabled or self._failed:
            return None
        if rate != RATE:
            try:
                pcm = resample_audio(pcm, rate, RATE)
            except Exception as exc:
                print(f"ASR-подсказки отключены: {exc}")
                self._failed = True
                return None
        if self._transcriber is None:
            try:
                from app.config import load_config as load_app_config
                from app.stt import create_transcriber

                print("Загружаю GigaAM для подсказок (один раз)…")
                self._transcriber = create_transcriber(load_app_config())
                self._transcriber.load()
            except Exception as exc:
                print(f"ASR-подсказки отключены: {exc}")
                self._failed = True
                return None
        try:
            return self._transcriber.transcribe(
                pcm.astype(np.int16).tobytes())
        except Exception as exc:
            print(f"ASR-подсказка не сработала: {exc}")
            return None


def level_hint(metrics: dict) -> str:
    """Подсказка по уровню пилота: −18…−3 dBFS — норма."""
    if metrics["clipped"]:
        return "клиппинг: отодвиньтесь от микрофона или уменьшите усиление"
    if metrics["peak_db"] < -18.0:
        return "тихо: ближе к микрофону или громче"
    if metrics["peak_db"] > -3.0:
        return "громко: дальше от микрофона или тише"
    return "уровень в норме"


def run_pilot(root: Path, speaker: str, mic: str, device: int | None,
              rec: dict) -> None:
    """Пробный дубль для проверки уровня; в manifest не попадает."""
    path = root / "pilot" / f"{speaker}_{mic}_pilot.wav"
    while True:
        print()
        print("Пилот: скажите «хей луна» в рабочей позе.")
        try:
            pcm, rate = capture_take(
                device, float(rec["window_s"]), float(rec["countdown_s"]))
        except Exception as exc:
            print(f"Пилот не записался: {exc}")
            return
        write_wav(path, pcm, rate)
        metrics = pcm_metrics(
            pcm, rate, rec["trim_threshold_db"], rec["trim_padding"])
        print(f"  пик {metrics['peak_db']:.1f} dBFS, "
              f"RMS {metrics['rms_db']:.1f} dBFS, "
              f"речь {metrics['speech_s']:.2f} с, {rate} Гц")
        print(f"  {level_hint(metrics)}")
        while True:
            print("  [Пробел] дальше  [R] записать ещё  [P] прослушать")
            drain_keys()
            key = read_key()
            if key in (" ", "\r", "\n"):
                return
            if key in ("p", "з"):
                play_wav(path)
            elif key in ("r", "к"):
                break


def print_cell(session: dict, cell: dict, index: int, total: int) -> None:
    done = total_accepted(session)
    phrase = cell["phrase"]
    distance = f"{cell['distance_m']:g}"
    condition = CONDITION_RU.get(cell["condition"], cell["condition"])
    level = LEVEL_RU.get(cell["level"], cell["level"])
    print()
    print(f"Ячейка {index}/{total} · {phrase} · {distance} м · "
          f"{condition} · {level}")
    print(f"  принято {cell['accepted']}/{cell['target']} · "
          f"всего в сессии {done}")


def run_session(session: dict, path: Path, root: Path) -> int:
    settings = session["settings"]
    try:
        device = resolve_device(session.get("device"))
    except ValueError as exc:
        print(f"Микрофон: {exc}", file=sys.stderr)
        return 2
    print()
    print(f"Сессия: {session['speaker']} × {session['mic']}")
    if device is None:
        print("Микрофон: по умолчанию")
    else:
        info = None
        try:
            devices = list_input_devices()
        except Exception:
            devices = []
        for item in devices:
            if item["index"] == device:
                info = item
                break
        print(f"Микрофон: {info['name'] if info else device}")
    calibration = session.get("calibration") or {}
    if calibration.get("noise_floor_db") is not None:
        rate = calibration.get("sample_rate")
        suffix = f" · запись {rate} Гц" if rate else ""
        print(f"Фон: {calibration['noise_floor_db']:.1f} dBFS{suffix}")
    print("Остановите ассистента, если он занимает микрофон.")
    print("Дубль можно принять пробелом/Enter; принятые файлы не "
          "перезаписываются.")

    asr = AsrHints(bool(session.get("asr_hints")))
    known_hashes = accepted_hashes(root, session)
    while True:
        if session.get("limit") and total_accepted(session) >= session["limit"]:
            session["status"] = "paused"
            save_session(path, session)
            print(f"Лимит {session['limit']} принятых дублей достигнут.")
            return 0
        cell = next_cell(session)
        if cell is None:
            session["status"] = "complete"
            save_session(path, session)
            count = total_accepted(session)
            low = int(settings.get("target_min") or 0)
            high = int(settings.get("target_max") or 0)
            target = f" (цель {low}–{high})" if low or high else ""
            print(f"План выполнен: {count} дублей{target}.")
            return 0

        index = sum(1 for item in session["plan"]
                    if item["accepted"] >= item["target"]) + 1
        print_cell(session, cell, index, len(session["plan"]))
        print("  [Пробел] записать  [Q] выход")
        drain_keys()
        key = read_key()
        if key in ("q", "й"):
            session["status"] = "paused"
            save_session(path, session)
            print("Сессия приостановлена, продолжить: --resume")
            return 0
        if key not in (" ", "\r", "\n"):
            continue

        take, name = next_take(root, session, cell)
        print(f"  Запись {settings['window_s']:.1f} с…")
        try:
            pcm, rate = capture_take(
                device, float(settings["window_s"]),
                float(settings["countdown_s"]))
        except Exception as exc:
            print(f"Ошибка записи: {exc}")
            continue

        rel_file = f"raw/{name}"
        raw_path = root / rel_file
        write_wav(raw_path, pcm, rate)
        metrics = pcm_metrics(
            pcm, rate, settings["trim_threshold_db"],
            settings["trim_padding"])
        text = asr.text(pcm, rate) if asr.enabled else None
        duplicate = metrics["pcm_md5"] in known_hashes
        warnings = evaluate_take(metrics, text, duplicate, settings)

        decision, note = review_take(raw_path, metrics, warnings)
        if decision == "quit":
            register_take(session, cell, take, rel_file, "rejected",
                          metrics, note, warnings)
            session["status"] = "paused"
            save_session(path, session)
            print("Сессия приостановлена, продолжить: --resume")
            return 0
        register_take(session, cell, take, rel_file, decision, metrics,
                      note, warnings)
        if decision == "accepted":
            known_hashes.add(metrics["pcm_md5"])
        save_session(path, session)
        if decision == "accepted":
            print("  Принято.")
        else:
            print(f"  Брак: {note or 'переснято'}")


def start_session(args: argparse.Namespace, cfg: dict, root: Path) -> int:
    rec = recording_config(cfg)
    try:
        speaker = token(args.speaker, "--speaker")
        mic = token(args.mic, "--mic")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    path = session_file(root, speaker, mic)
    if path.exists():
        if args.fresh:
            archived = archive_session(path)
            print(f"Старая сессия сохранена: {archived.name}")
        elif args.resume:
            try:
                session = load_json(path)
            except Exception as exc:
                print(f"Не читается журнал {path}: {exc}", file=sys.stderr)
                return 2
            if args.device:
                session["device"] = args.device
                save_session(path, session)
            print(f"Продолжаю сессию {path.name}")
            return run_session(session, path, root)
        else:
            print(f"Сессия уже существует: {path}", file=sys.stderr)
            print("Продолжить: --resume; начать заново: --fresh",
                  file=sys.stderr)
            return 2
    else:
        if args.resume:
            print(f"Нет сессии {path}", file=sys.stderr)
            return 2

    takes_per_cell = int(args.takes_per_cell or rec["takes_per_cell"])
    if takes_per_cell < 1:
        print("--takes-per-cell должен быть ≥ 1", file=sys.stderr)
        return 2
    if int(args.limit or 0) < 0:
        print("--limit должен быть ≥ 0", file=sys.stderr)
        return 2
    device_arg = args.device
    try:
        device = resolve_device(device_arg)
    except ValueError as exc:
        print(f"Микрофон: {exc}", file=sys.stderr)
        return 2
    calibration = {}
    if not args.skip_calibration:
        print(f"Калибровка фона: {rec['calibration_s']:.0f} с тишины…")
        try:
            pcm, rate = capture_take(
                device, float(rec["calibration_s"]), 0.0, announce=False)
        except Exception as exc:
            print(f"Калибровка не удалась: {exc}", file=sys.stderr)
            return 2
        metrics = pcm_metrics(
            pcm, rate, rec["trim_threshold_db"], rec["trim_padding"])
        calibration = {
            "seconds": float(rec["calibration_s"]),
            "noise_floor_db": round(float(metrics["rms_db"]), 1),
            "sample_rate": rate,
        }
        print(f"Фон: {calibration['noise_floor_db']:.1f} dBFS "
              f"· частота {rate} Гц")

    if not args.skip_pilot:
        run_pilot(root, speaker, mic, device, rec)

    settings = dict(rec)
    settings["takes_per_cell"] = takes_per_cell
    session = {
        "version": 1,
        "speaker": speaker,
        "mic": mic,
        "device": device_arg,
        "created": iso_now(),
        "updated": iso_now(),
        "status": "active",
        "settings": settings,
        "calibration": calibration,
        "asr_hints": bool(args.asr_hints),
        "limit": int(args.limit or 0),
        "plan": build_plan(speaker, mic, settings, takes_per_cell,
                           bool(args.with_extra), int(args.seed)),
        "takes": [],
    }
    save_session(path, session)
    print(f"Сессия: {path}")
    return run_session(session, path, root)


def resume_latest(root: Path, explicit: Path | None,
                  device: str | None) -> int:
    path = explicit or find_latest_session(root)
    if path is None:
        print(f"Нет незавершённых сессий в {sessions_dir(root)}",
              file=sys.stderr)
        return 2
    if not path.exists():
        print(f"Нет файла сессии: {path}", file=sys.stderr)
        return 2
    try:
        session = load_json(path)
    except Exception as exc:
        print(f"Не читается журнал {path}: {exc}", file=sys.stderr)
        return 2
    if device:
        session["device"] = device
    if session.get("status") == "complete" and next_cell(session) is None:
        print("Сессия уже завершена.")
        return 0
    session["status"] = "active"
    save_session(path, session)
    return run_session(session, path, root)


# --------------------------------------------------------------------------
# Финализация: raw → processed, QC, manifest, страница
# --------------------------------------------------------------------------

def read_exclusions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip().replace("\\", "/")
        if value:
            result.add(value)
    return result


def append_exclusion(path: Path, rel_file: str, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Забраковано на слух: пути от positives_real, "
            "один на строку.\n", encoding="utf-8")
    needs_newline = False
    if path.exists() and path.stat().st_size:
        needs_newline = not path.read_text(
            encoding="utf-8").endswith("\n")
    line = rel_file.replace("\\", "/")
    if note:
        line += "  # " + " ".join(note.split())
    with open(path, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(line + "\n")


def load_sessions(root: Path) -> list[tuple[Path, dict]]:
    """Читает активные и архивные журналы: данные не теряются при --fresh."""
    result = []
    paths = sorted(sessions_dir(root).glob("*.json"),
                   key=lambda p: p.stat().st_mtime)
    for path in paths:
        try:
            result.append((path, load_json(path)))
        except Exception as exc:
            print(f"Пропускаю битый журнал {path.name}: {exc}")
    return result


def make_asr(no_stt: bool):
    if no_stt:
        return None
    try:
        from app.config import load_config as load_app_config
        from app.stt import create_transcriber

        print("Загружаю GigaAM для QC…")
        transcriber = create_transcriber(load_app_config())
        transcriber.load()
        return transcriber
    except Exception as exc:
        print(f"ASR недоступен, QC без распознавания: {exc}")
        return None


def finalize(root: Path, no_stt: bool) -> int:
    sessions = load_sessions(root)
    if not sessions:
        print(f"Нет журналов сессий в {sessions_dir(root)}",
              file=sys.stderr)
        return 2
    rec = recording_config(load_config())
    excluded = read_exclusions(root / "exclude.txt")
    transcriber = make_asr(no_stt)

    manifest_rows: list[dict] = []
    qc_rows: list[dict] = []
    seen_hashes: set[str] = set()
    total = 0
    skipped = 0
    for session_path, session in sessions:
        settings = dict(rec)
        settings.update(session.get("settings") or {})
        noise = session.get("calibration", {}).get("noise_floor_db")
        for take in session.get("takes", []):
            if take.get("status") != "accepted":
                skipped += 1
                continue
            total += 1
            raw_path = root / take["file"]
            if not raw_path.exists():
                print(f"Нет raw-файла: {take['file']}")
                continue
            try:
                pcm, rate = read_wav(raw_path)
            except Exception as exc:
                print(f"Не читается {take['file']}: {exc}")
                continue
            pcm = resample_audio(pcm, rate, RATE)
            speech = trim_speech(
                pcm, RATE, settings["trim_threshold_db"],
                settings["trim_padding"])
            processed = pad_speech(
                speech, RATE, settings["lead_silence"],
                settings["tail_silence"])
            rel_file = f"processed/{session_path.stem}/{raw_path.name}"
            out_path = root / rel_file
            write_wav(out_path, processed, RATE)
            metrics = pcm_metrics(
                processed, RATE, settings["trim_threshold_db"],
                settings["trim_padding"])
            text = ""
            has_luna = has_hey = exact = 0
            if transcriber is not None:
                try:
                    text = transcriber.transcribe(
                        processed.astype(np.int16).tobytes())
                    has_luna, has_hey, exact = classify(text)
                except Exception as exc:
                    print(f"ASR не распознал {rel_file}: {exc}")
            reason = []
            if metrics["pcm_md5"] in seen_hashes:
                reason.append("дубль")
            seen_hashes.add(metrics["pcm_md5"])
            if (rel_file in excluded or take["file"] in excluded
                    or raw_path.name in excluded):
                reason.append("исключено вручную")
            warnings = evaluate_take(
                metrics, text or None, "дубль" in reason, settings)
            kept = not reason
            row = {
                "file": rel_file,
                "speaker": session["speaker"],
                "mic": session["mic"],
                "distance_m": f"{float(take['distance_m']):g}",
                "condition": take["condition"],
                "level": take["level"],
                "phrase": take.get("phrase") or settings["phrase"],
                "take": int(take["take"]),
                "duration_s": f"{metrics['duration_s']:.3f}",
                "peak_db": f"{metrics['peak_db']:.1f}",
                "rms_db": f"{metrics['rms_db']:.1f}",
                "clipping": int(metrics["clipped"]),
                "noise_floor_db": (
                    f"{float(noise):.1f}" if noise is not None else ""),
                "has_luna": int(has_luna),
                "has_hey": int(has_hey),
                "exact": int(exact),
                "text": text,
                "timestamp": take.get("timestamp", ""),
                "notes": take.get("notes", ""),
                "kept": 1,
            }
            qc = dict(row)
            qc["speech_s"] = f"{metrics['speech_s']:.3f}"
            qc["kept"] = int(kept)
            qc["reason"] = ",".join(reason)
            qc["warnings"] = "; ".join(warnings)
            qc_rows.append(qc)
            if kept:
                manifest_rows.append(row)

    manifest_rows.sort(key=lambda r: (
        r["speaker"], r["mic"], r["phrase"], r["condition"],
        float(r["distance_m"]), r["level"], r["take"]))
    write_manifest(root / "manifest.csv", manifest_rows)
    write_csv(root / "qc_files.csv", QC_FIELDS, qc_rows)
    write_report(root, sessions, manifest_rows, qc_rows, total, skipped)
    write_listen_page(root, manifest_rows)
    print()
    print(f"Принято дублей: {total}, в manifest: {len(manifest_rows)}, "
          f"брак/пересдачи: {skipped}")
    print(f"Manifest: {root / 'manifest.csv'}")
    print(f"Страница: {root / 'listen_positives.html'}")
    return 0


def write_manifest(path: Path, rows: list[dict]) -> None:
    write_csv(path, FIELDS, rows)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(root: Path, sessions: list[tuple[Path, dict]],
                 manifest: list[dict], qc_rows: list[dict], total: int,
                 skipped: int) -> None:
    lines = [
        "Реальные позитивы — QC",
        f"  сессий: {len(sessions)}",
        f"  принято: {total}, в manifest: {len(manifest)}, "
        f"пересдач/брака: {skipped}",
    ]
    if manifest:
        durations = [float(row["duration_s"]) for row in manifest]
        peaks = [float(row["peak_db"]) for row in manifest]
        exact = sum(int(row["exact"]) for row in manifest)
        has_asr = any((row.get("text") or "").strip() for row in manifest)
        lines += [
            f"  длительность: {min(durations):.2f}–{max(durations):.2f} с",
            f"  пик: {min(peaks):.1f}…{max(peaks):.1f} dBFS",
        ]
        if has_asr:
            lines.append(
                f"  GigaAM точно «хей луна»: {exact}/{len(manifest)}")
        else:
            lines.append("  GigaAM: отключён (--no-stt)")
        lines += ["", "По сессиям"]
        for (path, session) in sessions:
            prefix = f"processed/{path.stem}/"
            count = sum(1 for row in manifest
                        if row["file"].startswith(prefix))
            low = int(session.get("settings", {}).get("target_min") or 0)
            complete = session.get("status") == "complete"
            mark = (f" (цель ≥{low})"
                    if complete and low and count < low else "")
            lines.append(f"  {session['speaker']} × {session['mic']} "
                         f"[{path.stem}]: {count}{mark}")
        lines += ["", "По условиям"]
        for condition, count in sorted(Counter(
                row["condition"] for row in manifest).items()):
            lines.append(f"  {condition}: {count}")
    problems = [row for row in qc_rows if row["reason"] or row["warnings"]]
    lines += ["", "Проблемы"]
    if problems:
        for row in problems:
            detail = ", ".join(
                item for item in (row["reason"], row["warnings"]) if item)
            lines.append(f"  {row['file']}: {detail}")
    else:
        lines.append("  нет")
    report = "\n".join(lines) + "\n"
    (root / "qc_report.txt").write_text(report, encoding="utf-8")
    print(report)


def write_listen_page(root: Path, rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["speaker"], row["mic"], row["phrase"],
               row["condition"], row["distance_m"], row["level"])
        groups[key].append(row)
    cards = []
    for key in sorted(groups):
        speaker, mic, phrase, condition, distance, level = key
        picked = sorted(groups[key], key=lambda r: int(r["take"]))[:2]
        for row in picked:
            meta = [
                f"{speaker} × {mic}",
                f"{phrase}",
                f"{distance} м",
                CONDITION_RU.get(condition, condition),
                LEVEL_RU.get(level, level),
                f"{row['duration_s']} с",
            ]
            if row["text"]:
                meta.append(f"ASR: «{row['text']}»")
            cards.append(
                '<div class="card">'
                f'<div class="title">{html.escape(str(key[0]))} · '
                f'{html.escape(str(key[1]))}</div>'
                f'<audio controls preload="metadata" '
                f'src="{html.escape(row["file"])}"></audio>'
                f'<div class="meta">{html.escape(" · ".join(meta))}</div>'
                "</div>")
    page = f"""<!doctype html>
<html lang="ru">
<meta charset="utf-8">
<title>Реальные позитивы — контроль</title>
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
<h1>Реальные позитивы: по два дубля на ячейку</h1>
<p class="hint">Проверьте на слух: слышно ли «хей луна», нет ли запинок,
щелчков, обрывов и лишнего фона. Забракованные клипы — через
<code>--review</code> или вручную в <code>exclude.txt</code>.</p>
<p>Всего клипов: {len(rows)}, показано: {len(cards)}.</p>
{''.join(cards)}
</html>
"""
    (root / "listen_positives.html").write_text(page, encoding="utf-8")


# --------------------------------------------------------------------------
# Прослушивание manifest
# --------------------------------------------------------------------------

def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return [row for row in csv.DictReader(f, delimiter=";")
                if row.get("kept", "1") == "1"]


def review(root: Path, manifest_path: Path | None = None) -> int:
    manifest = manifest_path or root / "manifest.csv"
    rows = read_manifest(manifest)
    if not rows:
        print(f"Нет manifest с kept=1: {manifest}", file=sys.stderr)
        return 2
    remaining: list[dict] = []
    rejected = 0
    total = len(rows)
    for index, row in enumerate(rows, 1):
        path = root / row["file"]
        if not path.exists():
            print(f"Нет файла: {row['file']}")
            remaining.append(row)
            continue
        print()
        print(f"{index}/{total} · {row['speaker']} × {row['mic']} · "
              f"{row['phrase']} · {row['distance_m']} м · "
              f"{CONDITION_RU.get(row['condition'], row['condition'])} · "
              f"{LEVEL_RU.get(row['level'], row['level'])}")
        while True:
            play_wav(path)
            print("  [Пробел] оставить  [X] брак  [P] ещё раз  [Q] выход")
            drain_keys()
            key = read_key()
            if key in (" ", "\r", "\n"):
                remaining.append(row)
                break
            if key in ("x", "ч"):
                note = ask_comment()
                append_exclusion(root / "exclude.txt", row["file"], note)
                rejected += 1
                print("  Исключено.")
                break
            if key in ("p", "з"):
                continue
            if key in ("q", "й"):
                remaining.extend(rows[index - 1:])
                write_manifest(manifest, remaining)
                print(f"Выход: оставлено {len(remaining)}, "
                      f"исключено {rejected}")
                return 0
    write_manifest(manifest, remaining)
    print(f"Готово: оставлено {len(remaining)}, исключено {rejected}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def list_devices() -> int:
    try:
        devices = list_input_devices()
    except Exception as exc:
        print(f"Не удалось получить устройства: {exc}", file=sys.stderr)
        return 2
    if not devices:
        print("Нет входных устройств.")
        return 2
    for device in devices:
        mark = " *" if device["default"] else ""
        print(f"{device['index']:>3}: {device['name']} · "
              f"{device['channels']} кан. · {device['rate']} Гц{mark}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запись реальных позитивов «хей луна» (этап 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list-devices", action="store_true",
                        help="показать входные устройства")
    parser.add_argument("--setup", action="store_true",
                        help="начать новую сессию")
    parser.add_argument("--resume", action="store_true",
                        help="продолжить сессию")
    parser.add_argument("--fresh", action="store_true",
                        help="архивировать старую сессию и начать заново")
    parser.add_argument("--review", nargs="?", const="", default=None,
                        metavar="ПУТЬ",
                        help="прослушать manifest и забраковать лишнее")
    parser.add_argument("--finalize", action="store_true",
                        help="raw → 16 кГц, QC, manifest, страница")
    parser.add_argument("--speaker", help="id диктора (anna, tim, …)")
    parser.add_argument("--mic", default="laptop",
                        help="id микрофона (laptop, studio, …)")
    parser.add_argument("--device", default=None,
                        help="индекс или часть имени входного устройства")
    parser.add_argument("--session", type=Path, default=None,
                        help="путь к журналу для --resume")
    parser.add_argument("--root", type=Path, default=None,
                        help="каталог positives_real (по умолчанию data)")
    parser.add_argument("--takes-per-cell", type=int, default=None,
                        help="дублей на ячейку (по умолчанию 3)")
    parser.add_argument("--limit", type=int, default=0,
                        help="остановиться после N принятых дублей")
    parser.add_argument("--with-extra", dest="with_extra",
                        action="store_true", default=True,
                        help="блок «эй луна» (по умолчанию включён)")
    parser.add_argument("--no-extra", dest="with_extra",
                        action="store_false",
                        help="не записывать блок «эй луна»")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="не записывать 5 с фона")
    parser.add_argument("--skip-pilot", action="store_true",
                        help="не делать пробный дубль уровня")
    parser.add_argument("--asr-hints", dest="asr_hints",
                        action="store_true", default=True,
                        help="подсказки GigaAM (по умолчанию включены)")
    parser.add_argument("--no-asr-hints", dest="asr_hints",
                        action="store_false",
                        help="отключить подсказки GigaAM")
    parser.add_argument("--no-stt", action="store_true",
                        help="финализация без GigaAM")
    parser.add_argument("--seed", type=int, default=0,
                        help="порядок ячеек внутри блока")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    root = args.root or data_root(cfg)

    if args.list_devices:
        return list_devices()
    if args.finalize:
        return finalize(root, args.no_stt)
    if args.review is not None:
        if not args.review:
            return review(root)
        review_path = Path(args.review)
        if review_path.is_file():
            return review(review_path.parent, review_path)
        return review(review_path)
    if args.resume:
        return resume_latest(root, args.session, args.device)
    if args.setup:
        if not args.speaker:
            print("Для --setup нужен --speaker", file=sys.stderr)
            return 2
        return start_session(args, cfg, root)

    parse_args(["--help"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
