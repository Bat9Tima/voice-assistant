from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import wave
from pathlib import Path

from .audio import Microphone, list_input_devices, rms
from .brain import OpenCodeBrain
from .config import (
    GIGAAM_DIR,
    ROOT,
    STATE_DIR,
    VOSK_DIR,
    VOSK_EN_DIR,
    WHISPER_DIR,
    load_config,
)
from .local_llm import LocalLLM
from .stt import create_transcriber
from .vad import ensure_model
from .wake import WakeWordDetector


def _check_vosk() -> bool:
    return VOSK_DIR.exists() and any(VOSK_DIR.iterdir())


def _check_vosk_en() -> bool:
    return VOSK_EN_DIR.exists() and any(VOSK_EN_DIR.iterdir())


def _check_faster_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _check_gigaam() -> bool:
    encoder = GIGAAM_DIR / "encoder.int8.onnx"
    if not encoder.exists():
        encoder = GIGAAM_DIR / "encoder.onnx"
    required = [
        encoder,
        GIGAAM_DIR / "decoder.onnx",
        GIGAAM_DIR / "joiner.onnx",
        GIGAAM_DIR / "tokens.txt",
    ]
    return all(path.exists() for path in required)


def _check_whisper() -> bool:
    if not WHISPER_DIR.exists():
        return False
    return any(WHISPER_DIR.rglob("model.bin"))

def _check_cdp(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def _check_control(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/ping", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def _read_wav(path: Path, chunk_samples: int = 4000) -> tuple[Path, list[bytes]]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
            raise ValueError(
                f"Ожидается WAV 16 кГц mono 16 бит, получено: "
                f"{wav.getframerate()} Гц, {wav.getnchannels()} кан., "
                f"{wav.getsampwidth() * 8} бит"
            )
        data = wav.readframes(wav.getnframes())
    size = chunk_samples * 2
    return path, [data[index : index + size] for index in range(0, len(data), size)]


def _record(seconds: float, device) -> tuple[Path, list[bytes]]:
    mic = Microphone(device=device)
    mic.start()
    print(f"Говорите сейчас ({seconds:.0f} с)...")
    frames: list[bytes] = []
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            chunk = mic.read(timeout=0.5)
            if chunk:
                frames.append(chunk)
    finally:
        mic.stop()

    target = STATE_DIR / "doctor-recording.wav"
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"".join(frames))
    return target, frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Диагностика голосового ассистента")
    parser.add_argument("--record", type=float, metavar="СЕК", help="Записать пробу с микрофона")
    parser.add_argument("--file", metavar="WAV", help="Проверить WAV-файл вместо микрофона")
    parser.add_argument("--wake", action="store_true", help="Проверить ключевое слово на записи")
    parser.add_argument("--stt", action="store_true", help="Распознать запись выбранным движком STT")
    args = parser.parse_args(argv)

    cfg = load_config()
    print("=== Диагностика Voice Assistant ===")
    print(f"Корень проекта: {ROOT}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Wake word: «{cfg['wake_word']}» | горячая клавиша: {cfg['hotkey']}")

    print("\n[Микрофоны]")
    devices = list_input_devices()
    if not devices:
        print("  Не найдено ни одного устройства ввода!")
    for device in devices:
        mark = " *" if device["default"] else "  "
        print(
            f"{mark} [{device['index']}] {device['name']} "
            f"({device['channels']} кан., {device['default_samplerate']} Гц)"
        )

    print("\n[Модели]")
    print(f"  Vosk: {'OK' if _check_vosk() else 'НЕТ — запустите scripts\\setup.cmd'} ({VOSK_DIR})")
    vad_path = ensure_model()
    print(f"  Silero VAD: {'OK' if vad_path else 'нет — останется порог громкости'}")
    engine = str(cfg["stt"].get("engine") or "gigaam")
    if engine in {"gigaam", "giga-am", "sherpa", "sherpa-onnx", "nemo"}:
        gigaam_state = "OK" if _check_gigaam() else "НЕТ — запустите scripts\\setup.cmd"
        print(f"  GigaAM v3: {gigaam_state} ({GIGAAM_DIR})")
    else:
        whisper_state = "OK" if _check_whisper() else "будет скачана при первом запуске"
        print(f"  Whisper {cfg['stt']['model']}: {whisper_state} ({WHISPER_DIR})")
    if bool(cfg["stt"].get("parallel_english", False)):
        en_locator = "OK" if _check_vosk_en() else "НЕТ — запустите scripts\\setup.cmd"
        print(f"  Vosk EN: {en_locator} ({VOSK_EN_DIR})")
        en_model = str(cfg["stt"].get("english_model") or "tiny")
        en_whisper = "OK" if _check_whisper() else "НЕТ — запустите scripts\\setup.cmd"
        print(f"  Whisper EN {en_model}: {en_whisper} ({WHISPER_DIR})")
        en_runtime = (
            "установлен"
            if _check_faster_whisper()
            else "НЕТ — pip install -r requirements-whisper.txt"
        )
        print(f"  faster-whisper: {en_runtime}")
    print(f"  Движок STT: {engine}")

    print("\n[Браузер]")
    chrome = Path(cfg["browser"]["chrome_path"])
    print(f"  Chrome: {'найден' if chrome.exists() else 'НЕ найден'} ({chrome})")
    print(f"  Профиль: {cfg['browser']['profile_dir']}")
    print(
        "  CDP: "
        + ("работает" if _check_cdp(int(cfg["browser"]["port"])) else "не запущен (стартует по команде)")
    )

    print("\n[OpenCode]")
    brain = OpenCodeBrain(cfg)
    print(f"  CLI: {brain.cli or 'НЕ найден'}")
    print(f"  Агент: {brain.agent}")

    print("\n[Локальная LLM]")
    llm_cfg = cfg.get("llm") or {}
    if not llm_cfg.get("enabled"):
        print("  Выключена (llm.enabled = false)")
    else:
        llm = LocalLLM(cfg)
        print(f"  URL: {llm.base_url}")
        print(f"  Модель: {llm.model or 'НЕ задана'}")
        print("  Сервер: " + ("доступен" if llm.available() else "НЕ доступен"))

    print("\n[HTTP-мост]")
    control = cfg["control"]
    running = _check_control(control["host"], int(control["port"]))
    print(
        f"  http://{control['host']}:{control['port']}: "
        + ("работает" if running else "не запущен (стартует вместе с ассистентом)")
    )

    if args.record or args.file:
        if args.file:
            target, frames = _read_wav(Path(args.file))
            print(f"\n[Файл] {target}")
        else:
            target, frames = _record(args.record, cfg["audio"].get("input_device"))
        levels = [rms(frame) for frame in frames if frame]
        if levels:
            levels_sorted = sorted(levels)
            median = levels_sorted[len(levels_sorted) // 2]
            # Медиана на записи с речью завышена: большую часть файла занимает
            # голос. Для оценки шума берём нижний процентиль.
            noise = levels_sorted[max(0, int(len(levels_sorted) * 0.2) - 1)]
            recommended = min(0.05, max(0.008, noise * 2.5))
            print(
                f"  RMS: мин {min(levels):.4f} / шум (20%) {noise:.4f} / "
                f"медиана {median:.4f} / макс {max(levels):.4f}"
            )
            print(f"  Рекомендуемый energy_threshold: {recommended:.3f}")

        pcm = b"".join(frames)
        if args.wake and _check_vosk():
            detector = WakeWordDetector(
                VOSK_DIR,
                cfg.get("wake_aliases") or [cfg["wake_word"]],
                sample_rate=int(cfg["audio"]["sample_rate"]),
            )
            triggered = False
            for frame in frames:
                if detector.feed(frame):
                    triggered = True
                    break
            print(f"  Ключевое слово: {'НАЙДЕНО' if triggered else 'не найдено'}")

        if args.stt:
            print("  Распознавание (может занять время)...")
            transcriber = create_transcriber(cfg)
            details = transcriber.transcribe_details(pcm)
            print(f"  Текст: {details.text!r}")
            russian = getattr(details, "russian", None)
            if russian is not None:
                english = getattr(details, "english", None)
                print(
                    f"  RU: {russian.text!r} | "
                    f"EN: {(english.text if english else '')!r} | "
                    f"источник: {getattr(details, 'source', 'ru')}"
                )

    print("\nГотово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
