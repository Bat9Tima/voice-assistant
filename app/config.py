from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
# Модели лежат в ASCII-пути: Vosk и CTranslate2 на Windows плохо переносят
# кириллицу в путях, а папка проекта может быть названа по-русски.
DATA_DIR = Path(os.environ.get("LOCALAPPDATA") or str(ROOT)) / "VoiceAssistant"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = ROOT / "logs"
STATE_DIR = ROOT / "state"
VOSK_DIR = MODELS_DIR / "vosk-ru-small"
VOSK_EN_DIR = MODELS_DIR / "vosk-en-small"
WHISPER_DIR = MODELS_DIR / "whisper"
GIGAAM_DIR = MODELS_DIR / "sherpa-onnx-nemo-transducer-giga-am-v3-russian-2025-12-16"

DEFAULTS: dict[str, Any] = {
    "wake_word": "окей ассистент",
    "wake_aliases": ["окей ассистент", "ок ассистент", "эй ассистент"],
    # vosk — распознаёт речь и ищет фразу в тексте; openwakeword — лёгкая
    # модель под одну конкретную фразу.
    "wake_engine": "vosk",
    "wake_model": "hey_loona.onnx",
    "wake_threshold": 0.5,
    # Слова, похожие по звучанию на ключевое. Нужны, чтобы Vosk не
    # «подгонял» их под единственное слово в грамматике.
    "wake_confusables": [],
    "language": "ru",
    "hotkey": "ctrl+alt+space",
    "notifications": True,
    "log_level": "INFO",
    "audio": {
        "sample_rate": 16000,
        "block_size": 4000,
        "input_device": None,
        "energy_threshold": 0.010,
        "noise_multiplier": 2.0,
        "pre_roll_sec": 2.5,
        "silence_ms": 800,
        "min_speech_ms": 250,
        "max_command_sec": 15,
        "no_speech_timeout_sec": 6,
    },
    "stt": {
        "engine": "gigaam",
        "num_threads": 4,
        "tail_padding_sec": 0.4,
        "model": "small",
        "device": "cpu",
        "compute_type": "int8",
        "beam_size": 5,
        "initial_prompt": "",
        "parallel_english": False,
        "english_engine": "whisper",
        "english_model": "tiny",
        "english_num_threads": 4,
        "english_device": "cpu",
        "english_compute_type": "int8",
        "english_beam_size": 1,
        "english_crop_beam_size": 5,
        "english_locator": "vosk",
    },
    "browser": {
        "chrome_path": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "port": 9222,
        "profile_dir": r"%LOCALAPPDATA%\VoiceAssistant\chrome-profile",
        "start_url": "https://www.google.com",
        "autoplay": True,
    },
    "control": {"host": "127.0.0.1", "port": 8765},
    "opencode": {
        "enabled": True,
        "agent": "voice",
        "cli": "",
        "timeout_sec": 180,
        "standalone": False,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand_path(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(str(value)))


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Читает config.json, дополняет значениями по умолчанию."""
    cfg_path = path or CONFIG_PATH
    data: dict[str, Any] = {}
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    cfg = _deep_merge(DEFAULTS, data)

    cfg["browser"]["profile_dir"] = _expand_path(cfg["browser"]["profile_dir"])
    cfg["browser"]["chrome_path"] = _expand_path(cfg["browser"]["chrome_path"])

    for directory in (MODELS_DIR, LOGS_DIR, STATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    Path(cfg["browser"]["profile_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg
