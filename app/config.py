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
    "wake_word": "хей луна",
    "wake_aliases": ["хей луна", "эй луна"],
    # vosk — распознаёт речь и ищет фразу в тексте; openwakeword — лёгкая
    # модель под одну конкретную фразу.
    "wake_engine": "vosk",
    "wake_model": "hey_loona.onnx",
    "wake_threshold": 0.5,
    # Слова, похожие по звучанию на ключевое. Нужны, чтобы Vosk не
    # «подгонял» их под единственное слово в грамматике.
    "wake_confusables": [
        "луна", "луну", "лужа", "люда", "юля", "уна", "дума", "руна",
        "лупа", "нуна", "лун", "хей люда", "хей юля", "эй лужа",
        "лей луну", "хей ну на", "хей луна парк", "эй луна парк",
    ],
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
        "vad_enabled": True,
        "vad_threshold": 0.5,
        "vad_model": "silero_vad.onnx",
        "duck_while_listening": True,
        "duck_level": 0.15,
        "duck_processes": ["chrome.exe"],
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
    # Короткий контекст продолжений: «ещё один», «вторую», «тише».
    # TTL ограничивает, как долго старая команда может влиять на новую.
    "context": {"ttl_sec": 180},
    "opencode": {
        "enabled": True,
        "agent": "voice",
        # Явная модель: без неё CLI может выбрать модель, недоступную
        # в регионе (403 RegionError). Пустая строка — модель по умолчанию.
        "model": "opencode/deepseek-v4.1-flash-free",
        # Агент вызывается только явно: «спроси агента …». Остальные
        # незнакомые фразы получают быстрый ответ «Не понял команду».
        "triggers": ["спроси агента", "спроси у агента", "передай агенту", "агент"],
        "cli": "",
        "timeout_sec": 180,
        "standalone": False,
    },
    # Ответы на вопросы, новости и рассуждения — через облачную модель
    # ProxyAPI. Ключ можно не дублировать: при пустом api_key берётся
    # tts.api_key, потому что оба сервиса работают через ProxyAPI.
    "cloud_llm": {
        "enabled": True,
        "base_url": "https://api.proxyapi.ru/v1",
        "api_key": "",
        "model": "anthropic/claude-haiku-5-5",
        "fallback_models": ["xiaomi/mimo-v2.6-flash"],
        "reasoning_effort": "minimal",
        "max_completion_tokens": 2000,
        "temperature": 0.2,
        "timeout_sec": 90,
        "retries": 2,
    },
    # Аварийный локальный планировщик свободных команд (Ollama/LM Studio).
    # По умолчанию выключен и никогда не участвует в ответах на вопросы.
    "local_planner": {
        "enabled": False,
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:3b-instruct",
        "api_key": "",
        "timeout_sec": 20,
        "temperature": 0.0,
        "max_tokens": 256,
        "keep_alive": "30m",
    },
    # Вопросы: погода через Open-Meteo (без ключа), свежие данные через
    # Tavily (нужен ключ), остальное — облачная LLM.
    "qa": {
        "enabled": True,
        "weather": {
            "city": "Новосибирск",
            "forecast_days": 3,
            "timeout_sec": 10,
        },
        "search": {
            "enabled": True,
            "api_key": "",
            "max_results": 5,
            "search_depth": "basic",
            "topic": "general",
            "news_days": 7,
            "news_domains": [
                "rbc.ru",
                "lenta.ru",
                "interfax.ru",
                "ria.ru",
                "tass.ru",
                "kommersant.ru",
                "vedomosti.ru",
            ],
            "timeout_sec": 15,
        },
        "answer": {
            "max_tokens": 400,
            "temperature": 0.2,
            "timeout_sec": 45,
        },
    },
    # Озвучка: предзаписанные фразы играют с диска, ответы синтезируются
    # через ProxyAPI. Ключ и голос задаются в config.json.
    "tts": {
        "enabled": False,
        "base_url": "https://api.proxyapi.ru/v1/audio/speech",
        "api_key": "",
        "model": "fish-audio/drama-3",
        "voice": "",
        "phrases_dir": "",
        "cache_dir": "",
        "speak_answers": True,
        "max_chars": 900,
        "volume": 1.0,
        "speed": 1.0,
        "workers": 3,
        "timeout_sec": 60,
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
