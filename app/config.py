from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

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
        "луна",
        "луну",
        "лужа",
        "люда",
        "юля",
        "уна",
        "дума",
        "руна",
        "лупа",
        "нуна",
        "лун",
        "хей люда",
        "хей юля",
        "эй лужа",
        "лей луну",
        "хей ну на",
        "хей луна парк",
        "эй луна парк",
    ],
    # Перебивание голосом во время озвучки: «стоп», «хватит», «тише».
    # Пустой список стоп-слов — встроенный набор из app/wake.py.
    "barge_in": {
        "enabled": True,
        "min_level": 0.002,
        "stop_words": [],
    },
    "language": "ru",
    "hotkey": "ctrl+alt+space",
    # Пауза микрофона. Пустая строка выключает комбинацию.
    "pause_hotkey": "ctrl+alt+m",
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
        # Сохранять последнюю команду в state/last-command.wav для отладки.
        "save_last_command": False,
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
    },
    # Короткий контекст продолжений: «ещё один», «вторую», «тише».
    # TTL ограничивает, как долго старая команда может влиять на новую.
    "context": {"ttl_sec": 180},
    # Быстрое переключение облачных моделей: имя пресета из llm_presets.
    # Пресет накладывается последним и перекрывает model/model_params
    # секций, поэтому менять модель нужно только здесь.
    "llm": {"preset": "gpt"},
    "llm_presets": {
        "gpt": {
            "cloud_llm": {
                "model": "openai/gpt-6-luna",
                "model_params": {
                    "openai/gpt-6-luna": {
                        "temperature": None,
                        "reasoning_effort": "none",
                    }
                },
            },
            "dialog": {"addressee": {"model": "openai/gpt-6-luna-decisions"}},
            "intent": {"model": "openai/gpt-6-luna-decisions"},
        },
        "grok": {
            "cloud_llm": {"model": "x-ai/grok-4.3"},
            # Эндпоинт решений пока проверен только с этой моделью.
            "dialog": {"addressee": {"model": "openai/gpt-6-luna-decisions"}},
            "intent": {"model": "openai/gpt-6-luna-decisions"},
        },
    },
    # Ответы на вопросы, новости и рассуждения — через облачную модель
    # ProxyAPI. Ключ можно не дублировать: при пустом api_key берётся
    # tts.api_key, потому что оба сервиса работают через ProxyAPI.
    "cloud_llm": {
        "enabled": True,
        "base_url": "https://api.proxyapi.ru/v1",
        "api_key": "",
        "model": "openai/gpt-6-luna",
        "fallback_models": ["xiaomi/mimo-v2.6-flash"],
        "reasoning_effort": "minimal",
        # Ограничения конкретных моделей. null убирает поле из запроса.
        # GPT-6 Luna принимает только temperature по умолчанию и не
        # понимает minimal: без этого каждый запрос падал бы с HTTP 400.
        "model_params": {
            "openai/gpt-6-luna": {
                "temperature": None,
                "reasoning_effort": "none",
            }
        },
        "max_completion_tokens": 2000,
        "temperature": 0.2,
        "timeout_sec": 90,
        "retries": 2,
    },
    # Зрение по требованию: один снимок основного экрана по явной команде.
    # Модель отдельная: переключение чата на Grok не должно ломать зрение.
    "vision": {
        "enabled": True,
        "model": "openai/gpt-6-luna",
        "max_width": 1280,
        "jpeg_quality": 75,
        "max_tokens": 2000,
        "timeout_sec": 60,
        "followup_ttl_sec": 60,
        # Сколько секунд вопрос «что в углу?» ещё относится к последнему
        # снимку. Явные ссылки («на фото», «что было») от срока не зависят.
        "context_ttl_sec": 1800,
        "prompt": (
            "Опиши, что видно на экране, в двух-трёх предложениях. "
            "Если на экране текст, код или ошибка — назови главное. "
            "Если экран чёрный или пустой, так и скажи."
        ),
        "triggers": [
            "посмотри на экран",
            "посмотри на мой экран",
            "посмотри что на экране",
            "глянь на экран",
            "что на экране",
            "что видно на экране",
            "что ты видишь на экране",
            "что у меня на экране",
        ],
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
            "context_ttl_sec": 180,
            "search_fallback": True,
        },
    },
    # Непрерывный разговор: вход и выход по фразам, без wake-слова на
    # каждую реплику. История живёт в памяти на время разговора, в модель
    # уходит только окно history_messages/history_chars. Постоянный
    # контекст — текстовый файл profile_path, который пишет пользователь.
    # Завершённые сессии пишутся в SQLite db_path.
    "dialog": {
        "enabled": True,
        # off — вход только по entry-фразам; conversation — после wake,
        # если команды не было; always — любой wake открывает разговор.
        "auto_start": "off",
        "entry_phrases": ["давай поговорим", "давай поболтаем", "поговорим", "поболтаем"],
        "exit_phrases": [
            "хватит",
            "закончим",
            "закончим разговор",
            "заверши разговор",
            "до свидания",
        ],
        # Слой адресата: в разговоре решает, говорит ли пользователь с
        # Луной или звучит фон (телевизор, другой человек, телефон).
        # off — выключен, shadow — только пишет решение в лог, active —
        # фон пропускается, а closing завершает разговор. При сбое сети
        # реплика считается адресованной.
        "addressee": {
            "enabled": True,
            "mode": "shadow",
            "base_url": "",
            "api_key": "",
            "model": "openai/gpt-6-luna-decisions",
            "threshold": 0.6,
            "background_threshold": 0.5,
            "timeout_sec": 2.0,
            "first_turn_always_addressed": True,
            # Имена людей, с которыми пользователь говорит в комнате.
            # Имя рядом со вторым лицом («Максим, ты...») или повторённое
            # имя сразу считаются фоном.
            "other_names": [],
        },
        "idle_timeout_sec": 60,
        "max_utterance_sec": 60,
        "no_speech_timeout_sec": 10,
        "silence_ms": 1200,
        "max_tokens": 400,
        "temperature": 0.5,
        "timeout_sec": 60,
        "max_errors": 2,
        "history_messages": 12,
        "history_chars": 4000,
        # Потоковый ответ: модель отдаёт текст кусками, озвучка начинает
        # звучать, не дожидаясь всего ответа. Без TTS разницы нет.
        "streaming": {
            "enabled": True,
        },
        # Бюджет одного хода в токенах (оценка по символам). При
        # переполнении режутся память, факты Луны, профиль и старые
        # реплики; правила и текущая реплика — никогда. 0 отключает.
        "context": {
            "budget_tokens": 6000,
            "chars_per_token": 2.5,
            # Реплики, выпавшие из окна истории, сжимаются в краткое
            # содержание текущего разговора. Фоновый запрос, в базу не
            # пишется, живёт только до конца разговора.
            "rolling_summary": {
                "enabled": True,
                "min_messages": 4,
                "max_chars": 1200,
                "max_input_chars": 4000,
                "max_tokens": 250,
                "timeout_sec": 30,
            },
        },
        "profile_path": "state/profile.txt",
        "profile_chars": 4000,
        # Факты Луны о себе: файл пополняется после разговоров и идёт
        # в системный промпт отдельным блоком. Пустой путь выключает.
        "persona_path": "state/luna.txt",
        "persona_chars": 2000,
        # Сколько фактов о Луне хранить в файле. Новые вытесняют старые,
        # близкие повторы схлопываются. 0 — без ограничения.
        "persona_max_facts": 100,
        "db_path": "state/dialog.db",
        # Адаптивная манера: явные просьбы пользователя («без мата»,
        # «короче», «сегодня без шуток») хранятся версиями в той же базе.
        # Модель сама ничего в характер не записывает.
        "character": {
            "enabled": True,
        },
        # Наблюдатель стиля: считает манеру пользователя и по короткому
        # окну включает мат или меняет длину ответов. Остальные признаки
        # пока только пишет в лог и SQLite.
        "style_observer": {
            "enabled": True,
            "window_messages": 200,
            "decision_messages": 30,
            "log_every": 10,
        },
        # После выхода сессия сжимается в короткое резюме и ложится в
        # таблицу summaries; резюме уходит в семантическую память, а факты
        # Луны о себе — в persona_path. background — считать в фоновом
        # потоке, чтобы не задерживать прощание; пустой model — cloud_llm.
        "summary": {
            "enabled": True,
            "model": "",
            "max_input_chars": 6000,
            "max_tokens": 700,
            "temperature": 0.2,
            "timeout_sec": 45,
            "min_messages": 2,
            "background": True,
        },
    },
    # Классификатор намерений поверх правил. Модель решений ProxyAPI не
    # пишет текст, а отдаёт вероятности: команда, ответ, разговор, справка,
    # уточнение. shadow — только пишет решение в лог; active — может
    # отменить погоду, новости и поиск, если фраза оказалась разговором.
    # Ключ берётся из intent.api_key, иначе из cloud_llm.api_key.
    "intent": {
        "enabled": True,
        "mode": "shadow",
        "base_url": "https://api.proxyapi.ru/v1",
        "api_key": "",
        "model": "openai/gpt-6-luna-decisions",
        "threshold": 0.6,
        "timeout_sec": 4.0,
        # Сколько секунд вопрос Луны считается ожидающим ответа.
        "answer_ttl_sec": 120,
    },
    # Семантическая память: резюме прошлых разговоров ищутся по смыслу.
    # Векторы считает Voyage через ProxyAPI, хранит Chroma в path. Тексты
    # резюме остаются в SQLite; Chroma можно пересобрать. Память работает
    # только в режиме диалога, разовые команды её не трогают.
    "memory": {
        "enabled": True,
        "model": "mongodb/voyage-4",
        "base_url": "",
        "api_key": "",
        "path": "state/chroma",
        "collection": "summaries",
        "top_k": 3,
        "min_score": 0.35,
        "min_margin": 0.05,
        "max_chars": 2400,
        "digest_min_messages": 6,
        "digest_limit": 3,
        "digest_item_chars": 320,
        # Последние разговоры по времени: показываются до семантических
        # находок, чтобы свежая тема не проигрывала заметной старой.
        "recent_limit": 2,
        "recent_item_chars": 320,
        # Отдельные факты памяти: важное идёт в промпт перед дайджестом.
        # Лимит символов считается только по блоку фактов.
        "facts_limit": 6,
        "facts_max_chars": 900,
        "facts_collection": "facts",
        "facts_top_k": 4,
        "facts_min_score": 0.42,
        # Временные состояния и задачи не должны жить вечно. Ноль
        # отключает соответствующий срок.
        "temporary_ttl_days": 1,
        "unfinished_ttl_days": 14,
        "unfinished_limit": 5,
        "timeout_sec": 10,
        # Версия раскладки индекса: при изменении старые векторы не ищутся,
        # а пересчитываются из резюме при старте.
        "index_version": 2,
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
        # Сколько секунд ждать синтез фрагмента, когда предыдущий уже
        # доиграл. 0 — ждать без ограничения.
        "chunk_wait_sec": 30,
        # Небольшой буфер перед стартом фрагмента: если следующий уже
        # синтезируется, даём ему досинтезироваться, чтобы стык не был
        # слышен как пауза. Задерживает первый звук, поэтому по умолчанию
        # выключен. 0 — не ждать.
        "prebuffer_sec": 0,
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


def _apply_llm_preset(cfg: dict[str, Any]) -> None:
    """Накладывает выбранный пресет моделей поверх остального конфига."""
    name = str((cfg.get("llm") or {}).get("preset") or "").strip()
    if not name:
        return
    presets = cfg.get("llm_presets") or {}
    preset = presets.get(name)
    if not isinstance(preset, dict):
        log.warning(
            "Неизвестный пресет моделей: %s (доступны: %s)",
            name,
            ", ".join(sorted(presets)) or "нет",
        )
        return
    for section, values in preset.items():
        if isinstance(values, dict) and isinstance(cfg.get(section), dict):
            cfg[section] = _deep_merge(cfg[section], values)
        else:
            cfg[section] = values


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Читает config.json, дополняет значениями по умолчанию."""
    cfg_path = path or CONFIG_PATH
    data: dict[str, Any] = {}
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    cfg = _deep_merge(DEFAULTS, data)
    _apply_llm_preset(cfg)

    cfg["browser"]["profile_dir"] = _expand_path(cfg["browser"]["profile_dir"])
    cfg["browser"]["chrome_path"] = _expand_path(cfg["browser"]["chrome_path"])

    dialog_cfg = cfg.get("dialog") or {}
    for key in ("profile_path", "persona_path", "db_path"):
        value = str(dialog_cfg.get(key) or "").strip()
        if not value:
            continue
        expanded = _expand_path(value)
        if not Path(expanded).is_absolute():
            expanded = str(ROOT / expanded)
        dialog_cfg[key] = expanded

    memory_cfg = cfg.get("memory") or {}
    memory_path = str(memory_cfg.get("path") or "").strip()
    if memory_path:
        expanded = _expand_path(memory_path)
        if not Path(expanded).is_absolute():
            expanded = str(ROOT / expanded)
        memory_cfg["path"] = expanded

    for directory in (MODELS_DIR, LOGS_DIR, STATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    Path(cfg["browser"]["profile_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg
