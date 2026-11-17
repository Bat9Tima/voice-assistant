from __future__ import annotations

import argparse
import logging
import logging.handlers
import math
import queue
import re
import sys
import threading
import time
from collections import deque
from typing import Any

from .actions import ActionRouter
from .audio import Microphone, levels, rms, save_wav_pcm
from .brain import OpenCodeBrain
from .browser import BrowserController
from .cloud_llm import CloudLLM
from .config import DATA_DIR, LOGS_DIR, MODELS_DIR, STATE_DIR, VOSK_DIR, load_config
from .dialog import DialogService
from .duck import AppVolumeDucker
from .intent_classifier import MODE_ACTIVE, IntentClassifier
from .memory import MemoryService
from .memory_manager import MemoryManager
from .persona import PersonaMemory
from .profile import Profile
from .qa import QAService, Route
from .server import ControlServer
from .session_store import SessionStore
from .single_instance import acquire
from .stt import create_transcriber
from .summarizer import Summarizer
from .text import canonicalize, normalize, split_agent_trigger, strip_wake_word
from .tray import TrayUI
from .tts import TTSService
from .vad import EndpointTracker, SileroVad
from .wake import BargeInDetector, BargeInMatch, OpenWakeWordDetector, WakeWordDetector

log = logging.getLogger(__name__)

# Внутри разговора эти фразы обслуживают быстрые сервисы, остальное уходит
# в чат с историей. Уточнения и общие вопросы — тоже чат.
DIALOG_QA_KINDS = frozenset({"datetime", "weather", "news", "news_detail", "search"})
# Из них классификатор может отменить только те, что требуют внешних
# данных. Дата считается локально, а уточнения держатся за контекст.
GATED_QA_KINDS = frozenset({"weather", "news", "search"})


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "assistant.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    if sys.stderr is not None and getattr(sys.stderr, "isatty", lambda: False)():
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)
    for noisy in ("faster_whisper", "urllib3", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def beep(frequency: int, duration_ms: int) -> None:
    try:
        import winsound

        winsound.Beep(frequency, duration_ms)
    except Exception:
        pass


_PYNPUT_SPECIAL = {
    "space": "<space>",
    "enter": "<enter>",
    "return": "<enter>",
    "tab": "<tab>",
    "esc": "<esc>",
    "escape": "<esc>",
    "backspace": "<backspace>",
    "delete": "<delete>",
    "insert": "<insert>",
    "home": "<home>",
    "end": "<end>",
    "pageup": "<page_up>",
    "page_up": "<page_up>",
    "pagedown": "<page_down>",
    "page_down": "<page_down>",
    "up": "<up>",
    "down": "<down>",
    "left": "<left>",
    "right": "<right>",
}


def to_pynput(hotkey: str) -> str:
    parts = []
    for part in hotkey.lower().split("+"):
        part = part.strip()
        if part in {"ctrl", "alt", "shift"}:
            parts.append(f"<{part}>")
        elif part in {"win", "cmd", "super"}:
            parts.append("<cmd>")
        elif part in _PYNPUT_SPECIAL:
            parts.append(_PYNPUT_SPECIAL[part])
        elif re.fullmatch(r"f\d{1,2}", part):
            parts.append(f"<{part}>")
        else:
            parts.append(part)
    return "+".join(parts)


class Assistant:
    def __init__(self, cfg: dict[str, Any], use_tray: bool = True) -> None:
        self.cfg = cfg
        self.use_tray = use_tray
        audio_cfg = cfg["audio"]
        self.browser = BrowserController(cfg)
        self.router = ActionRouter(cfg, self.browser)
        self.brain = OpenCodeBrain(cfg)
        self.planner = None
        if (cfg.get("local_planner") or {}).get("enabled"):
            # Модуль тянет реестр инструментов, поэтому импортируем его
            # только когда планировщик действительно включён.
            from .local_planner import LocalPlanner

            self.planner = LocalPlanner(cfg)
        self.llm = CloudLLM(cfg)
        self.qa = QAService(cfg, self.llm, self.browser)
        self.intent = IntentClassifier(cfg)
        self.tts = TTSService(cfg)
        self.dialog: DialogService | None = None
        self.session_store: SessionStore | None = None
        self.summarizer: Summarizer | None = None
        self.memory: MemoryService | None = None
        self.memory_manager: MemoryManager | None = None
        dialog_cfg = cfg.get("dialog") or {}
        if bool(dialog_cfg.get("enabled", True)):
            profile = Profile(
                dialog_cfg.get("profile_path"),
                limit=int(dialog_cfg.get("profile_chars") or 4000),
            )
            db_path = str(dialog_cfg.get("db_path") or "").strip()
            persona = PersonaMemory(
                dialog_cfg.get("persona_path"),
                limit=int(dialog_cfg.get("persona_chars") or 2000),
            )
            if db_path:
                self.session_store = SessionStore(db_path)
                self.memory_manager = MemoryManager.from_config(
                    self.session_store, cfg.get("memory")
                )
                self.memory = MemoryService.from_config(
                    cfg, self.session_store, manager=self.memory_manager
                )
            self.dialog = DialogService(
                cfg,
                self.llm,
                profile,
                memory=self.memory,
                persona=persona,
            )
            if self.session_store is not None:
                self.session_store.attach(self.dialog.sessions)
                self.summarizer = Summarizer.from_config(
                    self.llm,
                    self.session_store,
                    dialog_cfg.get("summary"),
                    memory=self.memory,
                    persona=persona,
                    manager=self.memory_manager,
                ).attach(self.dialog.sessions)
                log.info("Сессии диалога сохраняются в %s", db_path)
                if persona.path is not None:
                    log.info("Память личности: %s", persona.path)
                if self.memory is not None:
                    log.info("Семантическая память включена: %s", self.memory.embedder.model)
                    self._restore_memory_async()
                self._recover_sessions_async()
        self._dialog_active = False
        self._dialog_last = 0.0
        self._dialog_errors = 0
        self.agent_triggers = list(cfg["opencode"].get("triggers") or [])
        self.stt = create_transcriber(cfg)
        self.mic = Microphone(
            samplerate=int(audio_cfg["sample_rate"]),
            block_size=int(audio_cfg["block_size"]),
            device=audio_cfg.get("input_device"),
        )
        self.ducker: AppVolumeDucker | None = None
        if bool(audio_cfg.get("duck_while_listening", False)):
            self.ducker = AppVolumeDucker(
                process_names=audio_cfg.get("duck_processes") or ["chrome.exe"],
                level=float(audio_cfg.get("duck_level", 0.15)),
                state_path=DATA_DIR / "duck-state.json",
            )
        self.vad: SileroVad | None = None
        if bool(audio_cfg.get("vad_enabled", True)):
            self.vad = SileroVad.create(
                model=audio_cfg.get("vad_model"),
                threshold=float(audio_cfg.get("vad_threshold", 0.5)),
            )
        self.wake: WakeWordDetector | OpenWakeWordDetector | None = None
        self.barge: BargeInDetector | None = None
        self.tray: TrayUI | None = TrayUI(self) if use_tray else None
        self.server: ControlServer | None = None
        self._hotkey_listener = None
        self._mutex = None

        self._state_lock = threading.Lock()
        self._state = "starting"
        self._paused = False
        self._shutdown = threading.Event()
        self._force_listen = threading.Event()
        self._command_queue: queue.Queue[str] = queue.Queue()
        self._noise_floor = 0.0
        self._last_wake = 0.0
        self._tts_was_speaking = False
        # Последние ~1-3 секунды звука: чтобы не потерять начало команды,
        # если пользователь заговорил сразу после ключевой фразы.
        self._preroll: deque[bytes] = deque(maxlen=12)

    # ------------------------------------------------------------------ memory
    def _recover_sessions_async(self) -> None:
        """Закрывает разговоры, оборванные прошлым сбоем, и суммирует их."""
        summarizer = self.summarizer
        if summarizer is None:
            return
        threading.Thread(
            target=self._recover_sessions,
            args=(summarizer,),
            name="dialog-recover",
            daemon=True,
        ).start()

    @staticmethod
    def _recover_sessions(summarizer: Summarizer) -> None:
        try:
            summarizer.recover()
        except Exception:  # noqa: BLE001
            log.exception("Не удалось закрыть оборванные сессии")

    def _restore_memory_async(self) -> None:
        """Достраивает индекс памяти в фоне, не задерживая запуск."""
        memory = self.memory
        if memory is None:
            return
        threading.Thread(
            target=self._restore_memory,
            args=(memory,),
            name="memory-restore",
            daemon=True,
        ).start()

    @staticmethod
    def _restore_memory(memory: MemoryService) -> None:
        try:
            restored = memory.restore()
        except Exception:  # noqa: BLE001
            log.exception("Память: восстановление индекса упало")
            return
        if restored:
            log.info("Память: восстановлено резюме %s", restored)

    # ------------------------------------------------------------------- state
    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    def set_state(self, state: str) -> None:
        with self._state_lock:
            self._state = state
        if self.tray is not None:
            self.tray.set_state(state)

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "paused": self._paused,
            "wake_word": self.cfg.get("wake_word"),
            "hotkey": self.cfg.get("hotkey"),
            "stt_loaded": self.stt.loaded,
            "stt_english_loaded": getattr(self.stt, "english_loaded", False),
            "wake_loaded": self.wake is not None,
            "speaking": bool(getattr(getattr(self, "tts", None), "busy", False)),
            "dialog": bool(getattr(self, "_dialog_active", False)),
            "tray": self.tray.status() if self.tray is not None else None,
        }

    def notify(self, message: str, title: str = "Ассистент") -> None:
        if self.tray is not None and self.cfg.get("notifications", True):
            self.tray.notify(message, title)
        else:
            log.info("%s: %s", title, message)

    def clear_notifications(self) -> None:
        """Убирает уведомление предыдущей команды перед новой."""
        if self.tray is not None:
            self.tray.clear()

    # ------------------------------------------------------------------- startup
    def run(self) -> int:
        self._mutex = acquire()
        if self._mutex is None:
            return 0

        if self.ducker is not None:
            # Предыдущий запуск мог быть прерван во время приглушения Chrome.
            try:
                self.ducker.recover()
            except Exception:  # noqa: BLE001
                log.exception("Не удалось восстановить звук после прежнего запуска")

        # winrt (Windows Toast) падает с access violation, если onnxruntime
        # загрузится уже после него. Грузим onnxruntime первым — его всё равно
        # использует GigaAM, независимо от движка wake-слова.
        try:
            import onnxruntime  # noqa: F401
        except Exception:  # noqa: BLE001
            log.debug("onnxruntime не предзагрузился", exc_info=True)

        if self.tray is not None:
            self.tray.start()

        self.server = ControlServer(
            self.cfg,
            self.browser,
            status_provider=self.status,
            command_sink=self.submit_text,
        )
        self.server.start()

        try:
            self.mic.start()
            self._noise_floor = self.mic.measure_noise(1.0)
            # Размер pre-roll зависит от частоты захвата: на 48 кГц каждый
            # чанк короче, чем на 16 кГц, и фиксированное число чанков дало бы
            # разную длину истории.
            block_sec = self.mic.block_size / max(1, self.mic.capture_rate)
            pre_roll_sec = float(self.cfg["audio"].get("pre_roll_sec", 2.5))
            self._preroll = deque(maxlen=max(4, int(round(pre_roll_sec / block_sec))))
            log.info(
                "Фоновый шум микрофона: %.4f, pre-roll: %.1f с",
                self._noise_floor,
                self._preroll.maxlen * block_sec,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Микрофон недоступен")
            self.notify(f"Микрофон недоступен: {exc}. Работает горячая клавиша и HTTP-команды.")
            self.mic = None  # type: ignore[assignment]

        aliases = self.cfg.get("wake_aliases") or [self.cfg.get("wake_word", "хей луна")]
        engine = str(self.cfg.get("wake_engine", "vosk")).lower()
        if engine == "openwakeword":
            try:
                model_name = self.cfg.get("wake_model") or "hey_luna.onnx"
                self.wake = OpenWakeWordDetector(
                    MODELS_DIR / "openwakeword" / model_name,
                    threshold=float(self.cfg.get("wake_threshold", 0.5)),
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("openWakeWord не загрузился, откатываюсь на Vosk")
                self.notify(f"openWakeWord не загрузился: {exc}. Использую Vosk.")

        if self.wake is None:
            audio_cfg = self.cfg["audio"]
            quiet_level = max(
                float(audio_cfg.get("energy_threshold", 0.01)),
                self._noise_floor * float(audio_cfg.get("noise_multiplier", 2.0)),
            )
            try:
                self.wake = WakeWordDetector(
                    VOSK_DIR,
                    aliases,
                    sample_rate=int(audio_cfg["sample_rate"]),
                    confusables=self.cfg.get("wake_confusables") or [],
                    quiet_level=quiet_level,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("Детектор ключевого слова не загрузился")
                self.notify(f"Vosk не загрузился: {exc}. Используйте {self.cfg['hotkey']}.")

        barge_cfg = self.cfg.get("barge_in") or {}
        if self.wake is None or not bool(barge_cfg.get("enabled", True)):
            log.info("Перебивание голосом выключено")
        elif not isinstance(self.wake, WakeWordDetector):
            # openWakeWord ищет только свою акустическую фразу и не умеет
            # распознавать «стоп», поэтому перебивание оставляем для Vosk.
            log.info("Перебивание голосом недоступно с openWakeWord")
        else:
            try:
                self.barge = BargeInDetector(
                    self.wake.model,
                    aliases,
                    sample_rate=int(self.cfg["audio"]["sample_rate"]),
                    stop_words=barge_cfg.get("stop_words") or None,
                    min_level=float(barge_cfg.get("min_level", 0.002)),
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("Детектор перебивания не загрузился")
                self.notify(f"Перебивание голосом недоступно: {exc}")

        threading.Thread(target=self._warmup_stt, daemon=True, name="stt-warmup").start()
        threading.Thread(target=self._start_hotkey, daemon=True, name="hotkey").start()

        self.set_state("idle")
        self.notify(
            f"Ассистент запущен. Скажите «{self.cfg.get('wake_word', 'хей луна')}» "
            f"или нажмите {self.cfg['hotkey']}."
        )
        if not self.tts.phrase("hello"):
            beep(880, 90)

        try:
            self._main_loop()
        except KeyboardInterrupt:
            log.info("Остановка по Ctrl+C")
        finally:
            self._cleanup()
        return 0

    def _main_loop(self) -> None:
        log.info("Основной цикл запущен")
        while not self._shutdown.is_set():
            if self._paused:
                time.sleep(0.15)
                continue

            if self.tts.speaking:
                if not self._tts_was_speaking:
                    self._tts_was_speaking = True
                    # В pre-roll не должно остаться звука от прошлой команды.
                    self._preroll.clear()
                    # Детектор wake слушает и во время озвучки: начинаем
                    # с чистого состояния, чтобы не сработать на хвосте.
                    if self.wake is not None:
                        self.wake.reset()

                # Команда из API важнее текущей озвучки: не ждём её конца.
                try:
                    text = self._command_queue.get_nowait()
                except queue.Empty:
                    text = None
                if text:
                    self._run_text_command(text)
                    continue

                self._handle_speech()
                continue
            if self._tts_was_speaking:
                self._tts_was_speaking = False
                if self.mic is not None:
                    self.mic.flush()
                if self.wake is not None:
                    self.wake.reset()
                if self.barge is not None:
                    self.barge.reset()
                self._preroll.clear()

            if self._force_listen.is_set():
                self._force_listen.clear()
                if self._dialog_active:
                    # Горячая клавиша закрывает разговор, как фраза «хватит».
                    self._stop_dialog("hotkey")
                    continue
                self._listen_and_execute()
                continue

            try:
                text = self._command_queue.get_nowait()
            except queue.Empty:
                text = None
            if text:
                self._run_text_command(text)
                continue

            if self._dialog_active:
                self._dialog_step()
                continue

            if self.wake is None or self.mic is None:
                time.sleep(0.1)
                continue

            try:
                chunk = self.mic.read(timeout=0.3)
                if not chunk:
                    continue
                triggered = self.wake.feed(chunk)
                self._preroll.append(chunk)
                if triggered and time.monotonic() - self._last_wake > 3.0:
                    self._last_wake = time.monotonic()
                    pre_roll = list(self._preroll)
                    self._preroll.clear()
                    self._listen_and_execute(pre_roll)
            except Exception:  # noqa: BLE001
                log.exception("Ошибка в основном цикле")
                self.set_state("error")
                time.sleep(1)
                self.set_state("idle")

    def _dialog_step(self) -> None:
        """Один шаг разговора: проверка тишины и запись следующей реплики."""
        dialog = getattr(self, "dialog", None)
        if dialog is None:
            self._dialog_active = False
            return
        if time.monotonic() - self._dialog_last >= dialog.idle_timeout:
            self._stop_dialog("idle")
            return
        if getattr(self.tts, "busy", False):
            # Ответ ещё в очереди озвучки: не начинаем запись раньше времени.
            time.sleep(0.05)
            return
        if self.mic is None:
            # Диалог из текстовых команд: микрофона нет, ждём API или hotkey.
            time.sleep(0.1)
            return
        self._listen_and_execute(chime=False, dialog=True)

    def _run_text_command(self, text: str) -> None:
        """Новая команда начинает новый ответ, старые уведомления не нужны."""
        self.clear_notifications()
        try:
            self._process(text)
            if getattr(self, "_dialog_active", False):
                self._dialog_last = time.monotonic()
        finally:
            if self.mic is not None:
                self.mic.flush()
            self._preroll.clear()

    def _handle_speech(self) -> None:
        """Слушает микрофон, пока ассистент говорит: «стоп» или wake-фраза."""
        if self.mic is None:
            time.sleep(0.05)
            return
        try:
            chunk = self.mic.read(timeout=0.15)
        except Exception:  # noqa: BLE001
            log.exception("Ошибка чтения микрофона во время озвучки")
            time.sleep(0.1)
            return
        if not chunk:
            return
        self._preroll.append(chunk)
        if self.barge is None:
            return
        try:
            self._barge_in(self.barge, chunk)
        except Exception:  # noqa: BLE001
            # Сбой детектора не должен завершать ассистента: приводим
            # детекторы в чистое состояние и продолжаем слушать.
            log.exception("Ошибка обработки перебивания")
            try:
                self.barge.reset()
                if self.wake is not None:
                    self.wake.reset()
            except Exception:  # noqa: BLE001
                log.exception("Не удалось сбросить детекторы после ошибки")

    def _barge_in(self, barge: BargeInDetector, chunk: bytes) -> None:
        match = barge.feed(chunk)
        if match is None and self.wake is not None:
            # Свободное распознавание часто теряет первое слово wake-фразы
            # («элвина», «коммуна», «сити»). Грамматический детектор на том
            # же звуке слышит её надёжно, поэтому спрашиваем и его.
            if self.wake.feed(chunk):
                match = BargeInMatch(
                    "wake",
                    getattr(self.wake, "last_match", None) or "",
                )
        if match is None:
            return
        # Ассистент мог сам произнести «стоп» или wake-фразу: не перебиваем
        # себя. Хватает текущего фрагмента: эхо с задержкой отсекает
        # энергетический гейт в детекторе.
        spoken = normalize(getattr(self.tts, "current_text", "") or "")
        if match.phrase and re.search(
            rf"(?<!\w){re.escape(match.phrase)}(?!\w)",
            spoken,
        ):
            log.debug("Перебивание: «%s» есть в озвучке, игнорирую", match.phrase)
            barge.reset()
            return

        log.info("Перебивание голосом: «%s» (%s)", match.phrase, match.action)
        self.tts.stop()
        barge.reset()
        if self.wake is not None:
            self.wake.reset()
        if match.action == "wake":
            # Отдаём весь pre-roll: Vosk финализирует фразу с задержкой,
            # и команда после «хей луна» уже успевает попасть в буфер.
            pre_roll = list(self._preroll)
            self._preroll.clear()
            self._last_wake = time.monotonic()
            if getattr(self, "_dialog_active", False):
                self._listen_and_execute(pre_roll, chime=False, dialog=True)
            else:
                self._listen_and_execute(pre_roll, chime=False)
        else:
            self._preroll.clear()
            self.set_state("idle")

    def _warmup_stt(self) -> None:
        try:
            self.stt.load()
            log.info(
                "Распознавание речи готово (%s), английский: %s",
                type(self.stt).__name__,
                "да" if getattr(self.stt, "english_loaded", False) else "нет",
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Распознавание речи не загрузилось")
            self.notify(
                f"Распознавание речи не загрузилось: {exc}. Быстрые команды работают без него."
            )

    def _start_hotkey(self) -> None:
        try:
            from pynput import keyboard

            combo = to_pynput(self.cfg.get("hotkey", "ctrl+alt+space"))
            listener = keyboard.GlobalHotKeys({combo: self.trigger_listen})
            self._hotkey_listener = listener
            listener.start()
            log.info("Горячая клавиша: %s (%s)", self.cfg.get("hotkey"), combo)
        except Exception:  # noqa: BLE001
            log.exception("Не удалось зарегистрировать горячую клавишу")

    # ------------------------------------------------------------------- listen
    def trigger_listen(self) -> None:
        if self._paused:
            self.notify("Ассистент на паузе")
            return
        if self.state in {"listening", "thinking"} and not getattr(self, "_dialog_active", False):
            return
        self._force_listen.set()

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        self.set_state("paused" if self._paused else "idle")
        self.notify("Пауза" if self._paused else "Снова слушаю")
        self.tts.phrase("mic", phrase_id="mic_1" if self._paused else "mic_2")

    def show_status(self) -> None:
        info = self.status()
        self.notify(
            f"Состояние: {info['state']}. Wake word: «{info['wake_word']}». "
            f"Горячая клавиша: {info['hotkey']}.",
            "Статус",
        )

    def _listen_and_execute(
        self,
        pre_roll: list[bytes] | None = None,
        *,
        chime: bool = True,
        dialog: bool = False,
    ) -> None:
        if self._paused or self.mic is None:
            return
        self.set_state("listening")
        # Новая команда: убираем результат предыдущей, не ждём, пока
        # системный центр уведомлений до неё доберётся. В разговоре
        # автопрослушивание не должно стирать ответ каждые 10 секунд.
        if not dialog:
            self.clear_notifications()
        if chime and not dialog:
            if not self.tts.phrase("wake"):
                beep(880, 80)
            time.sleep(0.12)
            # Фраза «Слушаю» не должна попасть в запись команды.
            if self.mic is not None:
                self.mic.flush()
        else:
            # Даём звуковой карте выбросить хвост остановленной озвучки.
            time.sleep(0.05)
        if self.wake is not None:
            self.wake.reset()
        if self.barge is not None:
            self.barge.reset()
        if self.ducker is not None:
            # Музыка из браузера не должна попадать в запись команды.
            # Громкость вернётся сразу после захвата, до распознавания.
            try:
                self.ducker.duck()
            except Exception:  # noqa: BLE001
                log.exception("Не удалось приглушить звук приложений")
        try:
            if self.mic is not None:
                if pre_roll is None:
                    # Горячая клавиша: до нажатия могла звучать обычная речь.
                    self.mic.flush()
                else:
                    # Активация голосом: не теряем начало команды, сказанной без
                    # паузы («хей луна открой ютуб»). Добавляем накопленное за
                    # сигнал аудио к pre-roll, но не считаем его речью команды —
                    # иначе сигнал запускал бы отсчёт тишины.
                    extra: list[bytes] = []
                    while True:
                        chunk = self.mic.read(timeout=0)
                        if not chunk:
                            break
                        extra.append(chunk)
                    pre_roll = [*pre_roll, *extra]

            pcm = self._record_command(pre_roll, dialog=dialog)
        except Exception:  # noqa: BLE001
            log.exception("Ошибка записи команды")
            pcm = b""
        finally:
            if self.ducker is not None:
                try:
                    self.ducker.restore()
                except Exception:  # noqa: BLE001
                    log.exception("Не удалось вернуть звук приложений")
            if self.wake is not None:
                self.wake.reset()

        if not pcm:
            if dialog:
                log.info("Диалог: тишина, продолжаю слушать")
                self.set_state("idle")
                return
            self.notify("Не услышал команду")
            beep(320, 120)
            self.tts.phrase("repeat")
            self.set_state("idle")
            return

        peak, level = levels(pcm)
        log.info(
            "Команда: %.2f с, пик %.1f dBFS, RMS %.1f dBFS",
            len(pcm) / 2 / 16000,
            20 * math.log10(max(peak, 1e-9)),
            20 * math.log10(max(level, 1e-9)),
        )
        if bool(self.cfg["audio"].get("save_last_command", False)):
            try:
                target = STATE_DIR / "last-command.wav"
                save_wav_pcm(target, pcm)
                log.info("Команда сохранена: %s", target)
            except Exception:  # noqa: BLE001
                log.exception("Не удалось сохранить команду")

        self.set_state("thinking")
        stt_started = time.monotonic()
        try:
            details = self.stt.transcribe_details(pcm)
            text = details.text
            russian = getattr(details, "russian", None)
            english = getattr(details, "english", None)
            log.info(
                "STT RU: %r | EN: %r | источник: %s | %.2f с",
                russian.text if russian is not None else text,
                english.text if english is not None else "",
                getattr(details, "source", "ru"),
                time.monotonic() - stt_started,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Ошибка распознавания")
            self.notify(f"Ошибка распознавания: {exc}")
            if not dialog:
                beep(320, 120)
                self.tts.phrase("error")
            self.set_state("idle")
            return

        if not text:
            if dialog:
                log.info("Диалог: пустая реплика")
                self.set_state("idle")
                return
            self.notify("Не разобрал команду")
            beep(320, 120)
            self.tts.phrase("repeat")
            self.set_state("idle")
            return

        log.info("STT: %r", text)
        text = canonicalize(strip_wake_word(text, self.cfg.get("wake_aliases") or []))
        if not text:
            if dialog:
                log.info("Диалог: после wake-фразы ничего не осталось")
                self.set_state("idle")
                return
            self.notify("Не разобрал команду")
            beep(320, 120)
            self.tts.phrase("repeat")
            self.set_state("idle")
            return

        log.info("Распознано: %s", text)
        if dialog:
            # Тишина в диалоге отсчитывается от последней реплики.
            self._dialog_last = time.monotonic()
        else:
            self.notify(f"«{text}»", "Услышал")
        try:
            self._process(text)
        finally:
            if self.mic is not None:
                self.mic.flush()
            self._preroll.clear()
            self.set_state("idle")

    def _record_command(
        self,
        pre_roll: list[bytes] | None = None,
        *,
        dialog: bool = False,
    ) -> bytes:
        assert self.mic is not None
        audio_cfg = self.cfg["audio"]
        threshold = max(
            float(audio_cfg.get("energy_threshold", 0.01)),
            self._noise_floor * float(audio_cfg.get("noise_multiplier", 2.0)),
        )
        min_speech = int(audio_cfg.get("min_speech_ms", 250))
        if dialog:
            # В разговоре паузы длиннее, а реплика может быть длинной.
            dialog_cfg = self.cfg.get("dialog") or {}
            silence_limit = int(dialog_cfg.get("silence_ms", 1200))
            max_ms = float(dialog_cfg.get("max_utterance_sec", 60)) * 1000
            no_speech_limit = float(dialog_cfg.get("no_speech_timeout_sec", 10)) * 1000
        else:
            silence_limit = int(audio_cfg.get("silence_ms", 800))
            max_ms = float(audio_cfg.get("max_command_sec", 15)) * 1000
            no_speech_limit = float(audio_cfg.get("no_speech_timeout_sec", 6)) * 1000

        initial = list(pre_roll or [])
        chunks: list[bytes] = list(initial)
        levels: list[float] = [rms(chunk) for chunk in initial]
        vad_flags: list[bool] = [False] * len(initial)
        preroll_ms = sum(len(chunk) / 2 / 16000 * 1000 for chunk in initial)
        # Vosk может определить фразу уже после того, как она закончилась.
        # Тогда команда целиком лежит в pre-roll, и новых громких чанков нет.
        preroll_has_speech = any(level >= threshold for level in levels)
        tracker = EndpointTracker(
            silence_ms=silence_limit,
            min_speech_ms=min_speech,
        )
        if self.vad is not None:
            self.vad.reset()

        total_ms = 0.0
        while total_ms < max_ms:
            chunk = self.mic.read(timeout=0.5)
            if chunk is None:
                continue
            duration_ms = len(chunk) / 2 / 16000 * 1000
            total_ms += duration_ms
            level = rms(chunk)
            vad_speech = self.vad.feed(chunk) if self.vad is not None else False
            chunks.append(chunk)
            levels.append(level)
            vad_flags.append(vad_speech)
            tracker.update(
                duration_ms,
                vad_speech=vad_speech,
                loud=level >= threshold,
            )

            if tracker.started and tracker.silence_ms >= tracker.silence_limit:
                break
            if not tracker.started and total_ms >= no_speech_limit:
                break

        if not tracker.has_speech and not preroll_has_speech:
            return b""

        start_index = 0
        if pre_roll is None:
            # Для горячей клавиши обрезаем ведущую тишину. При активации
            # голосом pre-roll уже выбран и может содержать тихое начало
            # команды — его не трогаем.
            first_speech: int | None = None
            for index, flag in enumerate(vad_flags):
                if flag:
                    first_speech = index
                    break
            for index, level in enumerate(levels):
                if level >= threshold * 0.8:
                    if first_speech is None or index < first_speech:
                        first_speech = index
                    break
            if first_speech is not None:
                start_index = max(0, first_speech - 2)
        log.info(
            "Запись команды: всего %.0f мс, речь %.0f мс, тишина %.0f мс, "
            "pre-roll %.0f мс, VAD %.0f мс (%s), порог %.4f",
            total_ms,
            tracker.speech_ms,
            tracker.silence_ms,
            preroll_ms,
            tracker.vad_ms,
            "вкл" if self.vad is not None else "выкл",
            threshold,
        )
        return b"".join(chunks[start_index:])

    # ------------------------------------------------------------------ process
    def submit_text(self, text: str) -> None:
        self._command_queue.put(text)
        log.info("Текстовая команда из API: %s", text)

    def _ack(self, text: str, route: Route | None = None) -> None:
        """Короткая фраза-подводка, чтобы ответ начинался сразу."""
        if not self.tts.enabled:
            return
        if split_agent_trigger(text, self.agent_triggers)[0]:
            category = "agent"
        elif route is not None and route.handled:
            if route.kind == "datetime":
                # Ответ мгновенный: подводка только добавит паузу.
                return
            category = route.ack
        else:
            category = "ack"
        self.tts.phrase(category)

    def _confirm(self, started: float) -> None:
        """«Готово» только если операция заняла заметное время."""
        if time.monotonic() - started >= 1.2:
            self.tts.phrase("done")

    def _start_dialog(self) -> str:
        """Включает непрерывный разговор и здоровается."""
        dialog = getattr(self, "dialog", None)
        if dialog is not None:
            dialog.start()
        self._dialog_active = True
        self._dialog_errors = 0
        self._dialog_last = time.monotonic()
        self.set_state("idle")
        message = "Слушаю. О чём поговорим?"
        advisor = getattr(self, "intent", None)
        if advisor is not None:
            advisor.state.note_question(message)
        self.notify(message)
        if not self.tts.say(message):
            beep(880, 80)
        log.info("Режим диалога включён")
        return message

    def _stop_dialog(self, reason: str = "phrase") -> str:
        """Выключает разговор. Причина: phrase, hotkey, idle, errors."""
        self._dialog_active = False
        self._dialog_errors = 0
        dialog = getattr(self, "dialog", None)
        if dialog is not None:
            dialog.stop()
        advisor = getattr(self, "intent", None)
        if advisor is not None:
            advisor.state.clear()
        self.set_state("idle")
        if reason == "idle":
            message = "Похоже, разговор закончился."
        elif reason == "errors":
            message = "Не получается ответить, выхожу из разговора."
        else:
            message = "Хорошо, закончили."
        self.notify(message)
        if not self.tts.phrase("bye"):
            self.tts.say(message)
        log.info("Режим диалога выключен (%s)", reason)
        return message

    def _dialog_reply(self, text: str) -> str:
        """Ответ модели в разговоре. Ошибки не рвут цикл с первой попытки."""
        dialog = getattr(self, "dialog", None)
        if dialog is None:
            return "Не понял команду"
        started = time.monotonic()
        stream = None
        streamed = False

        def on_chunk(piece: str) -> None:
            """Заводит озвучку с первого куска: без TTS задача не нужна."""
            nonlocal stream, streamed
            if stream is None:
                starter = getattr(self.tts, "stream", None)
                if starter is None:
                    return
                stream = starter()
                if stream is None:
                    return
            streamed = True
            stream.feed(piece)

        try:
            if getattr(dialog, "streaming", False):
                answer = dialog.reply(text, on_chunk=on_chunk)
            else:
                answer = dialog.reply(text)
        except Exception:  # noqa: BLE001
            if stream is not None:
                stream.close()
                stream = None
            log.exception("Ошибка ответа в диалоге")
            self._dialog_errors = int(getattr(self, "_dialog_errors", 0)) + 1
            message = "Не получилось ответить."
            self.notify(message)
            if self._dialog_errors >= dialog.max_errors:
                return self._stop_dialog("errors")
            self.tts.say(message)
            return message
        finally:
            if stream is not None:
                stream.close()
        self._dialog_errors = 0
        self._dialog_last = time.monotonic()
        advisor = getattr(self, "intent", None)
        if advisor is not None:
            advisor.state.note_reply(answer)
        self.notify(answer[:250])
        if not streamed:
            self.tts.say(answer)
        log.info(
            "Диалог: ответ за %.2f с, символов %s",
            time.monotonic() - started,
            len(answer),
        )
        return answer

    def _dialog_route(self, text: str, route: Route | None) -> Route | None:
        """Классификатор поверх правил внутри разговора.

        В shadow он только пишет статистику в фоне. В active перед
        маршрутом к внешним данным ждёт решение: если модель уверена, что
        это разговор, ответ или уточнение, маршрут отменяется. Дата и
        уточнения к прошлому ответу классификатор не трогает.
        """
        advisor = getattr(self, "intent", None)
        if advisor is None:
            return route
        if (
            route is not None
            and route.kind in GATED_QA_KINDS
            and advisor.mode == MODE_ACTIVE
        ):
            decision = advisor.classify(text, in_dialog=True)
            if advisor.blocks_lookup(decision):
                log.info(
                    "Маршрут %s отменён: реплика похожа на разговор (%s, %.2f)",
                    route.kind,
                    decision.intent,
                    decision.confidence,
                )
                return None
            return route
        advisor.observe(text, in_dialog=True)
        return route

    def _process(self, text: str) -> str:
        text = canonicalize(normalize(text))
        if getattr(self.tts, "busy", getattr(self.tts, "speaking", False)):
            log.info("Прерываю озвучку новой командой")
            stop = getattr(self.tts, "stop", None)
            if stop is not None:
                stop()
        self.set_state("thinking")
        try:
            dialog = getattr(self, "dialog", None)
            in_dialog = bool(getattr(self, "_dialog_active", False))
            if dialog is not None:
                if not in_dialog and dialog.wants_entry(text):
                    return self._start_dialog()
                if in_dialog and dialog.wants_exit(text):
                    return self._stop_dialog("phrase")

            qa = getattr(self, "qa", None)
            # Разбираем фразу один раз: и для подводки, и для ответа.
            route = qa.route(text) if qa is not None else None
            if in_dialog:
                route = self._dialog_route(text, route)
            if in_dialog and route is not None and route.kind not in DIALOG_QA_KINDS:
                # В разговоре всё, что не команда и не справка, уходит в чат.
                route = None

            triggered, request = split_agent_trigger(text, self.agent_triggers)
            if not in_dialog or route is not None or triggered:
                self._ack(text, route)
            started = time.monotonic()

            if triggered:
                if not request:
                    message = "Что спросить у агента?"
                    self.notify(message)
                    self.tts.phrase("clarify", phrase_id="clarify_1")
                    return message
                answer = self.brain.ask(request)
                answer = (answer or "").strip()
                if answer:
                    self.notify(answer[:250])
                    self.tts.say(answer)
                else:
                    self.notify("Готово")
                    self.tts.phrase("done")
                return answer

            result = self.router.handle(text)
            if result is not None:
                message = result.message
                if message:
                    self.notify(message)
                if result.ok:
                    self._confirm(started)
                else:
                    beep(320, 120)
                    self.tts.phrase("error")
                return message

            if qa is not None and route is not None and route.handled:
                answer = qa.answer(text, route=route)
                message = answer.short or answer.error or "Не смог ответить"
                self.notify(message[:250])
                if answer.ok:
                    # Вкладка с полным текстом не должна задерживать звук:
                    # Playwright открывает её, пока TTS уже говорит.
                    render = threading.Thread(
                        target=qa.show,
                        args=(answer,),
                        name="qa-render",
                        daemon=True,
                    )
                    render.start()
                    self.tts.say(message)
                    render.join(timeout=2)
                else:
                    beep(320, 120)
                    if answer.kind == "no_model":
                        # Причина понятна и без сети: озвучиваем её, а не общую фразу.
                        self.tts.say(message)
                    else:
                        self.tts.phrase("error")
                return message

            if in_dialog:
                # Команды и справки уже отработали; остальное — свободный разговор.
                return self._dialog_reply(text)

            plan = None
            planner = getattr(self, "planner", None)
            if planner is not None:
                try:
                    plan = planner.plan(text)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Локальный планировщик: %s", exc)
            if plan is not None:
                result = self.router.execute_plan(plan.action, plan.args)
                message = result.message
                if message:
                    self.notify(message)
                if result.ok:
                    self._confirm(started)
                else:
                    beep(320, 120)
                    self.tts.phrase("error")
                return message

            # Вне разговора классификатор пока только наблюдает: ответ
            # остаётся прежним, но в лог попадает, что это была за фраза.
            advisor = getattr(self, "intent", None)
            if advisor is not None:
                advisor.observe(text, in_dialog=False)
            message = "Не понял команду"
            self.notify(message)
            beep(320, 120)
            self.tts.phrase("unknown")
            return message
        except Exception as exc:  # noqa: BLE001
            log.exception("Ошибка обработки команды")
            self.notify(f"Ошибка: {exc}"[:250])
            beep(320, 120)
            self.tts.phrase("error")
            return f"Ошибка: {exc}"
        finally:
            self.set_state("idle")

    def run_text(self, text: str) -> str:
        """Обрабатывает текст без микрофона (для отладки и тестов)."""
        return self._process(normalize(text))

    # ------------------------------------------------------------------ cleanup
    def shutdown(self) -> None:
        log.info("Запрошено завершение")
        self._shutdown.set()

    def _cleanup(self) -> None:
        log.info("Завершаю работу")
        dialog = getattr(self, "dialog", None)
        if dialog is not None:
            # Программу закрыли посреди разговора: сессия всё равно должна
            # закрыться, иначе резюме её не увидит.
            dialog.stop()
        summarizer = getattr(self, "summarizer", None)
        if summarizer is not None:
            # Сырая сессия уже в базе; даём фоновому резюме дописаться,
            # но не держим закрытие программы дольше десяти секунд.
            summarizer.close(timeout=10)
        memory = getattr(self, "memory", None)
        if memory is not None:
            memory.close()
        if self.mic is not None:
            self.mic.stop()
        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop()
            except Exception:
                pass
        if self.server is not None:
            self.server.stop()
        self.tts.shutdown()
        self.browser.stop()
        if self.tray is not None:
            self.tray.stop()
        log.info("Ассистент остановлен")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Голосовой ассистент на базе OpenCode")
    parser.add_argument("--no-tray", action="store_true", help="Без иконки в трее")
    parser.add_argument("--text", help="Выполнить текстовую команду и выйти")
    parser.add_argument("--config", help="Путь к config.json")
    namespace = parser.parse_args(argv)

    cfg = load_config(namespace.config)
    setup_logging(cfg.get("log_level", "INFO"))
    log.info("=== Voice Assistant %s ===", "0.1.0")

    assistant = Assistant(cfg, use_tray=not namespace.no_tray)
    if namespace.text:
        result = assistant.run_text(namespace.text)
        # Озвучка теперь асинхронная: даём ей договорить перед выходом.
        assistant.tts.wait(30)
        assistant.tts.shutdown()
        print(result)
        return 0
    return assistant.run()


if __name__ == "__main__":
    raise SystemExit(main())
