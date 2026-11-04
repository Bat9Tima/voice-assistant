from __future__ import annotations

import argparse
import logging
import logging.handlers
import math
import os
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
from .config import DATA_DIR, LOGS_DIR, MODELS_DIR, STATE_DIR, VOSK_DIR, load_config
from .duck import AppVolumeDucker
from .server import ControlServer
from .single_instance import acquire
from .stt import create_transcriber
from .text import canonicalize, normalize, strip_wake_word
from .tray import TrayUI
from .vad import EndpointTracker, SileroVad
from .wake import OpenWakeWordDetector, WakeWordDetector

log = logging.getLogger(__name__)


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
        # Последние ~1-3 секунды звука: чтобы не потерять начало команды,
        # если пользователь заговорил сразу после ключевой фразы.
        self._preroll: deque[bytes] = deque(maxlen=12)

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

        aliases = self.cfg.get("wake_aliases") or [
            self.cfg.get("wake_word", "окей ассистент")
        ]
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
            try:
                self.wake = WakeWordDetector(
                    VOSK_DIR,
                    aliases,
                    sample_rate=int(self.cfg["audio"]["sample_rate"]),
                    confusables=self.cfg.get("wake_confusables") or [],
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("Детектор ключевого слова не загрузился")
                self.notify(f"Vosk не загрузился: {exc}. Используйте {self.cfg['hotkey']}.")

        threading.Thread(target=self._warmup_stt, daemon=True, name="stt-warmup").start()
        threading.Thread(target=self._start_hotkey, daemon=True, name="hotkey").start()

        self.set_state("idle")
        self.notify(
            f"Ассистент запущен. Скажите «{self.cfg.get('wake_word', 'окей ассистент')}» "
            f"или нажмите {self.cfg['hotkey']}."
        )
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

            if self._force_listen.is_set():
                self._force_listen.clear()
                self._listen_and_execute()
                continue

            try:
                text = self._command_queue.get_nowait()
            except queue.Empty:
                text = None
            if text:
                # Текстовая команда из API начинает новую команду — старые
                # уведомления больше не актуальны.
                self.clear_notifications()
                try:
                    self._process(text)
                finally:
                    if self.mic is not None:
                        self.mic.flush()
                    self._preroll.clear()
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
                f"Распознавание речи не загрузилось: {exc}. "
                "Быстрые команды работают без него."
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
        if self.state in {"listening", "thinking"}:
            return
        self._force_listen.set()

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        self.set_state("paused" if self._paused else "idle")
        self.notify("Пауза" if self._paused else "Снова слушаю")

    def show_status(self) -> None:
        info = self.status()
        self.notify(
            f"Состояние: {info['state']}. Wake word: «{info['wake_word']}». "
            f"Горячая клавиша: {info['hotkey']}.",
            "Статус",
        )

    def _listen_and_execute(self, pre_roll: list[bytes] | None = None) -> None:
        if self._paused or self.mic is None:
            return
        self.set_state("listening")
        # Новая команда: убираем результат предыдущей, не ждём, пока
        # системный центр уведомлений до неё доберётся.
        self.clear_notifications()
        beep(880, 80)
        time.sleep(0.12)
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

            pcm = self._record_command(pre_roll)
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
            self.notify("Не услышал команду")
            beep(320, 120)
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
        try:
            details = self.stt.transcribe_details(pcm)
            text = details.text
            russian = getattr(details, "russian", None)
            english = getattr(details, "english", None)
            log.info(
                "STT RU: %r | EN: %r | источник: %s",
                russian.text if russian is not None else text,
                english.text if english is not None else "",
                getattr(details, "source", "ru"),
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Ошибка распознавания")
            self.notify(f"Ошибка распознавания: {exc}")
            beep(320, 120)
            self.set_state("idle")
            return

        if not text:
            self.notify("Не разобрал команду")
            beep(320, 120)
            self.set_state("idle")
            return

        log.info("STT: %r", text)
        text = canonicalize(
            strip_wake_word(text, self.cfg.get("wake_aliases") or [])
        )
        if not text:
            self.notify("Не разобрал команду")
            beep(320, 120)
            self.set_state("idle")
            return

        log.info("Распознано: %s", text)
        self.notify(f"«{text}»", "Услышал")
        try:
            self._process(text)
        finally:
            if self.mic is not None:
                self.mic.flush()
            self._preroll.clear()
            self.set_state("idle")

    def _record_command(self, pre_roll: list[bytes] | None = None) -> bytes:
        assert self.mic is not None
        audio_cfg = self.cfg["audio"]
        threshold = max(
            float(audio_cfg.get("energy_threshold", 0.01)),
            self._noise_floor * float(audio_cfg.get("noise_multiplier", 2.0)),
        )
        silence_limit = int(audio_cfg.get("silence_ms", 800))
        min_speech = int(audio_cfg.get("min_speech_ms", 250))
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

    def _process(self, text: str) -> str:
        self.set_state("thinking")
        try:
            result = self.router.handle(text)
            if result is not None:
                message = result.message
                if message:
                    self.notify(message)
                if not result.ok:
                    beep(320, 120)
                return message

            answer = self.brain.ask(text)
            answer = (answer or "").strip()
            if answer:
                self.notify(answer[:250])
            else:
                self.notify("Готово")
            return answer
        except Exception as exc:  # noqa: BLE001
            log.exception("Ошибка обработки команды")
            self.notify(f"Ошибка: {exc}"[:250])
            beep(320, 120)
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
        if self.mic is not None:
            self.mic.stop()
        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop()
            except Exception:
                pass
        if self.server is not None:
            self.server.stop()
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
        print(result)
        return 0
    return assistant.run()


if __name__ == "__main__":
    raise SystemExit(main())
