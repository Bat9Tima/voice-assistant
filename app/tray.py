from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from typing import Any

from .config import LOGS_DIR

log = logging.getLogger(__name__)

# Одно и то же приложение, тег и группа: новое уведомление заменяет старое,
# а не встаёт за ним в очередь Explorer.
_TOAST_APP = "Voice Assistant"
_TOAST_TAG = "assistant"
_TOAST_GROUP = "voice-assistant"

# NIM_ADD — единственная операция Shell_NotifyIcon, без которой иконки нет.
# NIM_MODIFY (1) и NIM_DELETE (2) — обновление/удаление, их провал не фатален.
_NIM_ADD = 0

# Маркер на обёртках guard'ов: не даёт завернуть функцию дважды.
_GUARD_MARK = "_voice_assistant_guarded"

COLORS = {
    "starting": "#7f8c8d",
    "idle": "#4a90e2",
    "listening": "#2ecc71",
    "thinking": "#f39c12",
    "paused": "#95a5a6",
    "error": "#e74c3c",
}

TITLES = {
    "starting": "Ассистент: запуск",
    "idle": "Ассистент: ожидание",
    "listening": "Ассистент: слушаю",
    "thinking": "Ассистент: думаю",
    "paused": "Ассистент: пауза",
    "error": "Ассистент: ошибка",
}


def _make_image(color: str, size: int = 64):
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((2, 2, size - 2, size - 2), fill=color)
    white = "#ffffff"
    draw.rounded_rectangle((26, 14, 38, 38), radius=6, fill=white)
    draw.arc((18, 26, 46, 52), start=0, end=180, fill=white, width=4)
    draw.line((32, 52, 32, 58), fill=white, width=4)
    draw.line((23, 58, 41, 58), fill=white, width=4)
    return image


def _pystray_win32() -> Any:
    """Возвращает модуль ``pystray._util.win32`` (или None).

    Обычный ``from pystray._util import win32`` ломается, если тесты подменили
    ``sys.modules['pystray']`` фейком: подпакет при этом уже импортирован,
    поэтому пробуем достать его напрямую из кеша модулей.
    """
    try:
        from pystray._util import win32

        return win32
    except Exception:  # noqa: BLE001
        module = sys.modules.get("pystray._util")
        if module is None:
            return None
        return getattr(module, "win32", None)


def _last_win_error() -> int:
    """Код последней ошибки Win32 (0, если ctypes его не сохранил)."""
    try:
        code = int(ctypes.get_last_error() or 0)
    except Exception:  # noqa: BLE001
        code = 0
    if code:
        return code
    try:
        get_last_error = ctypes.windll.kernel32.GetLastError
        get_last_error.restype = ctypes.c_ulong
        return int(get_last_error() or 0)
    except Exception:  # noqa: BLE001
        return 0


def _error_text(error: object) -> str:
    """Текст ошибки для логов и ``status()['error']``."""
    if isinstance(error, BaseException):
        return str(error) or error.__class__.__name__
    return str(error)


class TrayUI:
    """Иконка в системном трее и всплывающие уведомления.

    Иконка pystray живёт в отдельном потоке и может умереть: например,
    ``ChangeWindowMessageFilterEx`` или ``Shell_NotifyIcon(NIM_ADD)`` могут
    вернуть отказ (WinError 5) при запуске с ограниченным токеном. Тогда
    setUp-колбэк падает, и без watchdog иконка пропадает до перезапуска
    процесса. Watchdog пересоздаёт иконку с backoff, а ``status()`` показывает
    состояние снаружи (для /status и диагностики).
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self.icon = None
        self._ready = threading.Event()
        self._icon_ok = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_started_at = 0.0
        self._generation = 0
        self._lock = threading.RLock()
        self._notify_lock = threading.Lock()
        self._toaster = None
        self._toast_class = None
        self._error: str | None = None
        self._available = False
        self._restarts = 0
        self._state = "starting"
        self._pystray = None
        self._stop_event = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._unavailable_notified = False
        # Поколения, для которых run() упал до готовности: их setup не должен
        # показывать иконку, когда мы освобождаем застрявший setup-поток.
        self._setup_blocked: set[int] = set()
        # Настройки; в тестах их можно уменьшать.
        self._startup_timeout = 3.0
        self._watchdog_interval = 5.0
        self._restart_initial_delay = 5.0
        self._restart_max_delay = 60.0
        self._thread_join_timeout = 5.0
        self._watchdog_join_timeout = 2.0

    # ------------------------------------------------------------------ guards
    def _init_toasts(self) -> None:
        try:
            from windows_toasts import Toast, WindowsToaster

            self._toaster = WindowsToaster(_TOAST_APP)
            self._toast_class = Toast
            log.info("Уведомления: Windows Toast")
        except Exception as exc:  # noqa: BLE001
            log.debug("Windows Toast недоступен, остаюсь на pystray: %s", exc)

    @staticmethod
    def _guard_message_filter() -> None:
        """Делает вызов ``ChangeWindowMessageFilterEx`` безопасным.

        pystray вызывает эту функцию в ``_win32._create_window`` без защиты:
        при WinError 5 (запуск с повышенными правами/ограниченным токеном)
        исключение рвёт создание окна трея. Оборачиваем настоящую функцию:
        при ``OSError`` пишем debug-лог и делаем вид, что фильтр разрешён,
        чтобы окно создалось. Если ОС разрешает вызов, фильтр реально
        регистрируется — WM_TASKBARCREATED продолжает доходить.

        Идемпотентно: повторный вызов не заворачивает обёртку в обёртку.
        """
        win32 = _pystray_win32()
        if win32 is None:
            log.debug("Не удалось загрузить win32-хелперы pystray")
            return
        original = getattr(win32, "ChangeWindowMessageFilterEx", None)
        if original is None or getattr(original, _GUARD_MARK, None) is True:
            return

        # Имя сохранено ради совместимости со старым тестом.
        def _allow_message_filter(hwnd, message, action, change_filter_struct):
            try:
                return original(hwnd, message, action, change_filter_struct)
            except OSError as exc:  # PermissionError, ctypes.WinError, ...
                log.debug(
                    "ChangeWindowMessageFilterEx недоступен (%s); "
                    "подписка на WM_TASKBARCREATED не подтверждена",
                    exc,
                )
                return True

        setattr(_allow_message_filter, _GUARD_MARK, True)
        try:
            win32.ChangeWindowMessageFilterEx = _allow_message_filter
            log.debug("Проверка ChangeWindowMessageFilterEx включена")
        except Exception as exc:  # noqa: BLE001
            log.debug("Не удалось включить обход ChangeWindowMessageFilterEx: %s", exc)

    # Алиас для кода/тестов, написанных до появления guard'а.
    _patch_message_filter = _guard_message_filter

    @staticmethod
    def _guard_shell_notify_icon() -> None:
        """Проверяет результат ``Shell_NotifyIcon``.

        pystray игнорирует BOOL от Shell_NotifyIcon: если NIM_ADD не добавил
        иконку (например, процесс запущен с ограниченным токеном), окно
        создаётся, ``_mark_ready()`` вызывается, и наш код ложно рапортует
        «Трей запущен». Оборачиваем функцию: при FALSE и NIM_ADD поднимаем
        ``OSError``, чтобы setup-колбэк pystray завершился ошибкой и watchdog
        увидел провал. Для NIM_MODIFY/NIM_DELETE (тосты, обновление) только
        debug-лог: их сбой не должен ломать уведомления.

        Идемпотентно: повторный вызов не заворачивает обёртку в обёртку.
        """
        win32 = _pystray_win32()
        if win32 is None:
            log.debug("Не удалось загрузить win32-хелперы pystray")
            return
        original = getattr(win32, "Shell_NotifyIcon", None)
        if original is None or getattr(original, _GUARD_MARK, None) is True:
            return

        def _guarded_shell_notify_icon(dw_message, data):
            result = original(dw_message, data)
            if result:
                return result
            code = _last_win_error()
            if dw_message == _NIM_ADD:
                message = f"Shell_NotifyIcon(NIM_ADD) не добавил иконку: WinError {code}"
                log.warning(message)
                raise OSError(code, message)
            log.debug("Shell_NotifyIcon(%s) вернул FALSE (WinError %s)", dw_message, code)
            return False

        setattr(_guarded_shell_notify_icon, _GUARD_MARK, True)
        try:
            win32.Shell_NotifyIcon = _guarded_shell_notify_icon
            log.debug("Проверка результата Shell_NotifyIcon включена")
        except Exception as exc:  # noqa: BLE001
            log.debug("Не удалось включить проверку Shell_NotifyIcon: %s", exc)

    # ------------------------------------------------------------------- start
    def start(self) -> None:
        self._init_toasts()
        self._guard_message_filter()
        self._guard_shell_notify_icon()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                log.debug("Трей уже запущен")
                return
        try:
            import pystray
        except Exception as exc:  # noqa: BLE001
            self._record_error(exc)
            log.warning("pystray недоступен, работаю без трея: %s", exc)
            self._notify_unavailable(exc)
            return

        with self._lock:
            self._pystray = pystray
        if self._spawn_icon() and self._wait_ready():
            log.info("Трей запущен")
        else:
            log.warning("Трей не запустился: %s", self._error or "таймаут")
            self._notify_unavailable(self._error or "трей не запустился")
        self._start_watchdog()

    def _spawn_icon(self) -> bool:
        """Пересоздаёт иконку и её поток. True — если поток запущен."""
        pystray = self._pystray
        if pystray is None or self._stop_event.is_set():
            return False
        self._teardown_icon()
        with self._lock:
            self._generation += 1
            generation = self._generation
            state = self._state
        try:
            icon = pystray.Icon(
                "voice-assistant",
                _make_image(COLORS.get(state, COLORS["starting"])),
                TITLES.get(state, TITLES["starting"]),
                self._make_menu(pystray),
            )
        except Exception as exc:  # noqa: BLE001
            self._record_error(exc)
            log.exception("Не удалось создать иконку трея")
            return False
        thread = threading.Thread(
            target=self._icon_loop,
            args=(generation, icon),
            daemon=True,
            name="tray",
        )
        with self._lock:
            if self._stop_event.is_set():
                return False
            self.icon = icon
            self._ready.clear()
            self._icon_ok.clear()
            self._thread = thread
            self._thread_started_at = time.monotonic()
        thread.start()
        return True

    def _make_menu(self, pystray: Any) -> Any:
        return pystray.Menu(
            pystray.MenuItem("Слушать сейчас", lambda: self.app.trigger_listen(), default=True),
            pystray.MenuItem("Пауза / продолжить", lambda: self.app.toggle_pause()),
            pystray.MenuItem("Статус", lambda: self.app.show_status()),
            pystray.MenuItem("Открыть лог", self._open_log),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Выход", lambda: self.app.shutdown()),
        )

    def _wait_ready(self, timeout: float | None = None) -> bool:
        """Ждёт, пока иконка реально появится (setup без ошибки)."""
        limit = self._startup_timeout if timeout is None else timeout
        deadline = time.monotonic() + limit
        while True:
            if self._icon_ok.is_set():
                return True
            with self._lock:
                thread = self._thread
                ready = self._ready.is_set()
            if ready:
                # setup отработал, но иконка не показалась — провал.
                return self._icon_ok.is_set()
            if thread is None or not thread.is_alive():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._ready.wait(min(remaining, 0.05))

    def _icon_loop(self, generation: int, icon: Any) -> None:
        def setup(shown_icon: Any) -> None:
            with self._lock:
                blocked = generation in self._setup_blocked
                self._setup_blocked.discard(generation)
            if blocked:
                # run() уже упал: показывать иконку для мёртвого поколения нельзя.
                try:
                    shown_icon.stop()
                except Exception:  # noqa: BLE001
                    log.debug("Не удалось остановить лишнюю иконку трея", exc_info=True)
                return
            failure: Exception | None = None
            try:
                shown_icon.visible = True
            except Exception as exc:  # noqa: BLE001
                failure = exc
            with self._lock:
                current = (
                    self._generation == generation
                    and self.icon is icon
                    and not self._stop_event.is_set()
                )
                if current and failure is None:
                    self._available = True
                    self._error = None
                    self._icon_ok.set()
                if current and failure is not None:
                    self._error = _error_text(failure)
                if current:
                    self._ready.set()
            if failure is not None and current:
                log.warning("Трей не смог показать иконку: %s", failure)
            if failure is not None or not current:
                try:
                    shown_icon.stop()
                except Exception:  # noqa: BLE001
                    log.debug("Не удалось остановить лишнюю иконку трея", exc_info=True)

        try:
            icon.run(setup=setup)
        except Exception as exc:  # noqa: BLE001
            self._record_error(exc)
            log.exception("Трей остановился с ошибкой")
        finally:
            with self._lock:
                current = self.icon is icon
                shown = self._icon_ok.is_set()
            if current and shown and not self._stop_event.is_set():
                log.warning("Трей-поток завершился неожиданно")
                with self._lock:
                    if self._error is None:
                        self._error = "трей-поток завершился"
            self._release_setup_thread(generation, icon)

    def _release_setup_thread(self, generation: int, icon: Any) -> None:
        """Освобождает setup-поток pystray, если run() упал до _mark_ready().

        pystray запускает setup-поток, который ждёт ``_mark_ready()``; если
        ``_run()`` падает раньше (создание окна), поток (не daemon) висит
        навсегда. Для watchdog это означало бы утечку потока на каждую
        попытку перезапуска, поэтому аккуратно будим его и гасим флаг.
        """
        mark_ready = getattr(icon, "_mark_ready", None)
        setup_thread = getattr(icon, "_setup_thread", None)
        if mark_ready is None or getattr(icon, "_running", False):
            return
        if setup_thread is None or not setup_thread.is_alive():
            return
        with self._lock:
            if self._ready.is_set():
                return
            self._setup_blocked.add(generation)
        try:
            mark_ready()
        except Exception:  # noqa: BLE001
            log.debug("Не удалось освободить setup-поток pystray", exc_info=True)
        finally:
            try:
                icon._running = False
            except Exception:  # noqa: BLE001
                pass
        log.debug("Освободил setup-поток pystray после сбоя запуска")

    # --------------------------------------------------------------- watchdog
    def _start_watchdog(self) -> None:
        with self._lock:
            if self._pystray is None or self._stop_event.is_set():
                return
            thread = self._watchdog_thread
            if thread is not None and thread.is_alive():
                return
            thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="tray-watchdog")
            self._watchdog_thread = thread
        thread.start()

    def _tray_unhealthy(self) -> bool:
        """Иконка (пере)запущена, но трей не работает?"""
        if self._stop_event.is_set():
            return False
        with self._lock:
            thread = self._thread
            started_at = self._thread_started_at
            ready = self._ready.is_set()
            shown = self._icon_ok.is_set()
        if thread is None or not thread.is_alive():
            return True
        if shown:
            return False
        if ready:
            # setup отработал, но иконки нет (например, Shell_NotifyIcon FALSE).
            return True
        hang_limit = max(self._startup_timeout * 2.0, self._watchdog_interval)
        return (time.monotonic() - started_at) > hang_limit

    def _watchdog_loop(self) -> None:
        delay = self._restart_initial_delay
        while not self._stop_event.wait(self._watchdog_interval):
            if self._pystray is None:
                return
            if not self._tray_unhealthy():
                continue
            with self._lock:
                self._restarts += 1
                attempt = self._restarts
            log.warning("Трей остановился, перезапускаю (попытка %d)", attempt)
            if self._spawn_icon() and self._wait_ready():
                if self._stop_event.is_set():
                    return
                log.info("Трей перезапущен (попытка %d)", attempt)
                delay = self._restart_initial_delay
                continue
            if self._stop_event.is_set():
                return
            log.warning(
                "Перезапуск трея не удался (попытка %d); следующая через %.0f с",
                attempt,
                delay,
            )
            self._notify_unavailable(self._error or "трей не запустился")
            if self._stop_event.wait(delay):
                return
            delay = min(delay * 2.0, self._restart_max_delay)

    # ------------------------------------------------------------------ state
    def set_state(self, state: str) -> None:
        with self._lock:
            self._state = state
            icon = self.icon
        if icon is None:
            return
        try:
            icon.icon = _make_image(COLORS.get(state, COLORS["idle"]))
            icon.title = TITLES.get(state, TITLES["idle"])
        except Exception:  # noqa: BLE001
            log.debug("Не удалось обновить иконку трея", exc_info=True)

    def status(self) -> dict[str, Any]:
        """Состояние трея для /status и диагностики."""
        with self._lock:
            thread = self._thread
            return {
                "available": bool(self._available),
                "running": bool(
                    thread is not None and thread.is_alive() and not self._stop_event.is_set()
                ),
                "error": self._error,
                "restarts": int(self._restarts),
            }

    def _record_error(self, error: object) -> None:
        with self._lock:
            self._error = _error_text(error) or None

    def _notify_unavailable(self, error: object) -> None:
        """Один раз сообщает, что иконки нет (Toast напрямую, без трея)."""
        with self._lock:
            if self._unavailable_notified:
                return
            self._unavailable_notified = True
        details = _error_text(error) if isinstance(error, BaseException) else str(error)
        text = f"Иконка в трее недоступна: {details}. Работает Ctrl+Alt+space"
        log.warning(text)
        if not self._show_toast(text, "Ассистент"):
            log.debug("Уведомление о недоступности трея не показано (Toast нет)")

    # ------------------------------------------------------------- notifications
    def _show_toast(self, message: str, title: str) -> bool:
        toaster = self._toaster
        toast_class = self._toast_class
        if toaster is None or toast_class is None:
            return False
        with self._notify_lock:
            try:
                # Чистим историю этого приложения: иначе Windows держит
                # предыдущие тосты и показывает их по очереди.
                toaster.clear_toasts()
                toast = toast_class(
                    [title[:60], message[:250]],
                    group=_TOAST_GROUP,
                    suppress_popup=False,
                )
                toast.tag = _TOAST_TAG
                toaster.show_toast(toast)
                return True
            except Exception as exc:  # noqa: BLE001
                log.warning("Windows Toast не сработал, откатываюсь на pystray: %s", exc)
                self._toaster = None
                self._toast_class = None
                return False

    def clear(self) -> None:
        """Сбрасывает текущее уведомление, чтобы новое не ждало очереди."""
        toaster = self._toaster
        if toaster is not None:
            with self._notify_lock:
                try:
                    toaster.clear_toasts()
                    return
                except Exception as exc:  # noqa: BLE001
                    log.debug("Не удалось очистить Windows Toast: %s", exc)

        with self._lock:
            icon = self.icon
            ready = self._ready.is_set()
        if icon is None or not ready:
            return
        with self._notify_lock:
            try:
                icon.remove_notification()
            except Exception:  # noqa: BLE001
                log.debug("Не удалось убрать уведомление", exc_info=True)

    def notify(self, message: str, title: str = "Ассистент") -> None:
        message = (message or "").strip()
        if not message:
            return
        log.info("Уведомление: %s", message)
        if self._show_toast(message, title):
            return
        with self._lock:
            icon = self.icon
            ready = self._ready.is_set()
        if icon is None or not ready:
            return
        with self._notify_lock:
            try:
                # Запасной путь для старых версий Windows.
                icon.remove_notification()
                icon.notify(message[:250], title[:60])
            except Exception:  # noqa: BLE001
                log.debug("Не удалось показать уведомление", exc_info=True)

    @staticmethod
    def _open_log() -> None:
        path = LOGS_DIR / "assistant.log"
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            log.exception("Не удалось открыть лог")

    # ------------------------------------------------------------------- stop
    def _teardown_icon(self) -> None:
        with self._lock:
            icon = self.icon
            thread = self._thread
            self.icon = None
            self._thread = None
            self._thread_started_at = 0.0
        if icon is not None:
            try:
                icon.stop()
            except Exception:  # noqa: BLE001
                log.debug("Не удалось остановить иконку трея", exc_info=True)
        if thread is not None and thread is not threading.current_thread():
            try:
                thread.join(timeout=self._thread_join_timeout)
            except Exception:  # noqa: BLE001
                log.debug("Не удалось дождаться трей-потока", exc_info=True)
            if thread.is_alive():
                log.warning("Трей-поток не завершился за %.0f с", self._thread_join_timeout)

    def stop(self) -> None:
        """Останавливает watchdog, иконку и трей-поток. Не бросает исключений."""
        self._stop_event.set()
        with self._lock:
            watchdog = self._watchdog_thread
            self._watchdog_thread = None
        if watchdog is not None and watchdog is not threading.current_thread():
            try:
                watchdog.join(timeout=self._watchdog_join_timeout)
            except Exception:  # noqa: BLE001
                log.debug("Не удалось дождаться watchdog трея", exc_info=True)
            if watchdog.is_alive():
                log.debug("Watchdog трея не завершился, но перезапуски запрещены")
        self._teardown_icon()
