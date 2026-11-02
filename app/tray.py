from __future__ import annotations

import logging
import os
import threading
from typing import Any

from .config import LOGS_DIR

log = logging.getLogger(__name__)

# Одно и то же приложение, тег и группа: новое уведомление заменяет старое,
# а не встаёт за ним в очередь Explorer.
_TOAST_APP = "Voice Assistant"
_TOAST_TAG = "assistant"
_TOAST_GROUP = "voice-assistant"

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


class TrayUI:
    """Иконка в системном трее и всплывающие уведомления."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.icon = None
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._notify_lock = threading.Lock()
        self._toaster = None
        self._toast_class = None

    def _init_toasts(self) -> None:
        try:
            from windows_toasts import Toast, WindowsToaster

            self._toaster = WindowsToaster(_TOAST_APP)
            self._toast_class = Toast
            log.info("Уведомления: Windows Toast")
        except Exception as exc:  # noqa: BLE001
            log.debug("Windows Toast недоступен, остаюсь на pystray: %s", exc)

    def start(self) -> None:
        self._init_toasts()
        try:
            import pystray
        except Exception as exc:  # noqa: BLE001
            log.warning("pystray недоступен, работаю без трея: %s", exc)
            return

        menu = pystray.Menu(
            pystray.MenuItem("Слушать сейчас", lambda: self.app.trigger_listen(), default=True),
            pystray.MenuItem("Пауза / продолжить", lambda: self.app.toggle_pause()),
            pystray.MenuItem("Статус", lambda: self.app.show_status()),
            pystray.MenuItem("Открыть лог", self._open_log),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Выход", lambda: self.app.shutdown()),
        )
        self.icon = pystray.Icon(
            "voice-assistant",
            _make_image(COLORS["starting"]),
            TITLES["starting"],
            menu,
        )

        def setup(icon) -> None:
            icon.visible = True
            self._ready.set()

        def run() -> None:
            try:
                assert self.icon is not None
                self.icon.run(setup=setup)
            except Exception:  # noqa: BLE001
                log.exception("Трей остановился с ошибкой")

        self._thread = threading.Thread(target=run, daemon=True, name="tray")
        self._thread.start()
        self._ready.wait(timeout=3)
        log.info("Трей запущен")

    def set_state(self, state: str) -> None:
        if self.icon is None:
            return
        try:
            self.icon.icon = _make_image(COLORS.get(state, COLORS["idle"]))
            self.icon.title = TITLES.get(state, TITLES["idle"])
        except Exception:  # noqa: BLE001
            log.debug("Не удалось обновить иконку трея", exc_info=True)

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
                log.warning(
                    "Windows Toast не сработал, откатываюсь на pystray: %s", exc
                )
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

        icon = self.icon
        if icon is None or not self._ready.is_set():
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
        icon = self.icon
        if icon is None or not self._ready.is_set():
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

    def stop(self) -> None:
        if self.icon is not None:
            try:
                self.icon.stop()
            except Exception:  # noqa: BLE001
                pass
