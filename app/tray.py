from __future__ import annotations

import logging
import os
import threading
from typing import Any

from .config import LOGS_DIR

log = logging.getLogger(__name__)

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

    def start(self) -> None:
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

    def notify(self, message: str, title: str = "Ассистент") -> None:
        message = (message or "").strip()
        if not message:
            return
        log.info("Уведомление: %s", message)
        if self.icon is None or not self._ready.is_set():
            return
        try:
            self.icon.notify(message[:250], title[:60])
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
