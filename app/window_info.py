"""Заголовок активного окна Windows без сторонних библиотек.

Нужен наблюдению как второй сигнал: картинка может почти не измениться,
а окно уже другое. Ошибки чтения не должны ломать захват экрана, поэтому
наружу отдаётся ``None``.
"""

from __future__ import annotations

import ctypes
import logging

log = logging.getLogger(__name__)


def normalize_title(title: str | None) -> str | None:
    """Схлопывает пробелы и пустой заголовок превращает в ``None``."""
    if not title:
        return None
    cleaned = " ".join(title.split())
    return cleaned or None


def _read_foreground_title() -> str | None:
    """Читает заголовок активного окна через Win32."""
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    user32.GetWindowTextLengthW.argtypes = [ctypes.c_void_p]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    length = int(user32.GetWindowTextLengthW(hwnd))
    if length <= 0:
        return None
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_int,
    ]
    user32.GetWindowTextW.restype = ctypes.c_int
    written = int(user32.GetWindowTextW(hwnd, buffer, length + 1))
    if written <= 0:
        return None
    return buffer.value


def active_window_title() -> str | None:
    """Возвращает заголовок активного окна или ``None``."""
    try:
        return normalize_title(_read_foreground_title())
    except Exception:  # noqa: BLE001
        log.debug("Не удалось прочитать заголовок активного окна", exc_info=True)
        return None


__all__ = ["active_window_title", "normalize_title"]
