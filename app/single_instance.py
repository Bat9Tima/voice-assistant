from __future__ import annotations

import ctypes
import logging
import os

log = logging.getLogger(__name__)

_MUTEX_NAME = "Local\\VoiceAssistantSingleton_v1"
_handle = None


def acquire() -> object | None:
    """Не даёт запустить второй экземпляр ассистента."""
    global _handle
    if os.name != "nt":
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    handle = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    if not handle:
        return True
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        log.warning("Ассистент уже запущен")
        return None
    _handle = handle
    return handle
