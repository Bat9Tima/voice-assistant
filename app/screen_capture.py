"""Захват экрана для наблюдения: один кадр без Vision и сети.

Модуль не знает о моделях, диалоге и голосе. Снимок живёт в памяти,
на диск ничего не пишется. Это фундамент режима наблюдения: v1.0
сравнивает кадры, позже сюда добавятся активное окно и выбор монитора.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from PIL import Image, ImageGrab

DEFAULT_MAX_WIDTH = 1280


class ScreenCaptureError(RuntimeError):
    """Не удалось снять экран."""


@dataclass(frozen=True)
class Frame:
    """Один снимок экрана.

    ``window_title`` пока всегда ``None``: заголовок активного окна
    появится отдельным этапом, без изменения остальных полей.
    """

    image: Image.Image
    captured_at: float
    window_title: str | None = None

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


class ScreenCapture:
    """Снимок основного экрана через PIL."""

    def __init__(self, *, max_width: int = DEFAULT_MAX_WIDTH) -> None:
        self.max_width = max(0, int(max_width))

    def grab(self) -> Frame:
        """Снимает экран и возвращает кадр в памяти."""
        try:
            image = ImageGrab.grab()
            if image.mode != "RGB":
                image = image.convert("RGB")
            if self.max_width > 0 and image.width > self.max_width:
                height = round(image.height * self.max_width / image.width)
                image = image.resize((self.max_width, height), Image.Resampling.LANCZOS)
        except Exception as exc:  # noqa: BLE001
            raise ScreenCaptureError(f"не удалось снять экран: {exc}") from exc
        return Frame(image=image, captured_at=time.monotonic())


__all__ = [
    "DEFAULT_MAX_WIDTH",
    "Frame",
    "ScreenCapture",
    "ScreenCaptureError",
]
