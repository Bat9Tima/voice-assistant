"""Захват экрана для наблюдения: один кадр без Vision и сети.

Модуль не знает о моделях, диалоге и голосе. Снимок живёт в памяти,
на диск ничего не пишется. Это фундамент режима наблюдения: v1.0
сравнивает кадры, позже сюда добавятся активное окно и выбор монитора.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image, ImageGrab

log = logging.getLogger(__name__)

DEFAULT_MAX_WIDTH = 1280
TitleProvider = Callable[[], str | None]


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

    def __init__(
        self,
        *,
        max_width: int = DEFAULT_MAX_WIDTH,
        title_provider: TitleProvider | None = None,
    ) -> None:
        self.max_width = max(0, int(max_width))
        self.title_provider = title_provider

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
        return Frame(
            image=image,
            captured_at=time.monotonic(),
            window_title=self._window_title(),
        )

    def _window_title(self) -> str | None:
        if self.title_provider is None:
            return None
        try:
            title = self.title_provider()
        except Exception:  # noqa: BLE001
            log.debug("Не удалось прочитать заголовок окна", exc_info=True)
            return None
        return " ".join(title.split()) if title else None


__all__ = [
    "DEFAULT_MAX_WIDTH",
    "Frame",
    "ScreenCapture",
    "ScreenCaptureError",
    "TitleProvider",
]
