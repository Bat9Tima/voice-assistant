"""Зрение по требованию: один снимок экрана и ответ модели.

Модуль не следит за экраном и не хранит кадры: снимок живёт в памяти,
пока идёт запрос. Команда вызывается только явно, фон и диалог сами
зрение не запускают.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any

from .cloud_llm import CloudLLM
from .text import normalize

log = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Опиши, что видно на экране, в двух-трёх предложениях. "
    "Если на экране текст, код или ошибка — назови главное. "
    "Если экран чёрный или пустой, так и скажи."
)
DEFAULT_TRIGGERS = (
    "посмотри на экран",
    "посмотри на мой экран",
    "посмотри что на экране",
    "глянь на экран",
    "что на экране",
    "что видно на экране",
    "что ты видишь на экране",
    "что у меня на экране",
)
# «посмотри на экран и скажи, какая ошибка» — служебное начало не вопрос.
_LEAD_RE = re.compile(r"^(?:и\s+)?(?:скажи|ответь|подскажи)\s+")
# «а что на экране?» — разговорная частица перед командой.
_FILLER_RE = re.compile(r"^(?:а|ну|и|так)\s+")


class VisionError(RuntimeError):
    """Не удалось снять экран или получить ответ по картинке."""


def match_request(text: str, triggers: list[str] | tuple[str, ...]) -> tuple[bool, str]:
    """Отделяет команду зрения от остального вопроса.

    Триггер учитывается только в начале фразы, поэтому «что на экране»
    срабатывает, а «включи видео на экране» — нет. Возвращает пару
    «сработало ли» и остаток вопроса.
    """
    normalized = normalize(text)
    while True:
        stripped = _FILLER_RE.sub("", normalized, count=1)
        if stripped == normalized:
            break
        normalized = stripped
    prepared = sorted(
        {normalize(trigger) for trigger in triggers if normalize(trigger)},
        key=len,
        reverse=True,
    )
    for trigger in prepared:
        if normalized == trigger:
            return True, ""
        if normalized.startswith(trigger + " "):
            rest = normalized[len(trigger) :].strip()
            return True, _LEAD_RE.sub("", rest).strip()
    return False, ""


class ScreenVision:
    """Снимок основного экрана и один вопрос к модели."""

    def __init__(self, cfg: dict[str, Any], llm: CloudLLM) -> None:
        section = (cfg or {}).get("vision") or {}
        self.enabled = bool(section.get("enabled", True))
        self.llm = llm
        self.model = str(section.get("model") or "").strip()
        self.max_width = int(section.get("max_width") or 1280)
        self.quality = int(section.get("jpeg_quality") or 75)
        self.max_tokens = int(section.get("max_tokens") or 2000)
        self.timeout = float(section.get("timeout_sec") or 60)
        self.prompt = str(section.get("prompt") or DEFAULT_PROMPT).strip()
        raw_triggers = section.get("triggers")
        if not isinstance(raw_triggers, list):
            raw_triggers = list(DEFAULT_TRIGGERS)
        self.triggers = [str(item).strip() for item in raw_triggers if str(item).strip()]

    def available(self) -> bool:
        return bool(self.enabled and self.model and self.llm.available())

    def match(self, text: str) -> tuple[bool, str]:
        return match_request(text, self.triggers)

    def describe(self, request: str = "") -> str:
        """Снимает экран и возвращает короткий ответ модели."""
        if not self.enabled:
            raise VisionError("зрение выключено")
        if not self.model:
            raise VisionError("не задана vision.model")
        image = self._capture()
        return self.llm.answer_image(
            image,
            self._prompt_for(request),
            model=self.model,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

    def _prompt_for(self, request: str) -> str:
        request = (request or "").strip()
        if not request:
            return self.prompt
        return (
            f"Вопрос пользователя про изображение экрана: {request}. "
            "Ответь по картинке, коротко, в двух-трёх предложениях. "
            "Если ответа на экране нет, так и скажи."
        )

    def _capture(self) -> bytes:
        """JPEG основного экрана в памяти, без записи на диск."""
        try:
            from PIL import Image, ImageGrab

            image = ImageGrab.grab()
            if image.mode != "RGB":
                image = image.convert("RGB")
            if self.max_width > 0 and image.width > self.max_width:
                height = round(image.height * self.max_width / image.width)
                image = image.resize((self.max_width, height), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self.quality, optimize=True)
            return buffer.getvalue()
        except Exception as exc:  # noqa: BLE001
            raise VisionError(f"не удалось снять экран: {exc}") from exc


__all__ = [
    "DEFAULT_PROMPT",
    "DEFAULT_TRIGGERS",
    "ScreenVision",
    "VisionError",
    "match_request",
]
