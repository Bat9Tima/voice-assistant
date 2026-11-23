"""Зрение по требованию: один снимок экрана и ответ модели.

Модуль не следит за экраном и не хранит кадры на диске: снимок живёт
в памяти, пока идёт запрос, и ещё до ``followup_ttl_sec`` секунд — чтобы
можно было задать отдельный вопрос по тому же кадру. После TTL кадр
считается устаревшим и не используется. Команда вызывается только явно,
фон и диалог сами зрение не запускают.
"""

from __future__ import annotations

import io
import logging
import re
import time
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
DEFAULT_FOLLOWUP_TTL = 60.0
# «посмотри на экран и скажи, какая ошибка» — служебное начало не вопрос.
_LEAD_RE = re.compile(r"^(?:и\s+)?(?:скажи|ответь|подскажи)\s+")
# «а что на экране?» — разговорная частица перед командой.
_FILLER_RE = re.compile(r"^(?:а|ну|и|так)\s+")

# «Посмотри на экран» без вопроса только запоминает кадр, а «что на экране»
# и «посмотри что на экране» остаются вопросами к модели.
_IMPERATIVE_WORDS = frozenset(
    {"посмотри", "посмотрите", "глянь", "гляньте", "взгляни", "взгляните"}
)
_QUESTION_WORDS = frozenset(
    {
        "что",
        "где",
        "какой",
        "какая",
        "какое",
        "какие",
        "видно",
        "видишь",
        "опиши",
        "расскажи",
    }
)

# Вопрос по уже снятому кадру: «где здесь кнопка», «что тут написано».
# Без указателей («здесь», «тут», «на экране», «кнопка») фраза уходит
# в обычный Q&A, поэтому «какая погода» зрение не перехватывает.
_FOLLOWUP_START_RE = re.compile(
    r"^(?:(?:а|ну|и|так)\s+)*"
    r"(?:что|где|куда|как|какой|какая|какое|какие|какого|какому|каком|какую|"
    r"каким|какими|каких|сколько|почему|зачем|кто|"
    r"есть|можно|покажи|подскажи|скажи|прочитай|объясни)\b",
    re.UNICODE,
)
_FOLLOWUP_PLACE_RE = re.compile(
    r"^(?:(?:а|ну|и|так)\s+)*(?:здесь|тут)\b",
    re.UNICODE,
)
_FOLLOWUP_MARKERS_RE = re.compile(
    r"\b(?:здесь|тут|на\s+экране|на\s+картинке|на\s+скриншоте|на\s+снимке|"
    r"на\s+фото|на\s+фотографии|на\s+скрине|"
    r"написан\w*|видн\w*|виден|"
    r"это|этот|эта|эти|этого|этой|этом|этих|"
    r"нем|ней|них|ним|ними|"
    r"кнопк\w*|ссылк\w*|пол[еяю]|окн\w*|ошибк\w*|текст\w*|"
    r"нажать|нажми|выбрать)\b",
    re.UNICODE,
)
# «а что в левом верхнем углу?» — вопрос по кадру без слов «экран» и «кнопка».
_FOLLOWUP_POSITION_RE = re.compile(
    r"\b(?:углу|угла|угле|слева|справа|сверху|снизу|наверху|внизу|на\s+фоне)\b",
    re.UNICODE,
)
_FOLLOWUP_QUESTION_RE = re.compile(
    r"\b(?:что|где|какой|какая|какое|какие|какого|какому|каком|какую|каким|"
    r"какими|каких|сколько|кто|почему|зачем)\b",
    re.UNICODE,
)


class VisionError(RuntimeError):
    """Не удалось снять экран или получить ответ по картинке."""


def match_trigger(text: str, triggers: list[str] | tuple[str, ...]) -> tuple[str | None, str]:
    """Возвращает сработавший триггер и остаток вопроса.

    Триггер учитывается только в начале фразы, поэтому «что на экране»
    срабатывает, а «включи видео на экране» — нет.
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
            return trigger, ""
        if normalized.startswith(trigger + " "):
            rest = normalized[len(trigger) :].strip()
            return trigger, _LEAD_RE.sub("", rest).strip()
    return None, ""


def match_request(text: str, triggers: list[str] | tuple[str, ...]) -> tuple[bool, str]:
    """Отделяет команду зрения от остального вопроса.

    Возвращает пару «сработало ли» и остаток вопроса.
    """
    trigger, rest = match_trigger(text, triggers)
    return trigger is not None, rest


def match_followup(text: str) -> bool:
    """Похоже ли на вопрос по уже снятому кадру.

    Срабатывает только на явные указатели («здесь», «тут», «на экране»,
    «написано», «кнопка») или на пространственную деталь в вопросе
    («что в левом верхнем углу»), поэтому «какая погода» не уходит в зрение.
    """
    normalized = normalize(text)
    if not normalized:
        return False
    if _FOLLOWUP_PLACE_RE.match(normalized):
        return True
    if _FOLLOWUP_POSITION_RE.search(normalized) and _FOLLOWUP_QUESTION_RE.search(normalized):
        return True
    if not _FOLLOWUP_START_RE.match(normalized):
        return False
    return bool(_FOLLOWUP_MARKERS_RE.search(normalized))


class ScreenVision:
    """Снимок основного экрана и вопросы по нему."""

    def __init__(self, cfg: dict[str, Any], llm: CloudLLM) -> None:
        section = (cfg or {}).get("vision") or {}
        self.enabled = bool(section.get("enabled", True))
        self.llm = llm
        self.model = str(section.get("model") or "").strip()
        self.max_width = int(section.get("max_width") or 1280)
        self.quality = int(section.get("jpeg_quality") or 75)
        self.max_tokens = int(section.get("max_tokens") or 2000)
        self.timeout = float(section.get("timeout_sec") or 60)
        self.followup_ttl = float(section.get("followup_ttl_sec") or DEFAULT_FOLLOWUP_TTL)
        self.prompt = str(section.get("prompt") or DEFAULT_PROMPT).strip()
        raw_triggers = section.get("triggers")
        if not isinstance(raw_triggers, list):
            raw_triggers = list(DEFAULT_TRIGGERS)
        self.triggers = [str(item).strip() for item in raw_triggers if str(item).strip()]
        self._pending: bytes | None = None
        self._pending_at = 0.0

    def available(self) -> bool:
        return bool(self.enabled and self.model and self.llm.available())

    def match(self, text: str) -> tuple[bool, str]:
        return match_request(text, self.triggers)

    def match_followup(self, text: str) -> bool:
        return match_followup(text)

    def wants_snapshot(self, text: str) -> bool:
        """Только запомнить кадр, без вопроса и запроса в модель."""
        trigger, rest = match_trigger(text, self.triggers)
        if trigger is None or rest:
            return False
        words = trigger.split()
        if not words or words[0] not in _IMPERATIVE_WORDS:
            return False
        return not any(word in _QUESTION_WORDS for word in words)

    def snapshot(self) -> bytes:
        """Снимает экран и запоминает кадр для следующих вопросов."""
        if not self.enabled:
            raise VisionError("зрение выключено")
        if not self.model:
            raise VisionError("не задана vision.model")
        if not self.llm.available():
            raise VisionError("зрение недоступно")
        image = self._capture()
        self._pending = image
        self._pending_at = time.monotonic()
        return image

    def has_pending(self) -> bool:
        """Есть ли свежий кадр для отдельного вопроса."""
        return self._pending_image() is not None

    def describe(self, request: str = "") -> str:
        """Снимает экран и возвращает короткий ответ модели."""
        image = self.snapshot()
        return self._answer(image, request)

    def describe_pending(self, request: str) -> str:
        """Отвечает по кадру, снятому предыдущей командой."""
        image = self._pending_image()
        if image is None:
            raise VisionError("нет свежего снимка экрана")
        if not self.enabled:
            raise VisionError("зрение выключено")
        if not self.model:
            raise VisionError("не задана vision.model")
        return self._answer(image, request)

    def _answer(self, image: bytes, request: str) -> str:
        return self.llm.answer_image(
            image,
            self._prompt_for(request),
            model=self.model,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

    def _pending_image(self) -> bytes | None:
        if self._pending is None:
            return None
        if time.monotonic() - self._pending_at > self.followup_ttl:
            self._pending = None
            return None
        return self._pending

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
    "DEFAULT_FOLLOWUP_TTL",
    "DEFAULT_PROMPT",
    "DEFAULT_TRIGGERS",
    "ScreenVision",
    "VisionError",
    "match_followup",
    "match_request",
    "match_trigger",
]
