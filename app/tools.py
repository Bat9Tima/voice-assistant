from __future__ import annotations

import re
import urllib.parse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

ParamType = Literal["string", "integer", "boolean"]

_SPACE_RE = re.compile(r"\s+", re.UNICODE)
_URL_MAX = 2048
_QUERY_MAX = 200


@dataclass(frozen=True)
class ToolParam:
    """Описание аргумента инструмента.

    Валидация общая для всех инструментов: тип, обязательность, диапазон,
    длина, enum. Это заменяет отдельные проверки под каждое действие.
    """

    name: str
    type: ParamType
    description: str = ""
    required: bool = False
    default: Any = None
    minimum: int | None = None
    maximum: int | None = None
    max_length: int | None = None
    enum: frozenset[str] | None = None
    lower: bool = False
    clamp: bool = False
    hint: str | None = None

    def prompt_hint(self) -> str:
        if self.hint:
            return self.hint
        if self.type == "integer":
            if self.minimum is not None and self.maximum is not None:
                return f"{self.minimum}..{self.maximum}"
            return "1"
        if self.type == "boolean":
            return "true"
        if self.enum:
            return '"' + "|".join(sorted(self.enum)) + '"'
        return '"..."'


@dataclass(frozen=True)
class Tool:
    """Действие, которое умеет выполнять ассистент."""

    name: str
    description: str
    params: tuple[ToolParam, ...] = ()
    risk: str = "low"
    validate: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None

    def prompt_line(self) -> str:
        if not self.params:
            return f"- {self.name} — {self.description}"
        args = ", ".join(
            f'"{param.name}": {param.prompt_hint()}' for param in self.params
        )
        return f"- {self.name}: {{{args}}} — {self.description}"


class ToolRegistry:
    """Набор инструментов и общая валидация их аргументов."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def prompt_lines(self) -> list[str]:
        return [tool.prompt_line() for tool in self._tools.values()]

    def validate(
        self, action: str, raw: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        tool = self._tools.get(action)
        if tool is None:
            return None
        args: dict[str, Any] = {}
        for param in tool.params:
            if param.name in raw:
                value = _coerce(param, raw[param.name])
                if value is None:
                    return None
                args[param.name] = value
            elif param.required:
                return None
            elif param.default is not None:
                args[param.name] = param.default
        if tool.validate is not None:
            return tool.validate(args)
        return args


def _coerce(param: ToolParam, value: Any) -> Any:
    if param.type == "string":
        text = _SPACE_RE.sub(" ", str(value or "")).strip()
        if param.lower:
            text = text.lower()
        if param.required and not text:
            return None
        if param.max_length is not None and len(text) > param.max_length:
            return None
        if param.enum is not None and text not in param.enum:
            return None
        return text
    if param.type == "integer":
        number = _as_int(value, None)
        if number is None:
            return None
        if param.minimum is not None and number < param.minimum:
            if not param.clamp:
                return None
            number = param.minimum
        if param.maximum is not None and number > param.maximum:
            if not param.clamp:
                return None
            number = param.maximum
        return number
    if param.type == "boolean":
        return _as_bool(value, None)
    return None


def _as_int(value: Any, default: int | None) -> int | None:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool | None) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "да"}:
            return True
        if text in {"false", "0", "no", "нет"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _validate_url(args: dict[str, Any]) -> dict[str, Any] | None:
    url = str(args.get("url") or "").strip()
    if not url or len(url) > _URL_MAX or re.search(r"\s", url):
        return None
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return {"url": url}


def build_default_registry(
    site_urls: Mapping[str, str],
    site_keys: Iterable[str],
) -> ToolRegistry:
    """Собирает инструменты, которые уже умеет ActionRouter.

    Описание и валидация живут здесь; промпт для LLM генерируется из
    этого же реестра. Новый инструмент добавляется в одном месте.
    """
    keys_list = list(site_keys)
    keys = ", ".join(keys_list)
    return ToolRegistry(
        [
            Tool(
                "open_site",
                f"открыть сайт. Ключи: {keys}.",
                (
                    ToolParam(
                        "site",
                        "string",
                        "ключ сайта",
                        required=True,
                        enum=frozenset(site_urls),
                        lower=True,
                        hint='"<ключ>"',
                    ),
                ),
            ),
            Tool(
                "goto",
                "открыть точный http(s)-адрес.",
                (
                    ToolParam(
                        "url",
                        "string",
                        "http(s)-адрес",
                        required=True,
                        max_length=_URL_MAX,
                        hint='"https://..."',
                    ),
                ),
                validate=_validate_url,
            ),
            Tool(
                "youtube",
                "найти и включить видео.",
                (
                    ToolParam(
                        "query",
                        "string",
                        "поисковый запрос",
                        required=True,
                        max_length=_QUERY_MAX,
                    ),
                    ToolParam(
                        "index",
                        "integer",
                        "номер результата",
                        default=1,
                        minimum=1,
                        maximum=20,
                        clamp=True,
                    ),
                ),
            ),
            Tool(
                "soundcloud",
                "найти и включить трек.",
                (
                    ToolParam(
                        "query",
                        "string",
                        "поисковый запрос",
                        required=True,
                        max_length=_QUERY_MAX,
                    ),
                    ToolParam(
                        "index",
                        "integer",
                        "номер результата",
                        default=1,
                        minimum=1,
                        maximum=20,
                        clamp=True,
                    ),
                ),
            ),
            Tool(
                "soundcloud_likes",
                "включить трек из лайкнутых SoundCloud. Без query — "
                "случайный лайкнутый; play=false — просто открыть лайкнутые.",
                (
                    ToolParam(
                        "query",
                        "string",
                        "название трека",
                        default="",
                        max_length=_QUERY_MAX,
                    ),
                    ToolParam("play", "boolean", "включить трек", default=True),
                ),
            ),
            Tool(
                "google",
                "поиск в Google.",
                (
                    ToolParam(
                        "query",
                        "string",
                        "поисковый запрос",
                        required=True,
                        max_length=_QUERY_MAX,
                    ),
                ),
            ),
            Tool("play", "продолжить воспроизведение."),
            Tool("pause", "пауза."),
            Tool("next", "следующий трек или видео."),
            Tool("prev", "предыдущий трек или видео."),
            Tool("newtab", "открыть новую вкладку."),
            Tool("closetab", "закрыть вкладку."),
            Tool(
                "tab",
                "переключить вкладку.",
                (
                    ToolParam(
                        "index",
                        "integer",
                        "номер вкладки",
                        required=True,
                        minimum=1,
                        maximum=99,
                    ),
                ),
            ),
            Tool("back", "назад."),
            Tool("forward", "вперёд."),
            Tool("reload", "обновить страницу."),
            Tool(
                "scroll",
                "прокрутить страницу.",
                (
                    ToolParam(
                        "direction",
                        "string",
                        "направление",
                        required=True,
                        enum=frozenset({"up", "down"}),
                        lower=True,
                    ),
                ),
            ),
            Tool(
                "volume",
                "системная громкость.",
                (
                    ToolParam(
                        "level",
                        "integer",
                        "уровень 0..100",
                        required=True,
                        minimum=0,
                        maximum=100,
                    ),
                ),
            ),
            Tool("volume_up", "сделать громче."),
            Tool("volume_down", "сделать тише."),
            Tool("mute", "выключить звук."),
            Tool("unmute", "включить звук."),
            Tool("status", "что открыто в браузере."),
        ]
    )
