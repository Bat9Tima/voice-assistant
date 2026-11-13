from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .tools import _QUERY_MAX, _URL_MAX, Tool, ToolParam, _validate_url

# ---------------------------------------------------------------------------
# Каталог действий
#
# Одна запись описывает действие ассистента и связывает слои:
#   * intents — какие интенты из intents.json его вызывают;
#   * tool    — какой инструмент видит локальный планировщик и агент;
#   * cli     — какая команда browser.cmd соответствует действию.
#
# Из этого списка собирается реестр инструментов, а тесты проверяют, что
# интенты, CLI и README не разошлись с каталогом.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionSpec:
    """Описание одного действия ассистента."""

    name: str
    title: str
    description: str
    category: str
    examples: tuple[str, ...] = ()
    intents: tuple[str, ...] = ()
    tool: str | None = None
    cli: tuple[str, ...] = ()
    result: str = ""
    risk: str = "low"
    documented: bool = True
    doc_example: str = ""


ACTIONS: tuple[ActionSpec, ...] = (
    # ------------------------------------------------------------- навигация
    ActionSpec(
        "open_site",
        "Открыть сайт",
        "открыть сайт из списка известных",
        "navigation",
        examples=("открой ютуб", "открой гитхаб"),
        intents=("open_site",),
        tool="open_site",
        cli=("open",),
        result="url",
        doc_example="открой ютуб",
    ),
    ActionSpec(
        "goto",
        "Открыть адрес",
        "открыть точный http(s)-адрес или домен",
        "navigation",
        examples=("открой example.com", "открой reddit.com"),
        tool="goto",
        cli=("goto", "open"),
        result="url, title",
        doc_example="открой example.com",
    ),
    ActionSpec(
        "back",
        "Назад",
        "вернуться на предыдущую страницу",
        "navigation",
        examples=("назад", "вернись назад"),
        intents=("back",),
        tool="back",
        cli=("back",),
        result="url",
        doc_example="назад",
    ),
    ActionSpec(
        "forward",
        "Вперёд",
        "перейти на следующую страницу",
        "navigation",
        examples=("вперёд", "страница вперёд"),
        intents=("forward",),
        tool="forward",
        cli=("forward",),
        result="url",
        doc_example="вперёд",
    ),
    ActionSpec(
        "reload",
        "Обновить страницу",
        "перезагрузить текущую страницу",
        "navigation",
        examples=("обнови", "обнови страницу"),
        intents=("reload",),
        tool="reload",
        cli=("reload",),
        result="url",
        doc_example="обнови",
    ),
    # ---------------------------------------------------------------- поиск
    ActionSpec(
        "google_search",
        "Поиск в Google",
        "найти запрос в Google и показать выдачу",
        "search",
        examples=("загугли погоду", "найди в гугле рецепт борща"),
        intents=("google_search",),
        tool="google",
        cli=("google",),
        result="title, url",
        doc_example="загугли",
    ),
    ActionSpec(
        "open_result",
        "Открыть результат",
        "открыть результат из выдачи Google по номеру",
        "search",
        examples=("открой первую ссылку", "открой ссылку 2"),
        intents=("open_result", "open_result_context"),
        cli=("search_result",),
        result="title, url",
        doc_example="открой первую ссылку",
    ),
    ActionSpec(
        "repeat_search",
        "Повторить поиск",
        "повторить последний поиск или включение",
        "search",
        examples=("повтори", "ещё раз"),
        intents=("repeat_search",),
        result="title, url",
        doc_example="повтори",
    ),
    ActionSpec(
        "next_result",
        "Следующий результат",
        "открыть следующий результат последнего поиска",
        "search",
        examples=("ещё один", "следующий результат"),
        intents=("next_result",),
        result="title, url",
        doc_example="ещё один",
    ),
    # ------------------------------------------------------------------ медиа
    ActionSpec(
        "youtube_search",
        "Поиск на YouTube",
        "найти и включить видео на YouTube",
        "media",
        examples=("включи на ютубе lo-fi", "найди на ютубе котиков"),
        intents=("youtube_search",),
        tool="youtube",
        cli=("youtube",),
        result="title, url",
        doc_example="включи на ютубе",
    ),
    ActionSpec(
        "soundcloud_open",
        "Открыть SoundCloud",
        "открыть главную страницу SoundCloud",
        "media",
        examples=("открой саундклауд", "перейди на саундклауд"),
        intents=("soundcloud_open",),
        result="url",
        doc_example="открой саундклауд",
    ),
    ActionSpec(
        "soundcloud_search",
        "Поиск на SoundCloud",
        "найти и включить трек на SoundCloud",
        "media",
        examples=("включи на саундклауде lofi", "найди трек на саундклауде"),
        intents=("soundcloud_search",),
        tool="soundcloud",
        cli=("soundcloud",),
        result="title, url",
        doc_example="включи на саундклауде",
    ),
    ActionSpec(
        "soundcloud_likes_open",
        "Открыть лайки SoundCloud",
        "открыть страницу лайкнутых треков SoundCloud",
        "media",
        examples=("открой лайкнутые треки на саундклауде",),
        intents=("soundcloud_likes_open",),
        tool="soundcloud_likes",
        cli=("soundcloud_likes",),
        result="count, url",
        doc_example="открой лайкнутые треки на саундклауде",
    ),
    ActionSpec(
        "soundcloud_likes_play",
        "Включить лайкнутый трек",
        "включить случайный или выбранный лайкнутый трек SoundCloud",
        "media",
        examples=(
            "включи случайный лайкнутый трек на саундклауде",
            "включи лайкнутый трек отпусти меня",
        ),
        intents=("soundcloud_likes_play",),
        tool="soundcloud_likes",
        cli=("soundcloud_likes",),
        result="title, url",
        doc_example="включи случайный лайкнутый трек на саундклауде",
    ),
    ActionSpec(
        "play",
        "Продолжить",
        "продолжить воспроизведение",
        "media",
        examples=("продолжи", "продолжи в саундклауде"),
        intents=("play",),
        tool="play",
        cli=("play",),
        result="state, url",
        doc_example="продолжи",
    ),
    ActionSpec(
        "soundcloud_resume",
        "Продолжить SoundCloud",
        "продолжить воспроизведение именно в SoundCloud",
        "media",
        examples=("продолжи в саундклауде",),
        intents=("soundcloud_resume",),
        tool="play",
        cli=("play",),
        result="state, url",
        doc_example="продолжи в саундклауде",
    ),
    ActionSpec(
        "pause",
        "Пауза",
        "поставить воспроизведение на паузу",
        "media",
        examples=("пауза", "поставь на паузу"),
        intents=("pause",),
        tool="pause",
        cli=("pause",),
        result="state, url",
        doc_example="пауза",
    ),
    ActionSpec(
        "soundcloud_pause",
        "Пауза SoundCloud",
        "поставить на паузу именно SoundCloud",
        "media",
        examples=("пауза в саундклауде",),
        intents=("soundcloud_pause",),
        tool="pause",
        cli=("pause",),
        result="state, url",
        doc_example="пауза в саундклауде",
    ),
    ActionSpec(
        "stop",
        "Стоп",
        "остановить воспроизведение, не меняя позицию",
        "media",
        examples=("стоп", "останови видео"),
        intents=("stop",),
        tool="stop",
        cli=("stop",),
        result="state, url, position",
        doc_example="стоп",
    ),
    ActionSpec(
        "seek",
        "Перемотка вперёд",
        "перемотать видео или трек на N секунд вперёд",
        "media",
        examples=("перемотай на 30 секунд", "перемотай вперёд на минуту"),
        intents=("seek",),
        tool="seek",
        cli=("seek",),
        result="url, position, duration",
        doc_example="перемотай на 30 секунд",
    ),
    ActionSpec(
        "seek_back",
        "Перемотка назад",
        "перемотать видео или трек на N секунд назад",
        "media",
        examples=("перемотай назад на 10 секунд", "отмотай минуту"),
        intents=("seek_back",),
        tool="seek",
        cli=("seek",),
        result="url, position, duration",
        doc_example="перемотай назад на 10 секунд",
    ),
    ActionSpec(
        "next",
        "Следующий трек",
        "включить следующий трек или видео",
        "media",
        examples=("следующий трек", "дальше"),
        intents=("next",),
        tool="next",
        cli=("next",),
        result="state, url",
        doc_example="следующий трек",
    ),
    ActionSpec(
        "soundcloud_next",
        "Следующий трек SoundCloud",
        "включить следующий трек именно в SoundCloud",
        "media",
        examples=("следующий трек в саундклауде",),
        intents=("soundcloud_next",),
        tool="next",
        cli=("next",),
        result="state, url",
        doc_example="следующий трек в саундклауде",
    ),
    ActionSpec(
        "prev",
        "Предыдущий трек",
        "включить предыдущий трек или видео",
        "media",
        examples=("предыдущий трек", "назад видео"),
        intents=("prev",),
        tool="prev",
        cli=("prev",),
        result="state, url",
        doc_example="предыдущий трек",
    ),
    ActionSpec(
        "soundcloud_prev",
        "Предыдущий трек SoundCloud",
        "включить предыдущий трек именно в SoundCloud",
        "media",
        examples=("предыдущий трек в саундклауде",),
        intents=("soundcloud_prev",),
        tool="prev",
        cli=("prev",),
        result="state, url",
        doc_example="предыдущий трек в саундклауде",
    ),
    # ---------------------------------------------------------------- вкладки
    ActionSpec(
        "new_tab",
        "Новая вкладка",
        "открыть новую вкладку",
        "tabs",
        examples=("новая вкладка", "открой новую вкладку"),
        intents=("new_tab",),
        tool="newtab",
        cli=("newtab",),
        result="url",
        doc_example="новая вкладка",
    ),
    ActionSpec(
        "close_tab",
        "Закрыть вкладку",
        "закрыть текущую вкладку",
        "tabs",
        examples=("закрой вкладку",),
        intents=("close_tab",),
        tool="closetab",
        cli=("closetab",),
        result="url",
        doc_example="закрой вкладку",
    ),
    ActionSpec(
        "switch_tab",
        "Переключить вкладку",
        "переключиться на вкладку по номеру",
        "tabs",
        examples=("переключись на вкладку 2", "вкладка два"),
        intents=("switch_tab",),
        tool="tab",
        cli=("tab",),
        result="index, url, title",
        doc_example="переключись на вкладку 2",
    ),
    ActionSpec(
        "list_tabs",
        "Список вкладок",
        "показать открытые вкладки",
        "tabs",
        examples=("какие вкладки", "покажи вкладки"),
        intents=("list_tabs",),
        cli=("tabs",),
        result="tabs",
        doc_example="какие вкладки",
    ),
    # ---------------------------------------------------------------- страница
    ActionSpec(
        "scroll_down",
        "Прокрутить вниз",
        "прокрутить страницу вниз",
        "page",
        examples=("прокрути вниз", "листай вниз"),
        intents=("scroll_down",),
        tool="scroll",
        cli=("scroll",),
        result="direction",
        doc_example="прокрути вниз",
    ),
    ActionSpec(
        "scroll_up",
        "Прокрутить вверх",
        "прокрутить страницу вверх",
        "page",
        examples=("прокрути вверх", "листай вверх"),
        intents=("scroll_up",),
        tool="scroll",
        cli=("scroll",),
        result="direction",
        doc_example="прокрути вверх",
    ),
    ActionSpec(
        "fullscreen",
        "Полный экран",
        "развернуть видео или страницу на весь экран",
        "page",
        examples=("во весь экран", "разверни видео"),
        intents=("fullscreen",),
        result="state",
        doc_example="во весь экран",
    ),
    ActionSpec(
        "status",
        "Статус",
        "показать текущую вкладку и адрес",
        "page",
        examples=("что играет", "статус"),
        intents=("status",),
        tool="status",
        cli=("status",),
        result="title, url",
        doc_example="что играет",
    ),
    # ---------------------------------------------------------------- громкость
    ActionSpec(
        "system_volume_set",
        "Системная громкость",
        "задать системную громкость",
        "volume",
        examples=("громкость 5", "громкость пятьдесят"),
        intents=("system_volume_set",),
        tool="volume",
        result="level",
        doc_example="громкость 5",
    ),
    ActionSpec(
        "system_volume_up",
        "Система громче",
        "сделать системную громкость громче",
        "volume",
        examples=("громче", "громче на 2"),
        intents=("system_volume_up",),
        tool="volume_up",
        result="level",
        doc_example="громче",
    ),
    ActionSpec(
        "system_volume_down",
        "Система тише",
        "сделать системную громкость тише",
        "volume",
        examples=("тише", "тише на 1"),
        intents=("system_volume_down",),
        tool="volume_down",
        result="level",
        doc_example="тише",
    ),
    ActionSpec(
        "system_mute",
        "Выключить звук",
        "выключить системный звук",
        "volume",
        examples=("выключи звук", "мьют"),
        intents=("system_mute",),
        tool="mute",
        result="muted",
        doc_example="выключи звук",
    ),
    ActionSpec(
        "system_unmute",
        "Включить звук",
        "включить системный звук",
        "volume",
        examples=("включи звук", "верни звук"),
        intents=("system_unmute",),
        tool="unmute",
        result="muted",
        doc_example="включи звук",
    ),
    ActionSpec(
        "soundcloud_volume_set",
        "Громкость SoundCloud",
        "задать громкость плеера SoundCloud",
        "volume",
        examples=("громкость саундклауда 5", "звук саундклауда 10"),
        intents=("soundcloud_volume_set",),
        cli=("soundcloud_volume",),
        result="volume",
        doc_example="громкость саундклауда 5",
    ),
    ActionSpec(
        "soundcloud_volume_up",
        "Громче SoundCloud",
        "сделать громче плеер SoundCloud",
        "volume",
        examples=("громче в саундклауде", "прибавь звук на саундклауде"),
        intents=("soundcloud_volume_up",),
        cli=("soundcloud_volume",),
        result="volume",
        doc_example="громче в саундклауде",
    ),
    ActionSpec(
        "soundcloud_volume_down",
        "Тише SoundCloud",
        "сделать тише плеер SoundCloud",
        "volume",
        examples=("тише на 1 в саундклауде", "убавь звук на саундклауде"),
        intents=("soundcloud_volume_down",),
        cli=("soundcloud_volume",),
        result="volume",
        doc_example="тише на 1 в саундклауде",
    ),
    ActionSpec(
        "soundcloud_volume_mute",
        "Выключить звук SoundCloud",
        "выключить звук плеера SoundCloud",
        "volume",
        examples=("выключи звук в саундклауде", "мьют в саундклауде"),
        intents=("soundcloud_volume_mute",),
        cli=("soundcloud_volume",),
        result="volume",
        doc_example="выключи звук в саундклауде",
    ),
    ActionSpec(
        "soundcloud_volume_unmute",
        "Включить звук SoundCloud",
        "включить звук плеера SoundCloud",
        "volume",
        examples=("включи звук на саундклауде", "верни звук в саундклауде"),
        intents=("soundcloud_volume_unmute",),
        cli=("soundcloud_volume",),
        result="volume",
        doc_example="включи звук на саундклауде",
    ),
    ActionSpec(
        "volume_up_context",
        "Громче по контексту",
        "сделать громче то, что меняли последним: систему или SoundCloud",
        "context",
        examples=("сделай громче",),
        intents=("volume_up_context",),
        result="level",
        doc_example="сделай громче",
    ),
    ActionSpec(
        "volume_down_context",
        "Тише по контексту",
        "сделать тише то, что меняли последним: систему или SoundCloud",
        "context",
        examples=("сделай тише",),
        intents=("volume_down_context",),
        result="level",
        doc_example="сделай тише",
    ),
    # ------------------------------------------------------- только для агента
    ActionSpec(
        "click",
        "Клик по селектору",
        "кликнуть по CSS-селектору на странице",
        "agent",
        cli=("click",),
        result="url",
        risk="medium",
        documented=False,
    ),
    ActionSpec(
        "fill",
        "Заполнить поле",
        "ввести текст в поле по CSS-селектору",
        "agent",
        cli=("fill",),
        result="url",
        risk="medium",
        documented=False,
    ),
    ActionSpec(
        "press",
        "Нажать клавишу",
        "нажать клавишу на странице",
        "agent",
        cli=("press",),
        result="url",
        documented=False,
    ),
    ActionSpec(
        "wait",
        "Подождать",
        "подождать заданное время",
        "agent",
        cli=("wait",),
        result="url",
        documented=False,
    ),
    ActionSpec(
        "screenshot",
        "Скриншот",
        "сохранить скриншот страницы",
        "agent",
        cli=("screenshot",),
        result="path",
        documented=False,
    ),
    ActionSpec(
        "eval",
        "Выполнить JS",
        "выполнить JavaScript на странице",
        "agent",
        cli=("eval",),
        result="result",
        risk="high",
        documented=False,
    ),
    ActionSpec(
        "type",
        "Ввести текст",
        "ввести текст в активный элемент",
        "agent",
        cli=("type",),
        result="url",
        documented=False,
    ),
)


def build_tools(
    site_urls: Mapping[str, str],
    site_keys: Iterable[str],
) -> tuple[Tool, ...]:
    """Собирает инструменты, которые уже умеет ActionRouter.

    Описание и валидация живут здесь; промпт для LLM генерируется из
    этого же реестра. Новый инструмент добавляется в одном месте.
    """
    keys_list = list(site_keys)
    keys = ", ".join(keys_list)
    return (
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
        Tool(
            "play",
            "продолжить воспроизведение.",
            (
                ToolParam(
                    "site",
                    "string",
                    "youtube или soundcloud; без site — активное медиа",
                    enum=frozenset({"youtube", "soundcloud"}),
                    lower=True,
                ),
            ),
        ),
        Tool(
            "pause",
            "пауза.",
            (
                ToolParam(
                    "site",
                    "string",
                    "youtube или soundcloud; без site — активное медиа",
                    enum=frozenset({"youtube", "soundcloud"}),
                    lower=True,
                ),
            ),
        ),
        Tool(
            "stop",
            "остановить воспроизведение (пауза без возврата в начало).",
            (
                ToolParam(
                    "site",
                    "string",
                    "youtube или soundcloud; без site — активное медиа",
                    enum=frozenset({"youtube", "soundcloud"}),
                    lower=True,
                ),
            ),
        ),
        Tool(
            "seek",
            "перемотать видео или трек на N секунд.",
            (
                ToolParam(
                    "seconds",
                    "integer",
                    "на сколько секунд",
                    default=10,
                    minimum=1,
                    maximum=3600,
                ),
                ToolParam(
                    "direction",
                    "string",
                    "forward или backward",
                    default="forward",
                    enum=frozenset({"forward", "backward"}),
                    lower=True,
                ),
                ToolParam(
                    "site",
                    "string",
                    "youtube или soundcloud; без site — активное медиа",
                    enum=frozenset({"youtube", "soundcloud"}),
                    lower=True,
                ),
            ),
        ),
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
    )


def action_names() -> tuple[str, ...]:
    return tuple(action.name for action in ACTIONS)


def get_action(name: str) -> ActionSpec | None:
    for action in ACTIONS:
        if action.name == name:
            return action
    return None


def validate_catalog() -> list[str]:
    """Возвращает список проблем каталога; пустой список — всё в порядке."""
    problems: list[str] = []
    seen: set[str] = set()
    for action in ACTIONS:
        if action.name in seen:
            problems.append(f"повторяющееся действие: {action.name}")
        seen.add(action.name)
        if not action.title or not action.description:
            problems.append(f"у действия {action.name} нет title/description")
        if not action.category:
            problems.append(f"у действия {action.name} нет category")
        if action.documented:
            if not action.doc_example:
                problems.append(f"у действия {action.name} нет doc_example")
            if not action.examples:
                problems.append(f"у действия {action.name} нет examples")
        elif action.doc_example:
            problems.append(f"у недокументированного действия {action.name} есть doc_example")
    return problems


def catalog_by_tool() -> dict[str, tuple[ActionSpec, ...]]:
    result: dict[str, list[ActionSpec]] = {}
    for action in ACTIONS:
        if action.tool:
            result.setdefault(action.tool, []).append(action)
    return {name: tuple(items) for name, items in result.items()}


def catalog_by_cli() -> dict[str, tuple[ActionSpec, ...]]:
    result: dict[str, list[ActionSpec]] = {}
    for action in ACTIONS:
        for command in action.cli:
            result.setdefault(command, []).append(action)
    return {name: tuple(items) for name, items in result.items()}


def tool_names() -> tuple[str, ...]:
    return tuple(name for name in catalog_by_tool())


def as_dict(action: ActionSpec) -> dict[str, Any]:
    """Представление для тестов и будущей генерации документации."""
    return {
        "name": action.name,
        "title": action.title,
        "description": action.description,
        "category": action.category,
        "examples": list(action.examples),
        "intents": list(action.intents),
        "tool": action.tool,
        "cli": list(action.cli),
        "result": action.result,
        "risk": action.risk,
        "documented": action.documented,
    }
