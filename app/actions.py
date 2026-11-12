from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from .context import CommandContext
from .intents import IntentMatch, IntentSet
from .slots import NUMBER_WORDS, RESULT_WORDS, TAB_WORDS
from .text import canonicalize, normalize

log = logging.getLogger(__name__)

SITE_URLS: dict[str, str] = {
    "ютуб": "https://www.youtube.com",
    "youtube": "https://www.youtube.com",
    "гугл": "https://www.google.com",
    "google": "https://www.google.com",
    "яндекс": "https://ya.ru",
    "yandex": "https://ya.ru",
    "вк": "https://vk.com",
    "вконтакте": "https://vk.com",
    "телеграм": "https://web.telegram.org",
    "telegram": "https://web.telegram.org",
    "почта": "https://mail.google.com",
    "gmail": "https://mail.google.com",
    "календарь": "https://calendar.google.com",
    "диск": "https://drive.google.com",
    "гитхаб": "https://github.com",
    "github": "https://github.com",
    "чатгпт": "https://chatgpt.com",
    "chatgpt": "https://chatgpt.com",
    "википедия": "https://ru.wikipedia.org",
    "кинопоиск": "https://www.kinopoisk.ru",
    "озон": "https://www.ozon.ru",
    "ozon": "https://www.ozon.ru",
    "вайлдберриз": "https://www.wildberries.ru",
    "wildberries": "https://www.wildberries.ru",
    "авито": "https://www.avito.ru",
    "карты": "https://yandex.ru/maps",
    "переводчик": "https://translate.yandex.ru",
    "музыка": "https://music.yandex.ru",
    "яндекс музыка": "https://music.yandex.ru",
    "хабр": "https://habr.com",
    "саундклауд": "https://soundcloud.com",
    "саунд клауд": "https://soundcloud.com",
    "саундклоуд": "https://soundcloud.com",
    "саунд клоуд": "https://soundcloud.com",
    "саунтлауди": "https://soundcloud.com",
    "саунтлауде": "https://soundcloud.com",
    "саундклауди": "https://soundcloud.com",
    "санклауд": "https://soundcloud.com",
    "soundcloud": "https://soundcloud.com",
    "sound cloud": "https://soundcloud.com",
}

CANCEL_WORDS = {"отмена", "отмени", "ничего", "неважно", "не надо"}

VOLUME_STEP = 0.10
DEFAULT_SEEK_SEC = 10
_SOUNDCLOUD_RE = re.compile(
    r"(?:сау(?:н[дт]?|[гк])?[ао]?|с[ао]ун[дт]?|с[ао]н[дт]?)\s*[кг]?\s*л[ао]у[дт]\w*"
    r"|sound\s*cloud",
    re.UNICODE,
)

# «Громкость … число» — если слово перед числом распознано с ошибкой
# («сомкала», «на паузу»), но SoundCloud открыт, это почти наверняка плеер.
_AMBIGUOUS_VOLUME_RE = re.compile(r"(?:громкость|громко|звук)", re.UNICODE)

# GigaAM на быстрой речи обрезает «ютуб» до «ту»/«тю»: «открой ту».
_YOUTUBE_SHORT_RE = re.compile(
    r"(?:открой|включи|запусти|поставь)\s+(?:ту|тю|тюб|ют|юта|юту|юбе|юб)",
    re.UNICODE,
)

def parse_volume_percent(text: str) -> int | None:
    """Разбирает громкость по шкале 0–10: «5» → 50%, «10» → 100%.

    Числа 11–100 понимаются как проценты (старый формат), а «5 процентов» —
    как 5%. «на максимум» → 100%, «на минимум» → 0%.
    """
    value = normalize(text)
    explicit_percent = bool(re.search(r"(?<!\w)процент", value))
    value = re.sub(r"(?<!\w)процент(?:а|ов)?(?!\w)", " ", value)
    value = re.sub(r"^на\s+", "", value.strip())
    value = re.sub(r"\s+", " ", value).strip(" .")
    if not value:
        return None

    def to_percent(number: float) -> int:
        percent = number if explicit_percent or number > 10 else number * 10
        return max(0, min(100, round(percent)))

    if value in {"максимум", "макс", "полную", "на всю", "все"}:
        return 100
    if value in {"минимум", "мин"}:
        return 0
    match = re.fullmatch(r"(\d{1,3}(?:[.,]\d{1,2})?)", value)
    if match:
        return to_percent(float(match.group(1).replace(",", ".")))
    total = 0
    seen = False
    for token in value.split():
        if token in {"и", "целых"}:
            continue
        if token in NUMBER_WORDS:
            total += NUMBER_WORDS[token]
            seen = True
        else:
            return None
    if not seen:
        return None
    return to_percent(total)


def _as_int(value: Any, default: int) -> int:
    """Мягкое приведение к int для аргументов локальной LLM."""
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class CommandResult:
    handled: bool
    ok: bool = True
    message: str = ""


class ActionRouter:
    """Быстрые детерминированные команды — без обращения к LLM."""

    def __init__(
        self,
        cfg: dict[str, Any],
        browser: Any,
        intents: IntentSet | None = None,
        context: CommandContext | None = None,
    ) -> None:
        self.cfg = cfg
        self.browser = browser
        self._volume = None
        context_cfg = cfg.get("context") if isinstance(cfg, dict) else None
        ttl = float((context_cfg or {}).get("ttl_sec", 180))
        self.context = context or CommandContext(ttl)
        self.intents = intents if intents is not None else IntentSet.load(
            site_names=SITE_URLS,
            volume_parser=parse_volume_percent,
        )

    # ------------------------------------------------------------------ public
    def handle(self, text: str) -> CommandResult | None:
        t = canonicalize(normalize(text))
        if not t:
            return CommandResult(True, True, "")

        if t in CANCEL_WORDS:
            self.context.clear()
            return CommandResult(True, True, "Отменено")

        match = self.intents.match(t, self.context)
        if match is not None:
            result = self._handle_intent(match)
            if result is not None:
                log.info(
                    "Интент: %s; слоты: %s",
                    match.intent.id,
                    dict(match.slots),
                )
                return result

        ambiguous = self._ambiguous_soundcloud_volume(t)
        if ambiguous:
            return ambiguous

        tabs = self._tabs_command(t)
        if tabs:
            return tabs

        media = self._media_command(t)
        if media:
            return media

        # Если в команде есть «саундклауд», но шаблон не совпал (STT добавил
        # лишнее слово), не отправляем её в YouTube/Google: открываем
        # SoundCloud или ищем на нём.
        if _SOUNDCLOUD_RE.search(t) and re.match(
            r"^(?:включи|поставь|запусти|найди|покажи|послушай|играй)\b", t
        ):
            query = self._soundcloud_query(t)
            if query:
                self.context.remember(
                    "search",
                    "soundcloud_search",
                    site="soundcloud",
                    query=query,
                    index=1,
                )
                self._remember_media("soundcloud_search", site="soundcloud")
                return self._simple(
                    "soundcloud",
                    f"Включаю на SoundCloud: {query}",
                    query=query,
                    index=1,
                )
            self._remember_media("open_site", site="soundcloud")
            return self._simple(
                "goto", "Открываю SoundCloud", url=SITE_URLS["саундклауд"]
            )

        short_youtube = bool(_YOUTUBE_SHORT_RE.fullmatch(t))
        if "ютуб" in t or "youtube" in t or "you tube" in t or short_youtube:
            if short_youtube:
                self._remember_media("open_site", site="youtube")
                return self._simple("goto", "Открываю YouTube", url=SITE_URLS["ютуб"])
            if re.search(
                r"(включи|поставь|запусти|найди|покажи|открой|"
                r"play|find|search|open|show|look)",
                t,
            ):
                query = self._youtube_query(t)
                if query:
                    self._remember_youtube_search(query)
                    return self._simple(
                        "youtube",
                        f"Включаю на YouTube: {query}",
                        query=query,
                        index=1,
                    )
                self._remember_media("open_site", site="youtube")
                return self._simple("goto", "Открываю YouTube", url=SITE_URLS["ютуб"])

        match = re.match(
            r"^(?:включи|поставь|запусти)\s+(?:видео|клип|песню|трек|ролик)\s+(.+)$", t
        )
        if match:
            query = match.group(1).strip()
            self._remember_youtube_search(query)
            return self._simple(
                "youtube", f"Включаю на YouTube: {query}", query=query, index=1
            )

        search = self._search_command(t)
        if search:
            return search

        result = self._result_command(t)
        if result:
            return result

        opened = self._open_command(t)
        if opened:
            return opened

        navigation = self._navigation_command(t)
        if navigation:
            return navigation

        misc = self._misc_command(t)
        if misc:
            return misc

        self.context.clear()
        return None

    def execute_plan(
        self, action: str, args: dict[str, Any] | None = None
    ) -> CommandResult:
        """Выполняет действие, выбранное локальной LLM.

        Аргументы уже проверены планировщиком; здесь только защита от прямых
        вызовов и преобразование в существующие команды браузера.
        """
        action = (action or "").strip().lower()
        args = dict(args or {})

        if action == "open_site":
            site = str(args.get("site") or "").strip().lower()
            url = SITE_URLS.get(site)
            if not url:
                return CommandResult(True, False, f"Неизвестный сайт: {site}")
            self.context.remember("media", "open_site", site=site)
            return self._simple("goto", f"Открываю {site}", url=url)

        if action == "goto":
            url = str(args.get("url") or "").strip()
            if not re.match(r"^https?://", url, re.IGNORECASE):
                return CommandResult(
                    True, False, "Поддерживаются только http(s)-адреса"
                )
            return self._simple("goto", "Открываю сайт", url=url)

        if action in {"youtube", "soundcloud"}:
            query = str(args.get("query") or "").strip()
            if not query:
                return CommandResult(True, False, "Пустой поисковый запрос")
            index = max(1, min(20, _as_int(args.get("index"), 1)))
            if action == "youtube":
                self._remember_youtube_search(query)
                return self._simple(
                    "youtube",
                    f"Включаю на YouTube: {query}",
                    query=query,
                    index=index,
                )
            self.context.remember(
                "search",
                "soundcloud_search",
                site="soundcloud",
                query=query,
                index=1,
            )
            self._remember_media("soundcloud_search", site="soundcloud")
            return self._simple(
                "soundcloud",
                f"Включаю на SoundCloud: {query}",
                query=query,
                index=index,
            )

        if action == "soundcloud_likes":
            query = self._likes_query(str(args.get("query") or ""))
            play = bool(args.get("play", True))
            random_choice = not query
            if play:
                message = (
                    "Включаю случайный лайкнутый трек"
                    if random_choice
                    else f"Ищу среди лайкнутых: {query}"
                )
            else:
                message = "Открываю лайкнутые треки SoundCloud"
            self.context.remember(
                "search",
                "soundcloud_likes",
                site="soundcloud",
                query=query,
                random=random_choice,
            )
            self._remember_media("soundcloud_likes", site="soundcloud")
            return self._simple(
                "soundcloud_likes",
                message,
                query=query,
                random=random_choice,
                play=play,
            )

        if action == "google":
            query = str(args.get("query") or "").strip()
            if not query:
                return CommandResult(True, False, "Пустой поисковый запрос")
            self._remember_google_search(query)
            return self._simple("google", f"Ищу: {query}", query=query)

        if action in {"play", "pause", "next", "prev"}:
            messages = {
                "play": "Продолжаю",
                "pause": "Пауза",
                "next": "Следующее видео",
                "prev": "Предыдущее видео",
            }
            site = None
            if action in {"play", "pause"}:
                site = str(args.get("site") or "").strip().lower()
                if site and site not in {"youtube", "soundcloud"}:
                    return CommandResult(True, False, f"Неизвестный сайт: {site}")
            self._remember_media(action, site=site)
            kwargs = {"site": site} if site else {}
            return self._simple(action, messages[action], **kwargs)

        if action == "stop":
            site = str(args.get("site") or "").strip().lower()
            if site and site not in {"youtube", "soundcloud"}:
                return CommandResult(True, False, f"Неизвестный сайт: {site}")
            self._remember_media("stop", site=site)
            kwargs = {"site": site} if site else {}
            return self._simple("stop", "Останавливаю", **kwargs)

        if action == "seek":
            seconds = _as_int(args.get("seconds"), DEFAULT_SEEK_SEC)
            if seconds <= 0 or seconds > 3600:
                return CommandResult(True, False, "Неверное время перемотки")
            direction = str(args.get("direction") or "forward").strip().lower()
            if direction not in {"forward", "backward"}:
                return CommandResult(
                    True, False, "Неверное направление перемотки"
                )
            if direction == "backward":
                seconds = -seconds
            site = str(args.get("site") or "").strip().lower()
            if site and site not in {"youtube", "soundcloud"}:
                return CommandResult(True, False, f"Неизвестный сайт: {site}")
            self._remember_media("seek", site=site)
            word = "назад" if seconds < 0 else "вперед"
            kwargs = {"site": site} if site else {}
            return self._simple(
                "seek",
                f"Перематываю {word} на {abs(seconds)} с",
                seconds=seconds,
                **kwargs,
            )

        if action == "newtab":
            return self._simple("newtab", "Открываю новую вкладку")
        if action == "closetab":
            return self._simple("closetab", "Закрываю вкладку")
        if action == "tab":
            index = _as_int(args.get("index"), 0)
            if index < 1 or index > 99:
                return CommandResult(True, False, "Неверный номер вкладки")
            self.context.remember("media", "switch_tab", site="browser")
            return self._simple(
                "tab", f"Переключаюсь на вкладку {index}", index=index
            )
        if action == "back":
            return self._simple("back", "Назад")
        if action == "forward":
            return self._simple("forward", "Вперёд")
        if action == "reload":
            return self._simple("reload", "Обновляю страницу")
        if action == "scroll":
            direction = str(args.get("direction") or "down").strip().lower()
            if direction not in {"up", "down"}:
                direction = "down"
            message = "Листаю вверх" if direction == "up" else "Листаю вниз"
            return self._simple("scroll", message, direction=direction)
        if action == "volume":
            level = _as_int(args.get("level"), -1)
            if level < 0 or level > 100:
                return CommandResult(True, False, "Неверная громкость")
            self._remember_system_volume()
            return self._set_volume(level / 100)
        if action in {"volume_up", "volume_down"}:
            self._remember_system_volume()
            delta = VOLUME_STEP if action == "volume_up" else -VOLUME_STEP
            return self._change_volume(delta)
        if action in {"mute", "unmute"}:
            self._remember_system_volume()
            return self._set_mute(action == "mute")
        if action == "status":
            ok, data, error = self._call_browser("status")
            if not ok:
                return CommandResult(True, False, f"Ошибка: {error}")
            title = data.get("title") or "(без названия)"
            url = data.get("url") or ""
            return CommandResult(True, True, f"{title} — {url}")

        return CommandResult(True, False, f"Неизвестное действие: {action}")

    # --------------------------------------------------------------- intents
    def _handle_intent(self, match: IntentMatch) -> CommandResult | None:
        handler = getattr(self, f"_intent_{match.intent.action}", None)
        if handler is None:
            log.warning("Нет исполнителя для интента %s", match.intent.id)
            return None
        return handler(match)

    def _intent_new_tab(self, match: IntentMatch) -> CommandResult:
        return self._simple("newtab", "Открываю новую вкладку")

    def _intent_close_tab(self, match: IntentMatch) -> CommandResult:
        return self._simple("closetab", "Закрываю вкладку")

    def _intent_switch_tab(self, match: IntentMatch) -> CommandResult:
        index = int(match.slots["tab"])
        self.context.remember("media", "switch_tab", site="browser")
        return self._simple("tab", f"Переключаюсь на вкладку {index}", index=index)

    def _intent_list_tabs(self, match: IntentMatch) -> CommandResult:
        return self._list_tabs()

    def _intent_pause(self, match: IntentMatch) -> CommandResult:
        site = match.slots.get("site")
        self._remember_media("pause", site=site)
        kwargs = {"site": site} if site else {}
        return self._simple("pause", "Пауза", **kwargs)

    def _intent_play(self, match: IntentMatch) -> CommandResult:
        site = match.slots.get("site")
        self._remember_media("play", site=site)
        kwargs = {"site": site} if site else {}
        return self._simple("play", "Продолжаю", **kwargs)

    def _intent_stop(self, match: IntentMatch) -> CommandResult:
        site = match.slots.get("site")
        self._remember_media("stop", site=site)
        kwargs = {"site": site} if site else {}
        return self._simple("stop", "Останавливаю", **kwargs)

    def _intent_seek(self, match: IntentMatch) -> CommandResult:
        return self._seek_media(match, backward=False)

    def _intent_seek_back(self, match: IntentMatch) -> CommandResult:
        return self._seek_media(match, backward=True)

    def _seek_media(self, match: IntentMatch, backward: bool) -> CommandResult:
        seconds = int(match.slots.get("seconds") or DEFAULT_SEEK_SEC)
        if seconds > 3600:
            return CommandResult(True, False, "Неверное время перемотки")
        if backward:
            seconds = -seconds
        site = match.slots.get("site")
        self._remember_media("seek", site=site)
        direction = "назад" if seconds < 0 else "вперед"
        kwargs = {"site": site} if site else {}
        return self._simple(
            "seek",
            f"Перематываю {direction} на {abs(seconds)} с",
            seconds=seconds,
            **kwargs,
        )

    def _intent_next(self, match: IntentMatch) -> CommandResult:
        self._remember_media("next")
        return self._simple("next", "Следующее видео")

    def _intent_prev(self, match: IntentMatch) -> CommandResult:
        self._remember_media("prev")
        return self._simple("prev", "Предыдущее видео")

    def _intent_back(self, match: IntentMatch) -> CommandResult:
        return self._simple("back", "Назад")

    def _intent_forward(self, match: IntentMatch) -> CommandResult:
        return self._simple("forward", "Вперёд")

    def _intent_reload(self, match: IntentMatch) -> CommandResult:
        return self._simple("reload", "Обновляю страницу")

    def _intent_scroll_down(self, match: IntentMatch) -> CommandResult:
        return self._simple("scroll", "Листаю вниз", direction="down")

    def _intent_scroll_up(self, match: IntentMatch) -> CommandResult:
        return self._simple("scroll", "Листаю вверх", direction="up")

    def _intent_fullscreen(self, match: IntentMatch) -> CommandResult:
        ok, data, _ = self._call_browser("status")
        if ok and "youtube.com" in str(data.get("url", "")):
            return self._simple("press", "Включаю полный экран", key="f")
        return self._simple(
            "eval",
            "Включаю полный экран",
            script="document.documentElement.requestFullscreen().catch(() => {})",
        )

    def _intent_status(self, match: IntentMatch) -> CommandResult:
        ok, data, error = self._call_browser("status")
        if not ok:
            return CommandResult(True, False, f"Ошибка: {error}")
        title = data.get("title") or "(без названия)"
        url = data.get("url") or ""
        return CommandResult(True, True, f"{title} — {url}")

    def _intent_open_site(self, match: IntentMatch) -> CommandResult:
        site = str(match.slots["site"])
        self.context.remember("media", "open_site", site=site)
        return self._simple("goto", f"Открываю {site}", url=SITE_URLS[site])

    def _intent_google_search(self, match: IntentMatch) -> CommandResult:
        query = self._clean_google_query(str(match.slots["query"]))
        if not query:
            return None
        self.context.remember(
            "search", "google_search", site="google", query=query, index=1
        )
        self.context.remember("results", "google_search", index=1)
        return self._simple("google", f"Ищу: {query}", query=query)

    def _intent_youtube_search(self, match: IntentMatch) -> CommandResult:
        query = self._youtube_query(str(match.slots["query"]))
        if not query:
            return None
        self.context.remember(
            "search", "youtube_search", site="youtube", query=query, index=1
        )
        self.context.remember("media", "youtube_search", site="youtube")
        return self._simple(
            "youtube", f"Включаю на YouTube: {query}", query=query, index=1
        )

    def _intent_open_result(self, match: IntentMatch) -> CommandResult:
        index = int(match.slots["index"])
        query = self._google_query_from_context()
        self.context.remember(
            "results", "search_result", index=index, query=query
        )
        return self._simple(
            "search_result",
            f"Открываю ссылку №{index}",
            index=index,
            query=query,
        )

    def _intent_repeat_search(self, match: IntentMatch) -> CommandResult | None:
        entry = self.context.get("search")
        if entry is None:
            return None
        action = entry.action
        slots = entry.slots
        query = str(slots.get("query") or "")
        index = int(slots.get("index") or 1)

        if action == "youtube_search" and query:
            self.context.remember(
                "search", action, site="youtube", query=query, index=index
            )
            self.context.remember("media", "youtube_search", site="youtube")
            return self._simple(
                "youtube", f"Включаю на YouTube: {query}", query=query, index=index
            )
        if action == "soundcloud_search" and query:
            self.context.remember(
                "search", action, site="soundcloud", query=query, index=index
            )
            self.context.remember("media", "soundcloud_search", site="soundcloud")
            return self._simple(
                "soundcloud",
                f"Включаю на SoundCloud: {query}",
                query=query,
                index=index,
            )
        if action == "soundcloud_likes":
            self.context.remember(
                "search",
                action,
                site="soundcloud",
                query=query,
                random=bool(slots.get("random")),
            )
            self.context.remember("media", "soundcloud_likes", site="soundcloud")
            message = (
                "Включаю случайный лайкнутый трек"
                if slots.get("random")
                else f"Ищу среди лайкнутых: {query}"
            )
            return self._simple(
                "soundcloud_likes",
                message,
                query=query,
                random=bool(slots.get("random")),
                play=True,
            )
        if action == "google_search" and query:
            self.context.remember(
                "search", action, site="google", query=query, index=1
            )
            self.context.remember("results", "google_search", index=1)
            return self._simple("google", f"Ищу: {query}", query=query)
        if action == "search_result":
            self.context.remember(
                "results", "search_result", index=index, query=query
            )
            return self._simple(
                "search_result",
                f"Открываю ссылку №{index}",
                index=index,
                query=query,
            )
        return None

    def _intent_next_result(self, match: IntentMatch) -> CommandResult | None:
        entry = self.context.get("search")
        if entry is None:
            return None
        action = entry.action
        slots = entry.slots
        query = str(slots.get("query") or "")
        index = int(slots.get("index") or 1) + 1

        if action == "youtube_search" and query:
            self.context.remember(
                "search", action, site="youtube", query=query, index=index
            )
            self.context.remember("media", "youtube_search", site="youtube")
            return self._simple(
                "youtube",
                f"Включаю следующий результат YouTube: {query}",
                query=query,
                index=index,
            )
        if action == "soundcloud_search" and query:
            self.context.remember(
                "search", action, site="soundcloud", query=query, index=index
            )
            self.context.remember("media", "soundcloud_search", site="soundcloud")
            return self._simple(
                "soundcloud",
                f"Включаю следующий трек SoundCloud: {query}",
                query=query,
                index=index,
            )
        if action == "soundcloud_likes":
            random_choice = bool(slots.get("random"))
            self.context.remember(
                "search",
                action,
                site="soundcloud",
                query=query,
                random=random_choice,
            )
            self.context.remember("media", "soundcloud_likes", site="soundcloud")
            return self._simple(
                "soundcloud_likes",
                "Включаю случайный лайкнутый трек"
                if random_choice
                else f"Ищу среди лайкнутых: {query}",
                query=query,
                random=random_choice,
                play=True,
            )
        if action in {"google_search", "search_result"}:
            results = self.context.get("results")
            if results is not None:
                index = int(results.slots.get("index") or 1) + 1
                query = str(results.slots.get("query") or "") or query
            self.context.remember(
                "results", "search_result", index=index, query=query
            )
            return self._simple(
                "search_result",
                f"Открываю ссылку №{index}",
                index=index,
                query=query,
            )
        return None

    # ------------------------------------------------------ volume (система)
    def _intent_volume_up_context(self, match: IntentMatch) -> CommandResult:
        entry = self.context.get("volume")
        if entry is not None and entry.slots.get("target") == "soundcloud":
            return self._soundcloud_volume_change("up", None)
        return self._system_volume_change(1, None)

    def _intent_volume_down_context(self, match: IntentMatch) -> CommandResult:
        entry = self.context.get("volume")
        if entry is not None and entry.slots.get("target") == "soundcloud":
            return self._soundcloud_volume_change("down", None)
        return self._system_volume_change(-1, None)

    def _intent_system_volume_set(self, match: IntentMatch) -> CommandResult:
        percent = int(match.slots["level"])
        self._remember_system_volume()
        return self._set_volume(percent / 100)

    def _intent_system_volume_up(self, match: IntentMatch) -> CommandResult:
        return self._system_volume_change(1, match.slots.get("amount"))

    def _intent_system_volume_down(self, match: IntentMatch) -> CommandResult:
        return self._system_volume_change(-1, match.slots.get("amount"))

    def _system_volume_change(
        self, sign: int, amount: int | None
    ) -> CommandResult:
        self._remember_system_volume()
        step = amount / 100 if amount is not None and amount > 0 else VOLUME_STEP
        return self._change_volume(sign * step)

    def _intent_system_mute(self, match: IntentMatch) -> CommandResult:
        self._remember_system_volume()
        return self._set_mute(True)

    def _intent_system_unmute(self, match: IntentMatch) -> CommandResult:
        self._remember_system_volume()
        return self._set_mute(False)

    # --------------------------------------------------- volume (SoundCloud)
    def _intent_soundcloud_volume_set(self, match: IntentMatch) -> CommandResult:
        percent = int(match.slots["level"])
        self._remember_soundcloud_volume()
        return self._simple(
            "soundcloud_volume",
            f"Громкость SoundCloud: {percent}%",
            mode="set",
            value=percent,
        )

    def _intent_soundcloud_volume_up(self, match: IntentMatch) -> CommandResult:
        return self._soundcloud_volume_change("up", match.slots.get("amount"))

    def _intent_soundcloud_volume_down(self, match: IntentMatch) -> CommandResult:
        return self._soundcloud_volume_change("down", match.slots.get("amount"))

    def _soundcloud_volume_change(
        self, mode: str, amount: int | None
    ) -> CommandResult:
        step = int(amount) if amount is not None else round(VOLUME_STEP * 100)
        self._remember_soundcloud_volume()
        return self._simple(
            "soundcloud_volume",
            "Делаю громче в SoundCloud" if mode == "up" else "Делаю тише в SoundCloud",
            mode=mode,
            value=step,
        )

    def _intent_soundcloud_volume_mute(self, match: IntentMatch) -> CommandResult:
        self._remember_soundcloud_volume()
        return self._simple(
            "soundcloud_volume", "Выключаю звук SoundCloud", mode="mute"
        )

    def _intent_soundcloud_volume_unmute(self, match: IntentMatch) -> CommandResult:
        self._remember_soundcloud_volume()
        return self._simple(
            "soundcloud_volume", "Включаю звук SoundCloud", mode="unmute"
        )

    # ---------------------------------------------------------- SoundCloud
    def _intent_soundcloud_open(self, match: IntentMatch) -> CommandResult:
        self._remember_media("open_site", site="soundcloud")
        return self._simple(
            "goto", "Открываю SoundCloud", url=SITE_URLS["саундклауд"]
        )

    def _intent_soundcloud_search(self, match: IntentMatch) -> CommandResult | None:
        query = self._soundcloud_query(str(match.slots["query"]))
        if not query:
            return None
        self.context.remember(
            "search",
            "soundcloud_search",
            site="soundcloud",
            query=query,
            index=1,
        )
        self._remember_media("soundcloud_search", site="soundcloud")
        return self._simple(
            "soundcloud", f"Включаю на SoundCloud: {query}", query=query, index=1
        )

    def _intent_soundcloud_likes_open(self, match: IntentMatch) -> CommandResult:
        return self._soundcloud_likes(match, play=False)

    def _intent_soundcloud_likes_play(self, match: IntentMatch) -> CommandResult:
        return self._soundcloud_likes(match, play=True)

    def _soundcloud_likes(
        self, match: IntentMatch, play: bool
    ) -> CommandResult:
        query = self._likes_query(str(match.slots.get("query") or ""))
        if play:
            random_choice = not query
            message = (
                "Включаю случайный лайкнутый трек"
                if random_choice
                else f"Ищу среди лайкнутых: {query}"
            )
            self.context.remember(
                "search",
                "soundcloud_likes",
                site="soundcloud",
                query=query,
                random=random_choice,
            )
            self._remember_media("soundcloud_likes", site="soundcloud")
            return self._simple(
                "soundcloud_likes",
                message,
                query=query,
                random=random_choice,
                play=True,
            )
        self.context.remember(
            "search",
            "soundcloud_likes",
            site="soundcloud",
            query=query,
            random=True,
        )
        self._remember_media("soundcloud_likes", site="soundcloud")
        return self._simple(
            "soundcloud_likes", "Открываю лайкнутые треки SoundCloud", play=False
        )

    def _intent_soundcloud_pause(self, match: IntentMatch) -> CommandResult:
        self._remember_media("pause", site="soundcloud")
        return self._simple("pause", "Пауза", site="soundcloud")

    def _intent_soundcloud_resume(self, match: IntentMatch) -> CommandResult:
        self._remember_media("play", site="soundcloud")
        return self._simple("play", "Продолжаю", site="soundcloud")

    def _intent_soundcloud_next(self, match: IntentMatch) -> CommandResult:
        self._remember_media("next", site="soundcloud")
        return self._simple("next", "Следующий трек")

    def _intent_soundcloud_prev(self, match: IntentMatch) -> CommandResult:
        self._remember_media("prev", site="soundcloud")
        return self._simple("prev", "Предыдущий трек")

    # ------------------------------------------------------------------ volume
    def _volume_interface(self):
        if self._volume is not None:
            return self._volume
        try:
            from ctypes import POINTER, cast

            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            speakers = AudioUtilities.GetSpeakers()
            if hasattr(speakers, "EndpointVolume"):
                self._volume = speakers.EndpointVolume
            else:
                interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self._volume = cast(interface, POINTER(IAudioEndpointVolume))
        except Exception as exc:  # noqa: BLE001
            log.warning("Не удалось получить доступ к громкости: %s", exc)
            return None
        return self._volume

    def _change_volume(self, delta: float) -> CommandResult:
        volume = self._volume_interface()
        if volume is None:
            return CommandResult(True, False, "Не удалось изменить системную громкость")
        try:
            current = float(volume.GetMasterVolumeLevelScalar())
            target = max(0.0, min(1.0, current + delta))
            if abs(target - current) < 0.005:
                return CommandResult(True, True, f"Громкость уже {round(current * 100)}%")
            was_muted = bool(volume.GetMute())
            volume.SetMasterVolumeLevelScalar(target, None)
            if was_muted:
                volume.SetMute(0, None)
            actual = float(volume.GetMasterVolumeLevelScalar())
            message = f"Громкость: {round(actual * 100)}%"
            if was_muted:
                message += " (звук включён)"
            return CommandResult(True, True, message)
        except Exception as exc:  # noqa: BLE001
            log.warning("Ошибка изменения громкости: %s", exc)
            return CommandResult(True, False, "Не удалось изменить громкость")

    def _set_volume(self, level: float) -> CommandResult:
        volume = self._volume_interface()
        if volume is None:
            return CommandResult(True, False, "Не удалось установить системную громкость")
        try:
            target = max(0.0, min(1.0, level))
            was_muted = bool(volume.GetMute())
            volume.SetMasterVolumeLevelScalar(target, None)
            if was_muted:
                volume.SetMute(0, None)
            actual = float(volume.GetMasterVolumeLevelScalar())
            message = f"Громкость: {round(actual * 100)}%"
            if was_muted:
                message += " (звук включён)"
            return CommandResult(True, True, message)
        except Exception as exc:  # noqa: BLE001
            log.warning("Ошибка установки громкости: %s", exc)
            return CommandResult(True, False, "Не удалось установить громкость")

    def _set_mute(self, mute: bool) -> CommandResult:
        volume = self._volume_interface()
        if volume is None:
            return CommandResult(True, False, "Не удалось переключить звук")
        try:
            volume.SetMute(1 if mute else 0, None)
            return CommandResult(True, True, "Звук выключен" if mute else "Звук включён")
        except Exception as exc:  # noqa: BLE001
            log.warning("Ошибка переключения звука: %s", exc)
            return CommandResult(True, False, "Не удалось переключить звук")

    # -------------------------------------------------------------------- tabs
    def _tabs_command(self, t: str) -> CommandResult | None:
        if re.fullmatch(r"(новая вкладка|открой новую вкладку|создай вкладку|новую вкладку)", t):
            return self._simple("newtab", "Открываю новую вкладку")

        if re.fullmatch(r"(закрой вкладку|закрыть вкладку|закрой текущую вкладку)", t):
            return self._simple("closetab", "Закрываю вкладку")

        match = re.match(r"^(?:переключись|перейди)\s+на\s+вкладку\s+(\w+)$", t)
        if not match:
            match = re.match(r"^вкладка\s+(\w+)$", t)
        if match:
            raw = match.group(1)
            index = TAB_WORDS.get(raw)
            if index is None:
                try:
                    index = int(raw)
                except ValueError:
                    return CommandResult(True, False, f"Не понял номер вкладки: {raw}")
            return self._simple("tab", f"Переключаюсь на вкладку {index}", index=index)

        if re.fullmatch(r"(какие вкладки|список вкладок|покажи вкладки|вкладки)", t):
            return self._list_tabs()
        return None

    def _list_tabs(self) -> CommandResult:
        ok, data, error = self._call_browser("tabs")
        if not ok:
            return CommandResult(True, False, f"Ошибка: {error}")
        tabs = data.get("tabs", [])
        if not tabs:
            return CommandResult(True, True, "Вкладок нет")
        message = "; ".join(
            f"{tab['index']}. {(tab.get('title') or tab.get('url', ''))[:40]}"
            for tab in tabs
        )
        return CommandResult(True, True, f"Вкладки: {message}")

    # ------------------------------------------------------------------- media
    def _media_command(self, t: str) -> CommandResult | None:
        if re.fullmatch(r"(пауза|поставь на паузу|приостанови)", t):
            return self._simple("pause", "Пауза")
        if re.fullmatch(
            r"(стоп|стоп видео|останови|останови видео|останови музыку|остановись)",
            t,
        ):
            return self._simple("stop", "Останавливаю")
        if re.fullmatch(r"(продолжи|продолжим|продолжай|играй|плей|возобнови|включи видео)", t):
            return self._simple("play", "Продолжаю")
        if re.fullmatch(
            r"(следующее видео|следующий|дальше|переключи видео|следующий трек|следующая песня)",
            t,
        ):
            return self._simple("next", "Следующее видео")
        if re.fullmatch(
            r"(предыдущее видео|предыдущий|назад видео|предыдущий трек|предыдущая песня)", t
        ):
            return self._simple("prev", "Предыдущее видео")
        return None

    # ----------------------------------------------------------------- youtube
    @staticmethod
    def _youtube_query(t: str) -> str:
        query = t
        for phrase in (
            "на ютубе", "в ютубе", "ютубе", "on youtube", "on you tube",
            "youtube", "you tube", "ютуб",
        ):
            query = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", " ", query)
        for word in (
            "включи", "поставь", "запусти", "найди", "покажи", "открой",
            "мне", "пожалуйста", "видео", "клип", "песню", "трек", "ролик",
            "play", "find", "search", "open", "show", "look", "for", "me",
            "video", "clip", "song", "track",
        ):
            query = re.sub(rf"(?<!\w){word}(?!\w)", " ", query)
        return re.sub(r"\s+", " ", query).strip(" ,.-")

    # ------------------------------------------------------------- soundcloud
    def _ambiguous_soundcloud_volume(self, t: str) -> CommandResult | None:
        """«громкость <искажённое слово> пять» при открытом SoundCloud.

        GigaAM на фоне музыки теряет «саундклауда»: «громкость на паузу пять»,
        «громкость сомкала на пять». Если в конце число, а SoundCloud открыт,
        это почти наверняка громкость плеера.
        """
        if not _AMBIGUOUS_VOLUME_RE.search(t):
            return None
        head, _, tail = t.rpartition(" ")
        if not head or not tail or not _AMBIGUOUS_VOLUME_RE.search(head):
            return None
        percent = parse_volume_percent(tail)
        if percent is None:
            return None
        if not self._soundcloud_open():
            return None
        self._remember_soundcloud_volume()
        return self._simple(
            "soundcloud_volume",
            f"Громкость SoundCloud: {percent}%",
            mode="set",
            value=percent,
        )

    def _soundcloud_open(self) -> bool:
        ok, data, _ = self._call_browser("tabs", timeout=5)
        if not ok or not isinstance(data, dict):
            return False
        tabs = data.get("tabs") or []
        return any("soundcloud.com" in str(tab.get("url", "")).lower() for tab in tabs)

    @staticmethod
    def _likes_query(t: str) -> str:
        query = _SOUNDCLOUD_RE.sub(" ", t)
        query = re.sub(r"\s+", " ", query).strip(" ,.-")
        query = re.sub(r"^(?:на|в|у|для)(?:\s+|$)", "", query)
        query = re.sub(r"(?:^|\s+)(?:на|в|у|для)$", "", query)
        for word in (
            "включи", "поставь", "запусти", "играй", "слушай",
            "открой", "покажи", "зайди", "перейди",
            "мне", "мои", "мой", "пожалуйста",
            "лайкнутые", "лайкнутый", "лайкнутых", "лайкнутое",
            "лайки", "лайков", "лайкам", "лайке",
            "избранное", "избранные", "избранных", "избранный",
            "любимые", "любимых", "любимый",
            "треки", "трек", "треках", "треков", "трекам",
            "песни", "песню", "песня", "песнях", "песен", "песням",
            "музыку", "музыка", "музыке",
            "случайный", "случайную", "случайное",
            "рандомный", "рандомную", "рандомное",
            "любой", "любую", "любое", "первый", "первую",
            "попавшийся", "попавшуюся", "из",
        ):
            query = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", " ", query)
        query = re.sub(r"^(?:и|а)\s+", "", query.strip())
        return re.sub(r"\s+", " ", query).strip(" ,.-")

    @staticmethod
    def _soundcloud_query(t: str) -> str:
        query = _SOUNDCLOUD_RE.sub(" ", t)
        for word in (
            "включи", "поставь", "запусти", "найди", "покажи", "открой",
            "мне", "пожалуйста", "трек", "песню", "музыку", "послушать",
            "play", "find", "search", "listen", "open", "for", "me",
            "song", "track", "video", "clip",
        ):
            query = re.sub(rf"(?<!\w){word}(?!\w)", " ", query)
        query = re.sub(r"^\s*(?:на|в|у|для|on|in)\s+", " ", query)
        query = re.sub(r"\s+(?:на|в|у|для|on|in)\s*$", " ", query)
        return re.sub(r"\s+", " ", query).strip(" ,.-")

    # ------------------------------------------------------------------ search
    def _search_command(self, t: str) -> CommandResult | None:
        match = re.match(
            r"^(?:загугли|погугли|найди в гугле|поищи в гугле|найди в интернете)\s+(.+)$",
            t,
        )
        if match:
            query = match.group(1).strip()
            self._remember_google_search(query)
            return self._simple("google", f"Ищу: {query}", query=query)

        match = re.match(r"^(?:найди|поищи)\s+(.+)$", t)
        if match:
            query = re.sub(r"^(?:(?:мне|пожалуйста)\s+)+", "", match.group(1).strip())
            self._remember_google_search(query)
            return self._simple("google", f"Ищу: {query}", query=query)

        match = re.match(
            r"^(?:search(?:\s+for)?|find|look\s+up|google)\s+(.+)$", t
        )
        if match:
            query = re.sub(r"^(?:(?:me|for)\s+)+", "", match.group(1).strip())
            query = re.sub(r"\s+(?:on|in)\s+(?:google|the\s+web)$", "", query)
            self._remember_google_search(query)
            return self._simple("google", f"Ищу: {query}", query=query)
        return None

    # -------------------------------------------------------------------- open
    def _result_command(self, t: str) -> CommandResult | None:
        """«открой первую ссылку», «открой результат 2» — из текущей выдачи Google."""
        match = re.fullmatch(
            r"(?:(?:открой|открыть|перейди(?:\s+на)?|нажми(?:\s+на)?)\s+)?"
            r"(?:на\s+)?(?:(\d+)|([а-яё]+))\s+"
            r"(?:ссылк[а-яё]*|результат[а-яё]*)"
            r"(?:\s+(?:в|из)\s+(?:гугле|google|выдаче))?",
            t,
        )
        if match:
            raw = (match.group(1) or match.group(2) or "").strip()
        else:
            match = re.fullmatch(
                r"(?:открой|открыть|перейди(?:\s+на)?|нажми(?:\s+на)?)\s+"
                r"(?:ссылк[а-яё]*|результат[а-яё]*)\s*(?:номер\s*)?(\d+)"
                r"(?:\s+(?:в|из)\s+(?:гугле|google|выдаче))?",
                t,
            )
            if not match:
                return None
            raw = match.group(1)

        index = int(raw) if raw.isdigit() else RESULT_WORDS.get(raw)
        if index is None or index < 1:
            return None
        self.context.remember("results", "search_result", index=index)
        return self._simple("search_result", f"Открываю ссылку №{index}", index=index)

    def _open_command(self, t: str) -> CommandResult | None:
        match = re.match(r"^открой\s+(.+)$", t)
        if not match:
            match = re.match(r"^(?:open|go\s+to)\s+(.+)$", t)
        if not match:
            return None
        target = re.sub(r"^(сайт|страницу|вкладку)\s+", "", match.group(1).strip())
        if not target:
            return None

        if target in SITE_URLS:
            self._remember_media("open_site", site=target)
            return self._simple("goto", f"Открываю {target}", url=SITE_URLS[target])

        if re.fullmatch(r"[a-zа-я0-9.-]+\.[a-zа-я]{2,}", target):
            self._remember_media("open_site", site=target)
            return self._simple("goto", f"Открываю {target}", url="https://" + target)

        self._remember_google_search(target)
        return self._simple("google", f"Ищу: {target}", query=target)

    # -------------------------------------------------------------- navigation
    def _navigation_command(self, t: str) -> CommandResult | None:
        if re.fullmatch(r"(назад|вернись назад|страница назад)", t):
            return self._simple("back", "Назад")
        if re.fullmatch(r"(вперёд|вперед|страница вперёд)", t):
            return self._simple("forward", "Вперёд")
        if re.fullmatch(r"(обнови|обнови страницу|перезагрузи страницу)", t):
            return self._simple("reload", "Обновляю страницу")
        return None

    # -------------------------------------------------------------------- misc
    def _misc_command(self, t: str) -> CommandResult | None:
        if re.fullmatch(r"(прокрути вниз|листай вниз|вниз страницы)", t):
            return self._simple("scroll", "Листаю вниз", direction="down")
        if re.fullmatch(r"(прокрути вверх|листай вверх|вверх страницы)", t):
            return self._simple("scroll", "Листаю вверх", direction="up")
        if re.fullmatch(r"(во весь экран|на весь экран|разверни видео|полный экран)", t):
            ok, data, error = self._call_browser("status")
            if ok and "youtube.com" in str(data.get("url", "")):
                return self._simple("press", "Включаю полный экран", key="f")
            return self._simple(
                "eval",
                "Включаю полный экран",
                script="document.documentElement.requestFullscreen().catch(() => {})",
            )
        if re.fullmatch(r"(что играет|что сейчас играет|какая вкладка|где я|статус)", t):
            ok, data, error = self._call_browser("status")
            if not ok:
                return CommandResult(True, False, f"Ошибка: {error}")
            title = data.get("title") or "(без названия)"
            url = data.get("url") or ""
            return CommandResult(True, True, f"{title} — {url}")
        return None

    # ----------------------------------------------------------------- helpers
    def _remember_media(self, action: str, site: str | None = None) -> None:
        if site is None:
            previous = self.context.get("search") or self.context.get("media")
            site = str(previous.slots.get("site") or "") if previous else ""
        self.context.remember("media", action, site=site)

    def _remember_google_search(self, query: str) -> None:
        self.context.remember(
            "search", "google_search", site="google", query=query, index=1
        )
        self.context.remember(
            "results", "google_search", query=query, index=1
        )

    def _google_query_from_context(self) -> str:
        for kind in ("results", "search"):
            entry = self.context.get(kind)
            if entry is None:
                continue
            query = str(entry.slots.get("query") or "")
            if query:
                return query
        return ""

    def _remember_youtube_search(self, query: str) -> None:
        self.context.remember(
            "search", "youtube_search", site="youtube", query=query, index=1
        )
        self.context.remember("media", "youtube_search", site="youtube")

    def _remember_system_volume(self) -> None:
        self.context.remember("volume", "system_volume", target="system")

    def _remember_soundcloud_volume(self) -> None:
        self.context.remember("volume", "soundcloud_volume", target="soundcloud")

    @staticmethod
    def _clean_google_query(query: str) -> str:
        query = re.sub(r"^(?:(?:мне|пожалуйста)\s+)+", "", query.strip())
        query = re.sub(r"^(?:(?:me|for)\s+)+", "", query)
        query = re.sub(r"\s+(?:on|in)\s+(?:google|the\s+web)$", "", query)
        return query.strip(" ,.-")

    def _call_browser(self, action: str, timeout: float = 90, **args: Any) -> tuple[bool, Any, str]:
        try:
            result = self.browser.call(action, timeout=timeout, **args)
        except Exception as exc:  # noqa: BLE001
            log.exception("Ошибка вызова браузера: %s", action)
            return False, None, str(exc)
        if not result.get("ok"):
            return False, None, str(result.get("error", "неизвестная ошибка"))
        return True, result.get("data") or {}, ""

    def _simple(self, action: str, message: str, **args: Any) -> CommandResult:
        ok, data, error = self._call_browser(action, **args)
        if not ok:
            return CommandResult(True, False, f"Не получилось: {error}")
        message = self._refine_message(action, message, data)
        return CommandResult(True, True, message)

    @staticmethod
    def _refine_message(action: str, message: str, data: Any) -> str:
        if not isinstance(data, dict):
            return message
        title = (data.get("title") or "").strip()
        if action in {"youtube", "soundcloud"} and title:
            return f"Включаю: {title[:120]}"
        if action == "soundcloud_volume":
            mode = data.get("mode")
            if mode == "mute":
                return "Звук SoundCloud выключен"
            if mode == "unmute":
                return "Звук SoundCloud включён"
            return f"Громкость SoundCloud: {data.get('volume', 0)}%"
        if action == "soundcloud_likes":
            if data.get("mode") == "open":
                return f"Лайкнутые треки: {data.get('count', 0)} шт."
            if title:
                return f"Включаю из лайкнутых: {title[:120]}"
        if action == "search_result" and title:
            return f"Открываю: {title[:120]}"
        if action in {"goto", "google"} and title:
            return f"{message}: {title[:80]}"
        return message
