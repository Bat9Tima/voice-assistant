from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from .text import normalize

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

_TAB_WORDS = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "один": 1, "два": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
}

_RESULT_WORDS = {
    "первую": 1, "первый": 1, "первая": 1, "первое": 1,
    "вторую": 2, "второй": 2, "вторая": 2, "второе": 2,
    "третью": 3, "третий": 3, "третья": 3, "третье": 3,
    "четвертую": 4, "четвёртую": 4, "четвертый": 4, "четвёртый": 4,
    "пятую": 5, "пятый": 5, "пятая": 5, "пятое": 5,
    "шестую": 6, "шестой": 6, "шестая": 6, "шестое": 6,
    "седьмую": 7, "седьмой": 7, "седьмая": 7, "седьмое": 7,
    "восьмую": 8, "восьмой": 8, "восьмая": 8, "восьмое": 8,
    "девятую": 9, "девятый": 9, "девятая": 9, "девятое": 9,
    "десятую": 10, "десятый": 10, "десятая": 10, "десятое": 10,
}

VOLUME_STEP = 0.10
_SOUNDCLOUD_RE = re.compile(
    r"сау?н[дт]?\s*к?л[ао]у[дт]\w*|sound\s*cloud",
    re.UNICODE,
)

_NUMBER_WORDS = {
    "ноль": 0, "нуль": 0,
    "один": 1, "одна": 1, "одну": 1,
    "два": 2, "две": 2,
    "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
    "десять": 10, "одиннадцать": 11, "двенадцать": 12,
    "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15,
    "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
    "девятнадцать": 19, "двадцать": 20, "тридцать": 30,
    "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
    "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
    "сто": 100,
}


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
        if token in _NUMBER_WORDS:
            total += _NUMBER_WORDS[token]
            seen = True
        else:
            return None
    if not seen:
        return None
    return to_percent(total)


@dataclass
class CommandResult:
    handled: bool
    ok: bool = True
    message: str = ""


class ActionRouter:
    """Быстрые детерминированные команды — без обращения к LLM."""

    def __init__(self, cfg: dict[str, Any], browser: Any) -> None:
        self.cfg = cfg
        self.browser = browser
        self._volume = None

    # ------------------------------------------------------------------ public
    def handle(self, text: str) -> CommandResult | None:
        t = normalize(text)
        if not t:
            return CommandResult(True, True, "")

        if t in CANCEL_WORDS:
            return CommandResult(True, True, "Отменено")

        soundcloud = self._soundcloud_command(t)
        if soundcloud:
            return soundcloud

        volume = self._volume_command(t)
        if volume:
            return volume

        tabs = self._tabs_command(t)
        if tabs:
            return tabs

        media = self._media_command(t)
        if media:
            return media

        if "ютуб" in t or "youtube" in t or "you tube" in t:
            if re.search(
                r"(включи|поставь|запусти|найди|покажи|открой|"
                r"play|find|search|open|show|look)",
                t,
            ):
                query = self._youtube_query(t)
                if query:
                    return self._simple("youtube", f"Включаю на YouTube: {query}", query=query)
                return self._simple("goto", "Открываю YouTube", url=SITE_URLS["ютуб"])

        match = re.match(
            r"^(?:включи|поставь|запусти)\s+(?:видео|клип|песню|трек|ролик)\s+(.+)$", t
        )
        if match:
            query = match.group(1).strip()
            return self._simple("youtube", f"Включаю на YouTube: {query}", query=query)

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

        return None

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

    def _volume_command(self, t: str) -> CommandResult | None:
        up = re.fullmatch(
            r"(?:громче|погромче|прибавь(?:\s+(?:звук|громкость))?|сделай\s+громче|"
            r"увеличь(?:\s+(?:звук|громкость))?|подними(?:\s+громкость)?|"
            r"громкость\s+выше|добавь\s+звук)(?:\s+на\s+(.+))?",
            t,
        )
        if up:
            return self._change_volume(self._volume_delta(up.group(1), 1))

        down = re.fullmatch(
            r"(?:тише|потише|убавь(?:\s+(?:звук|громкость))?|сделай\s+тише|"
            r"уменьши(?:\s+(?:звук|громкость))?|снизь(?:\s+громкость)?|"
            r"громкость\s+ниже|убери\s+звук)(?:\s+на\s+(.+))?",
            t,
        )
        if down:
            return self._change_volume(self._volume_delta(down.group(1), -1))

        if re.fullmatch(
            r"(?:выключи|отключи|убери|заглуши)\s+(?:звук|громкость)|без\s+звука|мьют|mute",
            t,
        ):
            return self._set_mute(True)
        if re.fullmatch(r"(?:включи|верни)\s+(?:звук|громкость)|unmute", t):
            return self._set_mute(False)

        match = re.fullmatch(
            r"(?:установи|поставь|сделай|выставь)\s+(?:громкость|звук)\s*(?:на\s+)?(.+)",
            t,
        )
        if not match:
            match = re.fullmatch(r"(?:громкость|звук)\s*(?:на\s+)?(.+)", t)
        if match:
            raw = re.sub(
                r"\b(?:поставь|поставил|установи|установил|сделай|сделал|пожалуйста)\b",
                " ",
                match.group(1),
            )
            percent = parse_volume_percent(raw)
            if percent is not None:
                return self._set_volume(percent / 100)
        return None

    @staticmethod
    def _volume_delta(text: str | None, sign: int) -> float:
        if text:
            percent = parse_volume_percent(text)
            if percent is not None and percent > 0:
                return sign * percent / 100
        return sign * VOLUME_STEP

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
            index = _TAB_WORDS.get(raw)
            if index is None:
                try:
                    index = int(raw)
                except ValueError:
                    return CommandResult(True, False, f"Не понял номер вкладки: {raw}")
            return self._simple("tab", f"Переключаюсь на вкладку {index}", index=index)

        if re.fullmatch(r"(какие вкладки|список вкладок|покажи вкладки|вкладки)", t):
            ok, data, error = self._call_browser("tabs")
            if not ok:
                return CommandResult(True, False, f"Ошибка: {error}")
            tabs = data.get("tabs", [])
            if not tabs:
                return CommandResult(True, True, "Вкладок нет")
            message = "; ".join(
                f"{tab['index']}. {(tab.get('title') or tab.get('url', ''))[:40]}" for tab in tabs
            )
            return CommandResult(True, True, f"Вкладки: {message}")
        return None

    # ------------------------------------------------------------------- media
    def _media_command(self, t: str) -> CommandResult | None:
        if re.fullmatch(
            r"(пауза|поставь на паузу|останови видео|стоп видео|приостанови|стоп)", t
        ):
            return self._simple("pause", "Пауза")
        if re.fullmatch(r"(продолжи|продолжай|играй|плей|возобнови|включи видео)", t):
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
    def _soundcloud_command(self, t: str) -> CommandResult | None:
        if not _SOUNDCLOUD_RE.search(t):
            return None

        volume = self._soundcloud_volume_command(t)
        if volume:
            return volume

        if re.search(r"(?:лайк|избранн|любим)", t):
            query = self._likes_query(t)
            wants_play = bool(
                re.search(r"(?:включи|поставь|запусти|играй|слушай|play|listen)", t)
            )
            wants_open = bool(re.search(r"(?:открой|покажи|зайди|перейди)", t))
            if wants_play:
                random_choice = bool(
                    re.search(r"(?:случайн|рандом|любой|первый\s+попавш)", t)
                ) or not query
                message = (
                    "Включаю случайный лайкнутый трек"
                    if random_choice
                    else f"Ищу среди лайкнутых: {query}"
                )
                return self._simple(
                    "soundcloud_likes",
                    message,
                    query=query,
                    random=random_choice,
                    play=True,
                )
            if wants_open or not query:
                return self._simple(
                    "soundcloud_likes", "Открываю лайкнутые треки SoundCloud", play=False
                )

        core = self._soundcloud_core(t)
        if re.fullmatch(r"(?:пауза|поставь\s+на\s+паузу|приостанови|стоп)", core):
            return self._simple("pause", "Пауза")
        if re.fullmatch(
            r"(?:продолжи|продолжай|играй|плей|возобнови|включи\s+видео)", core
        ):
            return self._simple("play", "Продолжаю")
        if re.fullmatch(
            r"(?:(?:включи|поставь|переключи)\s+)?(?:на\s+)?"
            r"(?:следующ(?:ий|ую|ее)|дальше)(?:\s+(?:трек|песню|видео))?",
            core,
        ):
            return self._simple("next", "Следующий трек")
        if re.fullmatch(
            r"(?:(?:включи|поставь|переключи)\s+)?(?:на\s+)?"
            r"(?:предыдущ(?:ий|ую|ее))(?:\s+(?:трек|песню|видео))?",
            core,
        ):
            return self._simple("prev", "Предыдущий трек")

        query = self._soundcloud_query(t)
        if query and re.search(
            r"(?:включи|поставь|запусти|найди|покажи|играй|слушай|"
            r"play|find|search|listen)",
            t,
        ):
            return self._simple("soundcloud", f"Включаю на SoundCloud: {query}", query=query)
        return self._simple("goto", "Открываю SoundCloud", url=SITE_URLS["саундклауд"])

    def _soundcloud_volume_command(self, t: str) -> CommandResult | None:
        if not _SOUNDCLOUD_RE.search(t):
            return None
        core = self._soundcloud_core(t)
        if not core:
            return None

        if re.fullmatch(
            r"(?:выключи|отключи|убери|заглуши)\s+(?:звук|громкость)|"
            r"без\s+звука|мьют|mute",
            core,
        ):
            return self._simple(
                "soundcloud_volume", "Выключаю звук SoundCloud", mode="mute"
            )
        if re.fullmatch(r"(?:включи|верни)\s+(?:звук|громкость)|unmute", core):
            return self._simple(
                "soundcloud_volume", "Включаю звук SoundCloud", mode="unmute"
            )

        if re.search(
            r"(?:громче|погромче|тише|потише|прибавь|убавь|увеличь|уменьши|"
            r"подними|снизь|громкость\s+выше|громкость\s+ниже)",
            core,
        ):
            mode = (
                "up"
                if re.search(
                    r"(?:громче|погромче|прибавь|увеличь|подними|выше)", core
                )
                else "down"
            )
            amount: int | None = None
            match = re.search(r"\bна\s+(.+)$", core)
            if match:
                amount = parse_volume_percent(match.group(1))
            step = amount if amount is not None else round(VOLUME_STEP * 100)
            return self._simple(
                "soundcloud_volume",
                "Делаю громче в SoundCloud" if mode == "up" else "Делаю тише в SoundCloud",
                mode=mode,
                value=step,
            )

        match = re.fullmatch(
            r"(?:установи|поставь|сделай|выставь)\s+(?:громкость|звук)\s*(?:на\s+)?(.+)",
            core,
        )
        if not match:
            match = re.fullmatch(r"(?:громкость|звук)\s*(?:на\s+)?(.+)", core)
        if match:
            raw = re.sub(
                r"\b(?:поставь|поставил|установи|установил|сделай|сделал|пожалуйста)\b",
                " ",
                match.group(1),
            )
            percent = parse_volume_percent(raw)
            if percent is not None:
                return self._simple(
                    "soundcloud_volume",
                    f"Устанавливаю громкость SoundCloud: {percent}%",
                    mode="set",
                    value=percent,
                )
        return None

    @staticmethod
    def _soundcloud_core(t: str) -> str:
        core = _SOUNDCLOUD_RE.sub(" ", t)
        core = re.sub(r"\s+", " ", core).strip(" ,.-")
        core = re.sub(r"^(?:на|в|у|для)\s+", "", core)
        core = re.sub(r"\s+(?:на|в|у|для)$", "", core)
        return core

    @staticmethod
    def _likes_query(t: str) -> str:
        query = _SOUNDCLOUD_RE.sub(" ", t)
        query = re.sub(r"\s+", " ", query).strip(" ,.-")
        query = re.sub(r"^(?:на|в|у|для)\s+", "", query)
        query = re.sub(r"\s+(?:на|в|у|для)$", "", query)
        for word in (
            "включи", "поставь", "запусти", "играй", "слушай",
            "открой", "покажи", "зайди", "перейди",
            "мне", "мои", "мой", "пожалуйста",
            "лайкнутые", "лайкнутый", "лайкнутых", "лайкнутое",
            "лайки", "лайков", "лайкам", "лайке",
            "избранное", "избранные", "избранных", "избранный",
            "любимые", "любимых", "любимый",
            "треки", "трек", "песни", "песню", "песня", "музыку", "музыка",
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
            return self._simple("google", f"Ищу: {query}", query=query)

        match = re.match(r"^(?:найди|поищи)\s+(.+)$", t)
        if match:
            query = re.sub(r"^(?:(?:мне|пожалуйста)\s+)+", "", match.group(1).strip())
            return self._simple("google", f"Ищу: {query}", query=query)

        match = re.match(
            r"^(?:search(?:\s+for)?|find|look\s+up|google)\s+(.+)$", t
        )
        if match:
            query = re.sub(r"^(?:(?:me|for)\s+)+", "", match.group(1).strip())
            query = re.sub(r"\s+(?:on|in)\s+(?:google|the\s+web)$", "", query)
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

        index = int(raw) if raw.isdigit() else _RESULT_WORDS.get(raw)
        if index is None or index < 1:
            return None
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
            return self._simple("goto", f"Открываю {target}", url=SITE_URLS[target])

        if re.fullmatch(r"[a-zа-я0-9.-]+\.[a-zа-я]{2,}", target):
            return self._simple("goto", f"Открываю {target}", url="https://" + target)

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
    def _call_browser(self, action: str, **args: Any) -> tuple[bool, Any, str]:
        try:
            result = self.browser.call(action, timeout=90, **args)
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
