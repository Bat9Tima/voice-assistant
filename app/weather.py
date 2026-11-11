from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Коды WMO, которые возвращает Open-Meteo.
WEATHER_CODES: dict[int, str] = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "изморозь",
    51: "слабая морось",
    53: "морось",
    55: "сильная морось",
    56: "ледяная морось",
    57: "сильная ледяная морось",
    61: "небольшой дождь",
    63: "дождь",
    65: "сильный дождь",
    66: "ледяной дождь",
    67: "сильный ледяной дождь",
    71: "небольшой снег",
    73: "снег",
    75: "сильный снег",
    77: "снежная крупа",
    80: "небольшие ливни",
    81: "ливни",
    82: "сильные ливни",
    85: "небольшой снегопад",
    86: "сильный снегопад",
    95: "гроза",
    96: "гроза с градом",
    99: "сильная гроза с градом",
}


class WeatherError(RuntimeError):
    """Погодный сервис недоступен или город не найден."""


@dataclass(frozen=True)
class GeoPlace:
    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str = "auto"


@dataclass(frozen=True)
class DailyForecast:
    date: str
    minimum: float
    maximum: float
    code: int


@dataclass(frozen=True)
class WeatherReport:
    place: GeoPlace
    temperature: float
    feels_like: float
    code: int
    wind: float
    humidity: int
    daily: tuple[DailyForecast, ...] = field(default_factory=tuple)


def _http_json(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    full = url + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(full, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise WeatherError(f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise WeatherError(f"сервис погоды недоступен: {exc}") from exc
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise WeatherError("сервис погоды вернул не JSON") from exc
    if not isinstance(data, dict):
        raise WeatherError("сервис погоды вернул неверный ответ")
    return data


def _clean_name(value: str) -> str:
    return value.strip().lower().replace("ё", "е")


def _same_name(left: str, right: str) -> bool:
    return _clean_name(left) == _clean_name(right)


def city_candidates(city: str) -> tuple[str, ...]:
    """Возвращает именительный падеж и запасные формы для геокодинга.

    STT слышит «погода в Новосибирске», а Open-Meteo знает только
    «Новосибирск». Поэтому пробуем отбросить падежное окончание.
    """
    city = city.strip()
    if not city:
        return ()
    candidates = [city]
    if len(city) >= 4:
        last = city[-1].lower()
        stem = city[:-1]
        if last == "е":
            candidates.extend([stem, stem + "а", stem + "й"])
        elif last == "и":
            candidates.extend([stem + "ь", stem + "я"])
        elif last == "у":
            candidates.extend([stem + "а", stem + "ь"])
        elif last == "ю":
            candidates.extend([stem + "я", stem + "ь"])
    seen: set[str] = set()
    result: list[str] = []
    for candidate in candidates:
        key = _clean_name(candidate)
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return tuple(result)


def describe_code(code: int) -> str:
    return WEATHER_CODES.get(int(code), "неизвестная погода")


def _temp(value: float) -> str:
    rounded = round(value)
    return f"+{rounded} °C" if rounded > 0 else f"{rounded} °C"


def _wind(value: float) -> str:
    rounded = round(value, 1)
    text = str(int(rounded)) if rounded == int(rounded) else f"{rounded:.1f}"
    return f"{text} м/с"


def _day_label(index: int) -> str:
    if index == 0:
        return "Сегодня"
    if index == 1:
        return "Завтра"
    if index == 2:
        return "Послезавтра"
    return "Далее"


def format_report(report: WeatherReport) -> str:
    parts = [
        f"{report.place.name}: сейчас {_temp(report.temperature)}, "
        f"ощущается как {_temp(report.feels_like)}, "
        f"{describe_code(report.code)}, ветер {_wind(report.wind)}, "
        f"влажность {report.humidity}%."
    ]
    for index, day in enumerate(report.daily[:3]):
        parts.append(
            f"{_day_label(index)}: от {_temp(day.minimum)} до {_temp(day.maximum)}, "
            f"{describe_code(day.code)}."
        )
    return " ".join(parts)


class WeatherClient:
    """Погода через Open-Meteo: без ключа и без поисковой выдачи."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        section = ((cfg or {}).get("qa") or {}).get("weather") or {}
        self.default_city = str(section.get("city") or "Москва").strip() or "Москва"
        self.forecast_days = max(1, min(int(section.get("forecast_days") or 3), 7))
        self.timeout = float(section.get("timeout_sec") or 10)
        self._cache: dict[str, GeoPlace] = {}

    def geocode(self, city: str) -> GeoPlace:
        city = (city or "").strip()
        if not city:
            raise WeatherError("не указан город")
        key = _clean_name(city)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        lookups: dict[str, list[GeoPlace]] = {}

        def lookup(candidate: str) -> list[GeoPlace]:
            if candidate not in lookups:
                lookups[candidate] = self._lookup(candidate)
            return lookups[candidate]

        candidates = city_candidates(city)
        for candidate in candidates:
            for place in lookup(candidate):
                if _same_name(place.name, candidate):
                    self._cache[key] = place
                    return place
        for candidate in candidates:
            for place in lookup(candidate):
                if _clean_name(place.country) in {"россия", "russia"}:
                    self._cache[key] = place
                    return place
        for candidate in candidates:
            places = lookup(candidate)
            if places:
                self._cache[key] = places[0]
                return places[0]
        raise WeatherError(f"город не найден: {city}")

    def forecast(self, city: str | None = None) -> WeatherReport:
        place = self.geocode(city or self.default_city)
        data = _http_json(
            FORECAST_URL,
            {
                "latitude": place.latitude,
                "longitude": place.longitude,
                "current": (
                    "temperature_2m,apparent_temperature,weather_code,"
                    "wind_speed_10m,relative_humidity_2m"
                ),
                "daily": "temperature_2m_max,temperature_2m_min,weather_code",
                "timezone": "auto",
                "forecast_days": self.forecast_days,
                "wind_speed_unit": "ms",
            },
            self.timeout,
        )
        current = data.get("current") or {}
        daily = data.get("daily") or {}
        try:
            return WeatherReport(
                place=place,
                temperature=float(current["temperature_2m"]),
                feels_like=float(current["apparent_temperature"]),
                code=int(current["weather_code"]),
                wind=float(current["wind_speed_10m"]),
                humidity=int(current["relative_humidity_2m"]),
                daily=_daily(daily),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WeatherError("сервис погоды вернул неполные данные") from exc

    def _lookup(self, name: str) -> list[GeoPlace]:
        data = _http_json(
            GEOCODE_URL,
            {"name": name, "count": 5, "language": "ru", "format": "json"},
            self.timeout,
        )
        results = data.get("results") or []
        places: list[GeoPlace] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                places.append(
                    GeoPlace(
                        name=str(item.get("name") or ""),
                        country=str(item.get("country") or ""),
                        latitude=float(item["latitude"]),
                        longitude=float(item["longitude"]),
                        timezone=str(item.get("timezone") or "auto"),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return places


def _daily(data: dict[str, Any]) -> tuple[DailyForecast, ...]:
    dates = data.get("time") or []
    maxima = data.get("temperature_2m_max") or []
    minima = data.get("temperature_2m_min") or []
    codes = data.get("weather_code") or []
    result: list[DailyForecast] = []
    for index, date in enumerate(dates):
        try:
            result.append(
                DailyForecast(
                    date=str(date),
                    minimum=float(minima[index]),
                    maximum=float(maxima[index]),
                    code=int(codes[index]),
                )
            )
        except (IndexError, TypeError, ValueError):
            continue
    return tuple(result)
