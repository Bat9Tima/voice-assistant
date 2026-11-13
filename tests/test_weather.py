from __future__ import annotations

import unittest
from unittest.mock import patch

from app.weather import (
    GEOCODE_URL,
    DailyForecast,
    GeoPlace,
    WeatherClient,
    WeatherError,
    WeatherReport,
    city_candidates,
    describe_code,
    format_report,
)

GEO = {
    "results": [
        {
            "name": "Новосибирск",
            "country": "Россия",
            "latitude": 55.02,
            "longitude": 82.93,
            "timezone": "Asia/Novosibirsk",
        }
    ]
}

FORECAST = {
    "current": {
        "temperature_2m": -3.0,
        "apparent_temperature": -7.0,
        "weather_code": 71,
        "wind_speed_10m": 4.0,
        "relative_humidity_2m": 80,
    },
    "daily": {
        "time": ["2026-11-11", "2026-11-12", "2026-11-13"],
        "temperature_2m_max": [-1.0, 0.0, 1.0],
        "temperature_2m_min": [-6.0, -5.0, -4.0],
        "weather_code": [71, 3, 0],
    },
}


def make_report() -> WeatherReport:
    return WeatherReport(
        place=GeoPlace("Новосибирск", "Россия", 55.02, 82.93),
        temperature=-3.0,
        feels_like=-7.0,
        code=71,
        wind=4.0,
        humidity=80,
        daily=(
            DailyForecast("2026-11-11", -6.0, -1.0, 71),
            DailyForecast("2026-11-12", -5.0, 0.0, 3),
        ),
    )


class CityCandidatesTest(unittest.TestCase):
    def test_prepositional_masculine(self) -> None:
        self.assertIn("Новосибирск", city_candidates("Новосибирске"))
        self.assertIn("Лондон", city_candidates("Лондоне"))

    def test_prepositional_feminine(self) -> None:
        self.assertIn("Москва", city_candidates("Москве"))
        self.assertIn("Казань", city_candidates("Казани"))

    def test_nominative_is_kept(self) -> None:
        self.assertEqual(city_candidates("Москва")[0], "Москва")

    def test_empty(self) -> None:
        self.assertEqual(city_candidates("  "), ())


class FormatTest(unittest.TestCase):
    def test_report_contains_city_and_days(self) -> None:
        text = format_report(make_report())
        self.assertIn("Новосибирск", text)
        self.assertIn("-3 °C", text)
        self.assertIn("Сегодня", text)
        self.assertIn("Завтра", text)

    def test_positive_temperature_has_plus(self) -> None:
        report = make_report()
        report = WeatherReport(report.place, 5.0, 3.0, 0, 2.0, 50, report.daily)
        self.assertIn("+5 °C", format_report(report))

    def test_unknown_code(self) -> None:
        self.assertEqual(describe_code(12345), "неизвестная погода")


class GeocodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.client = WeatherClient({"qa": {"weather": {"city": "Новосибирск", "timeout_sec": 1}}})

    def fake_http(self, url: str, params: dict, timeout: float) -> dict:
        self.calls.append((url, dict(params)))
        if url == GEOCODE_URL:
            if params["name"] == "Новосибирск":
                return GEO
            return {"results": []}
        return FORECAST

    def test_finds_nominative_from_prepositional(self) -> None:
        with patch("app.weather._http_json", side_effect=self.fake_http):
            place = self.client.geocode("Новосибирске")
        self.assertEqual(place.name, "Новосибирск")
        self.assertEqual(place.country, "Россия")

    def test_caches_geocode(self) -> None:
        with patch("app.weather._http_json", side_effect=self.fake_http):
            self.client.geocode("Новосибирске")
            self.client.geocode("Новосибирске")
        geocode_calls = [call for call in self.calls if call[0] == GEOCODE_URL]
        self.assertEqual(len(geocode_calls), 2)

    def test_unknown_city(self) -> None:
        with patch("app.weather._http_json", side_effect=self.fake_http):
            with self.assertRaises(WeatherError):
                self.client.geocode("Атлантида")

    def test_forecast_parses_current_and_daily(self) -> None:
        with patch("app.weather._http_json", side_effect=self.fake_http):
            report = self.client.forecast("Новосибирске")
        self.assertEqual(report.temperature, -3.0)
        self.assertEqual(report.feels_like, -7.0)
        self.assertEqual(report.humidity, 80)
        self.assertEqual(len(report.daily), 3)
        self.assertEqual(report.daily[0].minimum, -6.0)

    def test_incomplete_forecast(self) -> None:
        with patch(
            "app.weather._http_json",
            side_effect=lambda url, params, timeout: GEO if url == GEOCODE_URL else {"current": {}},
        ):
            with self.assertRaises(WeatherError):
                self.client.forecast("Новосибирске")

    def test_default_city_from_config(self) -> None:
        with patch("app.weather._http_json", side_effect=self.fake_http):
            report = self.client.forecast()
        self.assertEqual(report.place.name, "Новосибирск")


if __name__ == "__main__":
    unittest.main()
