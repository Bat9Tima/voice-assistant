from __future__ import annotations

import unittest

from app.local_llm import LocalLLMError
from app.qa import (
    QAAnswer,
    QAService,
    QASource,
    extract_city,
    first_sentence,
    is_news_question,
    is_question,
    is_weather_question,
    needs_search,
    search_query,
)
from app.search import SearchError, SearchResult
from app.weather import DailyForecast, GeoPlace, WeatherError, WeatherReport


class FakeWeather:
    def __init__(self, report: WeatherReport | None = None, error: Exception | None = None):
        self.report = report
        self.error = error
        self.cities: list[str | None] = []

    def forecast(self, city: str | None = None) -> WeatherReport:
        self.cities.append(city)
        if self.error is not None:
            raise self.error
        assert self.report is not None
        return self.report


class FakeSearch:
    def __init__(
        self,
        results: list[SearchResult] | None = None,
        available: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.results = results or []
        self.is_available = available
        self.error = error
        self.calls: list[tuple[str, bool]] = []

    def available(self) -> bool:
        return self.is_available

    def search(self, query: str, *, news: bool = False) -> list[SearchResult]:
        self.calls.append((query, news))
        if self.error is not None:
            raise self.error
        return self.results


class FakeLLM:
    def __init__(
        self,
        text: str = "Ответ [1].",
        available: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.is_available = available
        self.error = error
        self.calls: list[tuple[str, str]] = []
        self.kwargs: list[dict] = []

    def available(self) -> bool:
        return self.is_available

    def answer(self, question: str, context: str = "", **kwargs) -> str:
        self.calls.append((question, context))
        self.kwargs.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return self.text


class FakeBrowser:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, action: str, **args) -> dict:
        self.calls.append((action, args))
        return {"ok": True}


def make_report() -> WeatherReport:
    return WeatherReport(
        place=GeoPlace("Новосибирск", "Россия", 55.02, 82.93),
        temperature=-3.0,
        feels_like=-7.0,
        code=71,
        wind=4.0,
        humidity=80,
        daily=(DailyForecast("2026-11-11", -6.0, -1.0, 71),),
    )


def make_service(llm=None, browser=None, enabled: bool = True) -> QAService:
    cfg = {
        "qa": {
            "enabled": enabled,
            "weather": {"city": "Новосибирск"},
            "search": {"api_key": ""},
        }
    }
    return QAService(cfg, llm, browser)


class QuestionDetectionTest(unittest.TestCase):
    def test_questions(self) -> None:
        for text in (
            "какая погода",
            "что такое фотосинтез",
            "расскажи про Python",
            "правда ли, что Земля круглая",
            "будет ли дождь",
            "сколько стоит билет",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_question(text))

    def test_commands_are_not_questions(self) -> None:
        for text in (
            "открой ютуб",
            "включи музыку",
            "сделай что-нибудь странное",
            "есть хочу",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_question(text))

    def test_weather_detection(self) -> None:
        self.assertTrue(is_weather_question("какая погода"))
        self.assertTrue(is_weather_question("будет ли дождь завтра"))
        self.assertFalse(is_weather_question("расскажи про климат"))

    def test_news_detection(self) -> None:
        self.assertTrue(is_news_question("какие новости"))
        self.assertFalse(is_news_question("что такое фотосинтез"))

    def test_search_markers(self) -> None:
        self.assertTrue(needs_search("какие новости"))
        self.assertTrue(needs_search("сколько стоит билет"))
        self.assertTrue(needs_search("поищи рецепт борща"))
        self.assertFalse(needs_search("что такое фотосинтез"))
        self.assertFalse(needs_search("расскажи про Python"))


class SearchQueryTest(unittest.TestCase):
    def test_generic_news_request_becomes_concrete(self) -> None:
        for text in (
            "какие новости",
            "что нового",
            "что нового в России",
            "последние новости",
            "новости России",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    search_query(text, news=True),
                    "главные новости России сегодня",
                )

    def test_topical_news_request_is_kept(self) -> None:
        self.assertEqual(
            search_query("новости про выборы", news=True),
            "новости про выборы",
        )

    def test_general_search_is_kept(self) -> None:
        self.assertEqual(
            search_query("какие новости", news=False),
            "какие новости",
        )


class CityExtractionTest(unittest.TestCase):
    def test_city_after_weather(self) -> None:
        self.assertEqual(extract_city("погода в Москве"), "Москве")
        self.assertEqual(extract_city("какая погода в Казани"), "Казани")
        self.assertEqual(
            extract_city("погода в Санкт-Петербурге"), "Санкт-Петербурге"
        )

    def test_city_before_weather(self) -> None:
        self.assertEqual(extract_city("в Москве погода"), "Москве")

    def test_time_is_not_city(self) -> None:
        self.assertIsNone(extract_city("погода на завтра"))
        self.assertIsNone(extract_city("какая погода"))

    def test_time_and_city(self) -> None:
        self.assertEqual(extract_city("погода на завтра в Москве"), "Москве")
        self.assertEqual(extract_city("погода в Москве сегодня"), "Москве")


class FirstSentenceTest(unittest.TestCase):
    def test_first_sentence(self) -> None:
        self.assertEqual(first_sentence("Первое. Второе."), "Первое.")

    def test_truncates_long_text(self) -> None:
        text = first_sentence("а" * 400)
        self.assertEqual(len(text), 250)
        self.assertTrue(text.endswith("…"))


class QAServiceRoutingTest(unittest.TestCase):
    def test_handles_weather_and_questions(self) -> None:
        qa = make_service()
        self.assertTrue(qa.handles("какая погода"))
        self.assertTrue(qa.handles("что такое фотосинтез"))
        self.assertFalse(qa.handles("открой ютуб"))

    def test_disabled_handles_nothing(self) -> None:
        qa = make_service(enabled=False)
        self.assertFalse(qa.handles("какая погода"))

    def test_weather_uses_default_city(self) -> None:
        weather = FakeWeather(make_report())
        qa = make_service()
        qa.weather = weather
        answer = qa.answer("какая погода")
        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "weather")
        self.assertIn("Новосибирск", answer.short)
        self.assertEqual(weather.cities, [None])

    def test_weather_uses_named_city(self) -> None:
        weather = FakeWeather(make_report())
        qa = make_service()
        qa.weather = weather
        qa.answer("погода в Москве")
        self.assertEqual(weather.cities, ["Москве"])

    def test_weather_error(self) -> None:
        qa = make_service()
        qa.weather = FakeWeather(error=WeatherError("offline"))
        answer = qa.answer("какая погода")
        self.assertFalse(answer.ok)
        self.assertIn("Не смог узнать погоду", answer.short)

    def test_fresh_question_without_key(self) -> None:
        qa = make_service()
        qa.search = FakeSearch(available=False)
        answer = qa.answer("какие новости")
        self.assertFalse(answer.ok)
        self.assertIn("Tavily", answer.short)

    def test_search_answer_with_citations(self) -> None:
        qa = make_service(llm=FakeLLM("Курс вырос [1]."))
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Текст новости")]
        )
        answer = qa.answer("какой курс доллара")
        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "search")
        self.assertEqual(answer.full, "Курс вырос [1].")
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(qa.search.calls, [("какой курс доллара", False)])

    def test_news_query_uses_news_mode(self) -> None:
        qa = make_service(llm=FakeLLM("Новость [1]."))
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Текст")]
        )
        qa.answer("какие новости")
        self.assertEqual(
            qa.search.calls,
            [("главные новости России сегодня", True)],
        )

    def test_topical_news_query_is_kept(self) -> None:
        qa = make_service(llm=FakeLLM("Новость [1]."))
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Текст")]
        )
        qa.answer("новости про выборы")
        self.assertEqual(qa.search.calls, [("новости про выборы", True)])

    def test_search_fallback_without_citations(self) -> None:
        qa = make_service(llm=FakeLLM("Курс вырос."))
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Текст новости")]
        )
        answer = qa.answer("какой курс доллара")
        self.assertTrue(answer.ok)
        self.assertIn("Модель не дала ответа", answer.full)
        self.assertIn("[1] Новость", answer.full)

    def test_search_fallback_for_citation_only_answer(self) -> None:
        qa = make_service(llm=FakeLLM("[1] [2]"))
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Текст новости")]
        )
        answer = qa.answer("какой курс доллара")
        self.assertTrue(answer.ok)
        self.assertIn("Модель не дала ответа", answer.full)
        self.assertIn("[1] Новость", answer.full)

    def test_search_error(self) -> None:
        qa = make_service(llm=FakeLLM())
        qa.search = FakeSearch(error=SearchError("boom"))
        answer = qa.answer("какие новости")
        self.assertFalse(answer.ok)
        self.assertIn("Поиск не удался", answer.short)

    def test_local_answer(self) -> None:
        qa = make_service(llm=FakeLLM("Фотосинтез — это процесс."))
        qa.search = FakeSearch(available=False)
        answer = qa.answer("расскажи про фотосинтез")
        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "local")
        self.assertEqual(qa.search.calls, [])

    def test_local_answer_passes_qa_timeout(self) -> None:
        llm = FakeLLM("Фотосинтез — это процесс.")
        qa = make_service(llm=llm)
        qa.search = FakeSearch(available=False)
        qa.timeout = 42
        qa.answer("расскажи про фотосинтез")
        self.assertEqual(llm.kwargs[0]["timeout"], 42)

    def test_local_llm_error(self) -> None:
        qa = make_service(llm=FakeLLM(error=LocalLLMError("down")))
        qa.search = FakeSearch(available=False)
        answer = qa.answer("расскажи про фотосинтез")
        self.assertFalse(answer.ok)

    def test_no_backend(self) -> None:
        qa = make_service(llm=FakeLLM(available=False))
        qa.search = FakeSearch(available=False)
        answer = qa.answer("расскажи про фотосинтез")
        self.assertFalse(answer.ok)
        self.assertIn("Не могу ответить", answer.short)


class QARenderTest(unittest.TestCase):
    def test_escapes_html_and_links_sources(self) -> None:
        qa = make_service()
        answer = QAAnswer(
            True,
            "<script>alert(1)</script>",
            "short",
            "Текст <b>ответа</b>",
            "search",
            [QASource("A & B", "https://example.com/?a=1&b=2")],
        )
        page = qa.render_html(answer)
        self.assertIn("&lt;script&gt;", page)
        self.assertNotIn("<script>", page)
        self.assertIn("A &amp; B", page)
        self.assertIn("a=1&amp;b=2", page)

    def test_show_opens_tab_only_for_success(self) -> None:
        browser = FakeBrowser()
        qa = make_service(browser=browser)
        self.assertTrue(qa.show(QAAnswer(True, "q", "s", "f", "local")))
        self.assertEqual(browser.calls[0][0], "render")
        self.assertFalse(qa.show(QAAnswer(False, "q", "s", "f", "error")))
        self.assertEqual(len(browser.calls), 1)


if __name__ == "__main__":
    unittest.main()
