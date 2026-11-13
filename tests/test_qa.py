from __future__ import annotations

import unittest
from unittest.mock import patch

from app.cloud_llm import LLMError
from app.qa import (
    QAAnswer,
    QAService,
    QASource,
    _city_from_followup,
    _clean_answer,
    answer_style,
    extract_city,
    first_sentence,
    is_news_question,
    is_question,
    is_weather_question,
    needs_search,
    search_query,
    without_style,
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
        article: str = "",
        article_results: list[SearchResult] | None = None,
    ) -> None:
        self.results = results or []
        self.is_available = available
        self.error = error
        self.article = article
        self.article_results = article_results or []
        self.calls: list[tuple[str, bool]] = []
        self.extract_calls: list[str] = []
        self.article_calls: list[str] = []

    def available(self) -> bool:
        return self.is_available

    def search(self, query: str, *, news: bool = False) -> list[SearchResult]:
        self.calls.append((query, news))
        if self.error is not None:
            raise self.error
        return self.results

    def extract(self, url: str) -> str:
        self.extract_calls.append(url)
        if self.error is not None:
            raise self.error
        return self.article

    def search_article(self, query: str, *, limit: int = 3) -> list[SearchResult]:
        self.article_calls.append(query)
        if self.error is not None:
            raise self.error
        return self.article_results[:limit]


class FakeLLM:
    def __init__(
        self,
        text: str = "Ответ [1].",
        available: bool = True,
        error: Exception | None = None,
        texts: list[str] | None = None,
    ) -> None:
        self.text = text
        self.texts = list(texts or [])
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
        if self.texts:
            return self.texts.pop(0)
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


def make_three_day_report(place: str = "Москва") -> WeatherReport:
    return WeatherReport(
        place=GeoPlace(place, "Россия", 55.75, 37.61),
        temperature=2.0,
        feels_like=-1.0,
        code=3,
        wind=0.8,
        humidity=89,
        daily=(
            DailyForecast("2026-11-13", 0.0, 2.0, 77),
            DailyForecast("2026-11-14", 1.0, 4.0, 63),
            DailyForecast("2026-11-15", 2.0, 4.0, 61),
        ),
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
            "посоветуй фильм",
            "порекомендуй книгу",
            "можешь рассказать про космос",
            "не подскажешь, сколько стоит билет",
            "интересно, почему небо голубое",
            "мне интересно, почему небо голубое",
            "можно узнать, сколько стоит билет",
            "хотелось бы узнать про Python",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_question(text))

    def test_commands_are_not_questions(self) -> None:
        for text in (
            "открой ютуб",
            "включи музыку",
            "сделай что-нибудь странное",
            "есть хочу",
            "можешь открыть ютуб",
            "можно открыть ютуб",
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
        self.assertTrue(needs_search("новости"))
        self.assertTrue(needs_search("курс доллара"))
        self.assertTrue(needs_search("цены на бензин"))
        self.assertTrue(needs_search("сколько стоит билет"))
        self.assertTrue(needs_search("сколько будет стоить билет"))
        self.assertTrue(needs_search("что слышно"))
        self.assertTrue(needs_search("есть ли что-нибудь новое"))
        self.assertTrue(needs_search("поищи рецепт борща"))
        self.assertFalse(needs_search("что такое фотосинтез"))
        self.assertFalse(needs_search("расскажи про Python"))
        self.assertFalse(needs_search("посоветуй фильм"))


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

    def test_free_news_formulations_become_concrete(self) -> None:
        for text in (
            "новости за последний день",
            "новости за сегодня",
            "новости за вчера",
            "новости за последнюю неделю",
            "новости за последние 24 часа",
            "последние новости за день",
            "что нового за неделю",
            "что случилось за сегодня",
            "какие новости на сегодня",
            "расскажи новости за последние сутки",
            "сводка новостей за вчера",
            "что там по новостям",
            "что пишут в новостях",
            "что в мире творится",
            "последние события в мире",
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

    def test_topical_news_with_time_is_kept(self) -> None:
        self.assertEqual(
            search_query("новости про выборы за последний день", news=True),
            "новости про выборы за последний день",
        )
        self.assertEqual(
            search_query("новости за 2024 год", news=True),
            "новости за 2024 год",
        )

    def test_general_search_is_kept(self) -> None:
        self.assertEqual(
            search_query("какие новости", news=False),
            "какие новости",
        )

    def test_style_words_are_removed_from_query(self) -> None:
        self.assertEqual(
            search_query("новости кратко", news=True),
            "главные новости России сегодня",
        )
        self.assertEqual(
            search_query("какой курс доллара подробно", news=False),
            "какой курс доллара",
        )


class AnswerStyleTest(unittest.TestCase):
    def test_short_synonyms(self) -> None:
        for text in (
            "новости кратко",
            "новости коротко",
            "новости вкратце",
            "новости в двух словах",
            "новости одним предложением",
            "новости без подробностей",
        ):
            with self.subTest(text=text):
                self.assertEqual(answer_style(text), "short")

    def test_detailed_synonyms(self) -> None:
        for text in (
            "новости подробно",
            "новости подробнее",
            "новости поподробнее",
            "новости детально",
            "новости в деталях",
            "новости с подробностями",
            "новости развёрнуто",
        ):
            with self.subTest(text=text):
                self.assertEqual(answer_style(text), "detailed")

    def test_normal_by_default(self) -> None:
        self.assertEqual(answer_style("какие новости"), "normal")
        self.assertEqual(answer_style("какой курс доллара"), "normal")

    def test_last_marker_wins(self) -> None:
        self.assertEqual(answer_style("кратко, нет, подробнее"), "detailed")
        self.assertEqual(answer_style("подробно, нет, кратко"), "short")

    def test_without_style(self) -> None:
        self.assertEqual(without_style("новости кратко"), "новости")
        self.assertEqual(
            without_style("расскажи подробнее о курсе доллара"),
            "расскажи о курсе доллара",
        )
        self.assertEqual(
            without_style("новости, в двух словах"),
            "новости",
        )


class CleanAnswerTest(unittest.TestCase):
    def test_removes_meta_intro_with_colon(self) -> None:
        text = (
            "На основе предоставленных источников, можно отметить "
            "следующие новости:\n\n[1] Курс вырос [1]."
        )
        self.assertEqual(_clean_answer(text), "[1] Курс вырос [1].")

    def test_removes_run_on_meta_intro(self) -> None:
        text = (
            "На основе данных из источников [1] и [2], в эти дни были "
            "опубликованы новости о том, что курс вырос [1]."
        )
        self.assertEqual(_clean_answer(text), "курс вырос [1].")

    def test_removes_meta_intro_without_colon(self) -> None:
        text = "На основе предоставленных источников, курс вырос [1]."
        self.assertEqual(_clean_answer(text), "курс вырос [1].")

    def test_merges_standalone_citation_line(self) -> None:
        text = "[1]\nКурс вырос на 2%.\n\n[2]\nЦена снизилась."
        self.assertEqual(
            _clean_answer(text),
            "[1] Курс вырос на 2%.\n[2] Цена снизилась.",
        )


class CityExtractionTest(unittest.TestCase):
    def test_city_after_weather(self) -> None:
        self.assertEqual(extract_city("погода в Москве"), "Москве")
        self.assertEqual(extract_city("какая погода в Казани"), "Казани")
        self.assertEqual(extract_city("погода в Санкт-Петербурге"), "Санкт-Петербурге")

    def test_city_before_weather(self) -> None:
        self.assertEqual(extract_city("в Москве погода"), "Москве")

    def test_city_without_preposition(self) -> None:
        self.assertEqual(extract_city("погода Бийск"), "Бийск")
        self.assertEqual(extract_city("погода Бийск завтра"), "Бийск")
        self.assertEqual(extract_city("Бийск погода"), "Бийск")

    def test_question_words_are_not_city(self) -> None:
        self.assertIsNone(extract_city("мне интересно какая погода"))
        self.assertIsNone(extract_city("погода будет"))
        self.assertIsNone(extract_city("прогноз погоды"))

    def test_time_is_not_city(self) -> None:
        self.assertIsNone(extract_city("погода на завтра"))
        self.assertIsNone(extract_city("какая погода"))

    def test_time_and_city(self) -> None:
        self.assertEqual(extract_city("погода на завтра в Москве"), "Москве")
        self.assertEqual(extract_city("погода в Москве сегодня"), "Москве")

    def test_city_followup_strips_day_word(self) -> None:
        self.assertEqual(_city_from_followup("а в Москве завтра?"), "Москве")
        self.assertEqual(_city_from_followup("а на завтра в Казани?"), "Казани")
        self.assertIsNone(_city_from_followup("а завтра?"))


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
        self.assertTrue(qa.handles("новости"))
        self.assertTrue(qa.handles("курс доллара"))
        self.assertTrue(qa.handles("посоветуй фильм"))
        self.assertFalse(qa.handles("открой ютуб"))
        self.assertFalse(qa.handles("можешь открыть ютуб"))

    def test_disabled_handles_nothing(self) -> None:
        qa = make_service(enabled=False)
        self.assertFalse(qa.handles("какая погода"))

    def test_route_ack_for_phrase_choice(self) -> None:
        qa = make_service()
        self.assertEqual(qa.route("какая погода завтра").ack, "weather")
        self.assertEqual(qa.route("новости").ack, "news")
        self.assertEqual(qa.route("расскажи про 1 новость подробнее").ack, "news")
        self.assertEqual(qa.route("что такое фотосинтез").ack, "wait")

    def test_datetime_kind_is_not_weather_followup(self) -> None:
        qa = make_service()
        qa.weather = FakeWeather(make_report())
        qa.answer("какая погода завтра")
        self.assertEqual(qa.route("какое завтра число").ack, "wait")

    def test_handles_datetime_questions(self) -> None:
        qa = make_service()
        self.assertTrue(qa.handles("который час"))
        self.assertTrue(qa.handles("сколько сейчас времени"))
        self.assertTrue(qa.handles("какое сегодня число"))
        self.assertTrue(qa.handles("какой день недели"))
        self.assertTrue(qa.handles("какой сейчас год"))

    def test_route_classifies_phrases(self) -> None:
        qa = make_service()
        cases = {
            "который час": "datetime",
            "какое завтра число": "datetime",
            "какая погода": "weather",
            "новости": "news",
            "курс доллара": "search",
            "что такое фотосинтез": "general",
            "открой ютуб": "none",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(qa.route(text).kind, expected)

    def test_disabled_service_routes_nothing(self) -> None:
        qa = make_service(enabled=False)
        self.assertEqual(qa.route("какая погода").kind, "none")

    def test_ack_categories(self) -> None:
        qa = make_service()
        self.assertEqual(qa.route("какая погода").ack, "weather")
        self.assertEqual(qa.route("новости").ack, "news")
        self.assertEqual(qa.route("что такое фотосинтез").ack, "wait")

    def test_answer_does_not_reparse_when_route_passed(self) -> None:
        qa = make_service()
        route = qa.route("какое завтра число")
        with patch(
            "app.qa.parse_datetime_question",
            side_effect=AssertionError("фраза разобрана повторно"),
        ):
            answer = qa.answer("какое завтра число", route=route)
        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "time")

    def test_datetime_answer_is_local(self) -> None:
        llm = FakeLLM()
        qa = make_service(llm=llm)
        qa.search = FakeSearch(available=False)
        answer = qa.answer("который час")
        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "time")
        self.assertTrue(answer.short.startswith("Сейчас "))
        self.assertEqual(answer.full, answer.short)
        self.assertEqual(llm.calls, [])
        self.assertEqual(qa.search.calls, [])
        self.assertIsNone(qa._last_turn)

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

    def test_weather_uses_city_without_preposition(self) -> None:
        weather = FakeWeather(make_report())
        qa = make_service()
        qa.weather = weather
        qa.answer("погода Бийск")
        self.assertEqual(weather.cities, ["Бийск"])

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
        qa.search = FakeSearch([SearchResult("Новость", "https://example.com", "Текст новости")])
        answer = qa.answer("какой курс доллара")
        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "search")
        self.assertEqual(answer.full, "Курс вырос [1].")
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(qa.search.calls, [("какой курс доллара", False)])

    def test_search_answer_removes_meta_phrases(self) -> None:
        qa = make_service(llm=FakeLLM("В источнике [1] сообщается о том, что курс вырос [1]."))
        qa.search = FakeSearch([SearchResult("Новость", "https://example.com", "Курс вырос")])
        answer = qa.answer("какой курс доллара")
        self.assertEqual(answer.full, "курс вырос [1].")

    def test_search_answer_removes_news_meta_phrases(self) -> None:
        qa = make_service(llm=FakeLLM("[1] В новости о курсе говорится, что курс вырос [1]."))
        qa.search = FakeSearch([SearchResult("Новость", "https://example.com", "Курс вырос")])
        answer = qa.answer("какой курс доллара")
        self.assertEqual(answer.full, "[1] Курс вырос [1].")

    def test_search_answer_removes_meta_with_site_name(self) -> None:
        qa = make_service(
            llm=FakeLLM(
                "[1] В новости о пресс-секретаре Белого дома, Lenta.ru "
                "сообщает, что она показала фигуру [1]."
            )
        )
        qa.search = FakeSearch([SearchResult("Новость", "https://example.com", "Текст новости")])
        answer = qa.answer("какие новости")
        self.assertEqual(answer.full, "[1] Она показала фигуру [1].")

    def test_search_answer_removes_trailing_dont_know(self) -> None:
        qa = make_service(llm=FakeLLM("Курс вырос [1]. Не знаю."))
        qa.search = FakeSearch([SearchResult("Новость", "https://example.com", "Курс вырос")])
        answer = qa.answer("какой курс доллара")
        self.assertEqual(answer.full, "Курс вырос [1].")

    def test_search_answer_removes_refusal_sentences(self) -> None:
        qa = make_service(
            llm=FakeLLM("[1] Курс вырос [1]. Не знаю. В представленных источниках нет информации.")
        )
        qa.search = FakeSearch([SearchResult("Новость", "https://example.com", "Курс вырос")])
        answer = qa.answer("какой курс доллара")
        self.assertEqual(answer.full, "[1] Курс вырос [1].")

    def test_news_query_uses_news_mode(self) -> None:
        qa = make_service(llm=FakeLLM("Новость [1]."))
        qa.search = FakeSearch([SearchResult("Новость", "https://example.com", "Текст")])
        qa.answer("какие новости")
        self.assertEqual(
            qa.search.calls,
            [("главные новости России сегодня", True)],
        )

    def test_topical_news_query_is_kept(self) -> None:
        qa = make_service(llm=FakeLLM("Новость [1]."))
        qa.search = FakeSearch([SearchResult("Новость", "https://example.com", "Текст")])
        qa.answer("новости про выборы")
        self.assertEqual(qa.search.calls, [("новости про выборы", True)])

    def test_news_default_uses_news_style(self) -> None:
        llm = FakeLLM("Новость [1].")
        qa = make_service(llm=llm)
        qa.search = FakeSearch([SearchResult("Новость", "https://example.com", "Текст")])
        qa.answer("какие новости")
        self.assertEqual(llm.kwargs[0]["style"], "news")
        self.assertEqual(llm.kwargs[0]["max_tokens"], 800)

    def test_news_short_overrides_default(self) -> None:
        llm = FakeLLM("Новость [1].")
        qa = make_service(llm=llm)
        qa.search = FakeSearch([SearchResult("Новость", "https://example.com", "Текст")])
        qa.answer("новости кратко")
        self.assertEqual(llm.kwargs[0]["style"], "news_short")
        self.assertEqual(llm.kwargs[0]["max_tokens"], 800)
        self.assertEqual(
            qa.search.calls,
            [("главные новости России сегодня", True)],
        )

    def test_news_detailed_overrides_default(self) -> None:
        llm = FakeLLM("Новость [1].")
        qa = make_service(llm=llm)
        qa.search = FakeSearch([SearchResult("Новость", "https://example.com", "Текст")])
        qa.answer("новости подробно")
        self.assertEqual(llm.kwargs[0]["style"], "news_detailed")
        self.assertEqual(
            qa.search.calls,
            [("главные новости России сегодня", True)],
        )

    def test_news_digest_stores_items_for_followup(self) -> None:
        llm = FakeLLM(
            "[1] Экспорт легковых автомобилей вырос на 62%.\n"
            "[2] За ночь над Россией сбили 71 беспилотник."
        )
        qa = make_service(llm=llm)
        qa.search = FakeSearch(
            [
                SearchResult(
                    "Экспорт",
                    "https://example.com/export",
                    "Текст про экспорт",
                ),
                SearchResult(
                    "Беспилотники",
                    "https://example.com/drones",
                    "Текст про дроны",
                ),
            ]
        )
        answer = qa.answer("новости кратко")
        self.assertTrue(answer.ok)
        self.assertEqual([item.index for item in qa.last_news], [1, 2])
        self.assertEqual(qa.last_news[0].url, "https://example.com/export")
        self.assertIn("[2] За ночь", answer.full)

    def test_news_digest_skips_stale_items(self) -> None:
        llm = FakeLLM(
            "[1] 15 января 2020 года правительство ушло в отставку.\n"
            "[2] За ночь над Россией сбили 71 беспилотник."
        )
        qa = make_service(llm=llm)
        qa.search = FakeSearch(
            [
                SearchResult("Старое", "https://example.com/old", "Текст"),
                SearchResult("Дроны", "https://example.com/drones", "Текст"),
            ]
        )
        answer = qa.answer("новости кратко")
        self.assertEqual(
            [item.fact for item in qa.last_news],
            ["За ночь над Россией сбили 71 беспилотник."],
        )
        self.assertEqual(qa.last_news[0].index, 1)
        self.assertNotIn("2020", answer.full)
        self.assertIn("[1] За ночь", answer.full)

    def test_news_digest_skips_title_only_items(self) -> None:
        llm = FakeLLM(
            "[1] РИА Новости - события в Москве, России и мире сегодня.\n"
            "[2] За ночь над Россией сбили 71 беспилотник."
        )
        qa = make_service(llm=llm)
        qa.search = FakeSearch(
            [
                SearchResult(
                    "РИА Новости - события в Москве, России и мире сегодня",
                    "https://ria.ru",
                    "Текст",
                ),
                SearchResult("Дроны", "https://example.com/drones", "Текст"),
            ]
        )
        answer = qa.answer("новости кратко")
        self.assertEqual(
            [item.fact for item in qa.last_news],
            ["За ночь над Россией сбили 71 беспилотник."],
        )
        self.assertNotIn("РИА Новости", answer.full)

    def test_news_context_replaces_generic_title(self) -> None:
        llm = FakeLLM("[1] Экспорт вырос на 62%. Подробности поставок.")
        qa = make_service(llm=llm)
        qa.search = FakeSearch(
            [
                SearchResult(
                    "Выпуск за 10 ноября 2026",
                    "https://example.com/issue",
                    "Экспорт легковых автомобилей вырос на 62% в октябре. Ещё одна новость дня.",
                )
            ]
        )
        qa.answer("новости кратко")
        context = llm.calls[-1][1]
        self.assertNotIn("Выпуск за 10 ноября 2026", context)
        self.assertIn("Экспорт легковых автомобилей", context)
        self.assertEqual(qa.last_news[0].title, "Экспорт вырос на 62%.")

    def test_news_followup_by_number_uses_article(self) -> None:
        llm = FakeLLM(
            "[1] Экспорт легковых автомобилей вырос на 62%.\n"
            "[2] За ночь над Россией сбили 71 беспилотник."
        )
        qa = make_service(llm=llm)
        qa.search = FakeSearch(
            [
                SearchResult(
                    "Экспорт",
                    "https://example.com/export",
                    "Текст про экспорт",
                ),
                SearchResult(
                    "Беспилотники",
                    "https://example.com/drones",
                    "Текст про дроны",
                ),
            ],
            article_results=[
                SearchResult(
                    "Разбор экспорта",
                    "https://example.com/export-analysis",
                    "Поставки легковых автомобилей за рубеж выросли на 62% "
                    "год к году до 932 тыс. единиц.",
                )
            ],
        )
        qa.answer("новости кратко")
        llm.text = "Подробности: экспорт вырос на 62% [1]."
        answer = qa.answer("расскажи про 1 новость подробнее")
        self.assertTrue(answer.ok)
        self.assertEqual(qa.search.extract_calls, [])
        self.assertEqual(
            qa.search.article_calls,
            ["Экспорт легковых автомобилей вырос на 62%."],
        )
        self.assertEqual(llm.kwargs[-1]["style"], "article")
        self.assertIn("Поставки легковых автомобилей", llm.calls[-1][1])
        self.assertIn("Подробности", answer.full)

    def test_news_followup_by_topic_selects_item(self) -> None:
        llm = FakeLLM(
            "[1] Экспорт легковых автомобилей вырос на 62%.\n"
            "[2] За ночь над Россией сбили 71 беспилотник."
        )
        qa = make_service(llm=llm)
        qa.search = FakeSearch(
            [
                SearchResult(
                    "Экспорт",
                    "https://example.com/export",
                    "Текст про экспорт",
                ),
                SearchResult(
                    "Беспилотники",
                    "https://example.com/drones",
                    "Текст про дроны",
                ),
            ],
            article="Полный текст: экспорт легковых автомобилей вырос на 62% в октябре.",
            article_results=[],
        )
        qa.answer("новости кратко")
        llm.text = "Экспорт вырос на 62% [1]."
        qa.answer("что там с экспортом автомобилей?")
        self.assertEqual(qa.search.extract_calls, ["https://example.com/export"])
        self.assertIn("Полный текст", llm.calls[-1][1])

    def test_news_followup_topic_prefers_related_search(self) -> None:
        llm = FakeLLM(
            "[1] Экспорт легковых автомобилей вырос на 62%.\n"
            "[2] За ночь над Россией сбили 71 беспилотник."
        )
        qa = make_service(llm=llm)
        qa.search = FakeSearch(
            [
                SearchResult(
                    "Экспорт",
                    "https://example.com/export",
                    "Текст про экспорт",
                ),
                SearchResult(
                    "Беспилотники",
                    "https://example.com/drones",
                    "Текст про дроны",
                ),
            ],
            article="Полный текст про экспорт автомобилей.",
            article_results=[
                SearchResult(
                    "Разбор экспорта",
                    "https://example.com/export-analysis",
                    "Экспорт автомобилей из Китая вырос на 62% в октябре.",
                )
            ],
        )
        qa.answer("новости кратко")
        llm.text = "Экспорт вырос на 62% [1]."
        qa.answer("что там с экспортом автомобилей?")
        self.assertEqual(qa.search.extract_calls, [])
        self.assertIn("Экспорт автомобилей из Китая", llm.calls[-1][1])

    def test_news_followup_ambiguous_asks_clarification(self) -> None:
        llm = FakeLLM("[1] Экспорт автомобилей вырос.\n[2] Экспорт нефти вырос.")
        qa = make_service(llm=llm)
        qa.search = FakeSearch(
            [
                SearchResult("Авто", "https://example.com/cars", "Текст про авто"),
                SearchResult("Нефть", "https://example.com/oil", "Текст про нефть"),
            ]
        )
        qa.answer("новости кратко")
        answer = qa.answer("что там с экспортом?")
        self.assertIn("уточните номер", answer.short.lower())
        self.assertEqual(qa.search.extract_calls, [])

    def test_news_followup_with_unknown_number_asks_clarification(self) -> None:
        llm = FakeLLM("[1] Экспорт вырос.\n[2] Снег выпал.")
        qa = make_service(llm=llm)
        qa.search = FakeSearch(
            [
                SearchResult("Экспорт", "https://example.com/export", "Текст"),
                SearchResult("Снег", "https://example.com/snow", "Текст"),
            ]
        )
        qa.answer("новости кратко")
        answer = qa.answer("расскажи про 5 новость подробнее")
        self.assertIn("Не понял", answer.short)
        self.assertEqual(qa.search.extract_calls, [])

    def test_news_followup_falls_back_to_snippet(self) -> None:
        llm = FakeLLM("[1] Экспорт вырос.\n[2] Снег выпал.")
        qa = make_service(llm=llm)
        qa.search = FakeSearch(
            [
                SearchResult(
                    "Экспорт",
                    "https://example.com/export",
                    "Подробный текст про экспорт автомобилей из Китая.",
                ),
                SearchResult("Снег", "https://example.com/snow", "Текст про снег"),
            ]
        )
        qa.answer("новости кратко")
        llm.text = "Не знаю."
        answer = qa.answer("расскажи про 1 новость подробнее")
        self.assertTrue(answer.ok)
        self.assertIn("Подробный текст", answer.full)

    def test_news_followup_does_not_hijack_local_question(self) -> None:
        llm = FakeLLM("[1] Экспорт вырос.\n[2] Снег выпал.")
        qa = make_service(llm=llm)
        qa.search = FakeSearch(
            [
                SearchResult("Экспорт", "https://example.com/export", "Текст"),
                SearchResult("Снег", "https://example.com/snow", "Текст"),
            ]
        )
        qa.answer("новости кратко")
        llm.text = "Фотосинтез — это процесс."
        qa.answer("расскажи про фотосинтез подробно")
        self.assertEqual(qa.search.extract_calls, [])
        self.assertEqual(llm.kwargs[-1]["style"], "detailed")

    def test_search_style_is_stripped_from_question(self) -> None:
        llm = FakeLLM("Курс вырос [1].")
        qa = make_service(llm=llm)
        qa.search = FakeSearch([SearchResult("Новость", "https://example.com", "Текст")])
        qa.answer("какой курс доллара подробно")
        self.assertEqual(llm.calls[0][0], "какой курс доллара")
        self.assertEqual(llm.kwargs[0]["style"], "detailed")

    def test_general_style_is_passed(self) -> None:
        llm = FakeLLM("Фотосинтез — это процесс.")
        qa = make_service(llm=llm)
        qa.search = FakeSearch(available=False)
        qa.answer("расскажи про фотосинтез кратко")
        self.assertEqual(llm.calls[0][0], "расскажи про фотосинтез")
        self.assertEqual(llm.kwargs[0]["style"], "short")
        self.assertEqual(llm.kwargs[0]["max_tokens"], 220)

    def test_search_fallback_without_citations(self) -> None:
        qa = make_service(llm=FakeLLM("Курс вырос."))
        qa.search = FakeSearch([SearchResult("Новость", "https://example.com", "Текст новости")])
        answer = qa.answer("какой курс доллара")
        self.assertTrue(answer.ok)
        self.assertIn("Модель не смогла пересказать источники", answer.full)
        self.assertIn("[1] Новость", answer.full)

    def test_search_fallback_for_citation_only_answer(self) -> None:
        qa = make_service(llm=FakeLLM("[1] [2]"))
        qa.search = FakeSearch([SearchResult("Новость", "https://example.com", "Текст новости")])
        answer = qa.answer("какой курс доллара")
        self.assertTrue(answer.ok)
        self.assertIn("Модель не смогла пересказать источники", answer.full)
        self.assertIn("[1] Новость", answer.full)

    def test_search_fallback_for_refusal_answer(self) -> None:
        qa = make_service(llm=FakeLLM("[1] Не содержит актуального курса доллара."))
        qa.search = FakeSearch([SearchResult("Новость", "https://example.com", "Текст новости")])
        answer = qa.answer("какой курс доллара")
        self.assertTrue(answer.ok)
        self.assertIn("Модель не смогла пересказать источники", answer.full)
        self.assertIn("[1] Новость", answer.full)

    def test_search_fallback_cleans_markdown(self) -> None:
        qa = make_service(llm=FakeLLM(available=False))
        qa.search = FakeSearch(
            [
                SearchResult(
                    "Новость",
                    "https://example.com",
                    "## Лента\n\n### За ночь над Россией сбили 155 беспилотников",
                )
            ]
        )
        answer = qa.answer("какие новости")
        self.assertTrue(answer.ok)
        self.assertNotIn("##", answer.full)
        self.assertNotIn("###", answer.full)
        self.assertIn("За ночь над Россией", answer.short)
        self.assertIn("Вот главное из найденного", answer.full)

    def test_search_fallback_drops_tag_soup(self) -> None:
        qa = make_service(llm=FakeLLM(available=False))
        qa.search = FakeSearch(
            [
                SearchResult(
                    "Администрация США",
                    "https://example.com",
                    "Владимир ПутинДмитрий ПесковАдминистрация "
                    "СШАВАШИНГТОНКИТАЙ Глава Пентагона жестко ответил "
                    "на планы Конгресса по его импичменту",
                )
            ]
        )
        answer = qa.answer("какие новости")
        self.assertTrue(answer.ok)
        self.assertIn("Глава Пентагона", answer.short)
        self.assertNotIn("ПутинДмитрий", answer.full)
        self.assertNotIn("ВАШИНГТОНКИТАЙ", answer.full)

    def test_search_error(self) -> None:
        qa = make_service(llm=FakeLLM())
        qa.search = FakeSearch(error=SearchError("boom"))
        answer = qa.answer("какие новости")
        self.assertFalse(answer.ok)
        self.assertEqual(answer.kind, "error")
        self.assertEqual(answer.short, "Сервис не отвечает.")
        self.assertIn("Поиск не удался: boom", answer.full)

    def test_empty_search_is_not_an_error(self) -> None:
        qa = make_service(llm=FakeLLM())
        qa.search = FakeSearch(results=[])
        answer = qa.answer("какие новости")
        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "empty")
        self.assertEqual(answer.short, "Ничего не нашлось.")
        self.assertIn("ничего не нашлось", answer.full)
        self.assertEqual(answer.sources, [])
        self.assertIsNone(qa._last_turn)

    def test_general_answer(self) -> None:
        qa = make_service(llm=FakeLLM("Фотосинтез — это процесс."))
        qa.search = FakeSearch(available=False)
        answer = qa.answer("расскажи про фотосинтез")
        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "general")
        self.assertEqual(qa.search.calls, [])

    def test_general_answer_passes_qa_timeout(self) -> None:
        llm = FakeLLM("Фотосинтез — это процесс.")
        qa = make_service(llm=llm)
        qa.search = FakeSearch(available=False)
        qa.timeout = 42
        qa.answer("расскажи про фотосинтез")
        self.assertEqual(llm.kwargs[0]["timeout"], 42)

    def test_general_answer_removes_cjk(self) -> None:
        qa = make_service(llm=FakeLLM("Вода и二氧化碳."))
        qa.search = FakeSearch(available=False)
        answer = qa.answer("расскажи про воду")
        self.assertNotIn("二氧化碳", answer.full)
        self.assertIn("Вода и", answer.full)

    def test_llm_error(self) -> None:
        qa = make_service(llm=FakeLLM(error=LLMError("down")))
        qa.search = FakeSearch(available=False)
        answer = qa.answer("расскажи про фотосинтез")
        self.assertFalse(answer.ok)

    def test_no_model_is_reported_separately(self) -> None:
        qa = make_service(llm=FakeLLM(available=False))
        qa.search = FakeSearch(available=False)
        answer = qa.answer("расскажи про фотосинтез")
        self.assertFalse(answer.ok)
        self.assertEqual(answer.kind, "no_model")
        self.assertEqual(answer.short, "Модель недоступна.")


class QAServiceFollowUpTest(unittest.TestCase):
    def test_general_followup_passes_previous_turn(self) -> None:
        llm = FakeLLM("Фотосинтез важен, потому что даёт кислород.")
        qa = make_service(llm=llm)

        qa.answer("что такое фотосинтез")
        answer = qa.answer("а почему это важно?")

        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "general")
        self.assertEqual(len(llm.calls), 2)
        history = llm.kwargs[1]["history"]
        self.assertIn("что такое фотосинтез", history.lower())
        self.assertIn("Фотосинтез важен", history)

    def test_chained_followup_keeps_original_topic(self) -> None:
        llm = FakeLLM("Фотосинтез важен, потому что даёт кислород.")
        qa = make_service(llm=llm)

        qa.answer("что такое фотосинтез")
        qa.answer("а почему это важно?")
        qa.answer("а подробнее?")

        history = llm.kwargs[2]["history"]
        self.assertIn("Тема разговора: что такое фотосинтез", history)
        self.assertIn("а почему это важно?", history.lower())

    def test_followup_is_detected_by_handles_and_kind(self) -> None:
        qa = make_service(llm=FakeLLM("Ответ."))
        qa.answer("что такое фотосинтез")

        self.assertTrue(qa.handles("а почему это важно?"))
        self.assertTrue(qa.handles("а подробнее?"))
        self.assertEqual(qa.route("а почему это важно?").ack, "wait")

    def test_followup_expires(self) -> None:
        qa = make_service(llm=FakeLLM("Ответ."))
        qa.answer("что такое фотосинтез")
        turn = qa._last_turn
        self.assertIsNotNone(turn)
        turn.at -= qa.context_ttl + 1

        self.assertIsNone(qa._followup("а почему это важно?"))
        self.assertIsNone(qa._last_turn)

    def test_weather_followup_reuses_city_and_day(self) -> None:
        weather = FakeWeather(make_three_day_report())
        qa = make_service()
        qa.weather = weather

        qa.answer("какая погода в Москве")
        answer = qa.answer("а завтра?")

        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "weather")
        self.assertEqual(qa.route("а завтра?").ack, "weather")
        self.assertEqual(weather.cities, ["Москве", "Москва"])
        self.assertIn("Москва, завтра", answer.full)
        self.assertIn("от +1 °C до +4 °C", answer.full)
        self.assertNotIn("Послезавтра", answer.full)

    def test_weather_followup_uses_named_city(self) -> None:
        weather = FakeWeather(make_three_day_report())
        qa = make_service()
        qa.weather = weather

        qa.answer("какая погода в Москве")
        answer = qa.answer("а в Казани?")

        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "weather")
        self.assertEqual(weather.cities, ["Москве", "Казани"])


class QASearchFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.result = SearchResult(
            "Fortuna 812 — Википедия",
            "https://ru.wikipedia.org/wiki/Fortuna_812",
            "FORTUNA 812 (Александр Салахутдинов) — российский рэпер "
            "и продюсер из Санкт-Петербурга.",
        )

    def test_confident_answer_does_not_search(self) -> None:
        qa = make_service(llm=FakeLLM("Фотосинтез — это процесс получения энергии из света."))
        qa.search = FakeSearch([self.result])

        answer = qa.answer("что такое фотосинтез")

        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "general")
        self.assertEqual(qa.search.calls, [])

    def test_uncertain_answer_searches_once(self) -> None:
        llm = FakeLLM(
            texts=[
                "Не знаю, стоит поискать в интернете.",
                "[1] FORTUNA 812 — российский рэпер из Санкт-Петербурга.",
            ]
        )
        qa = make_service(llm=llm)
        qa.search = FakeSearch([self.result])

        answer = qa.answer("расскажи про фортуну 812")

        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "search")
        self.assertIn("рэпер", answer.full)
        self.assertEqual(qa.search.calls, [("расскажи про фортуну 812", False)])
        self.assertEqual(len(llm.calls), 2)

    def test_empty_search_keeps_honest_answer(self) -> None:
        qa = make_service(llm=FakeLLM("Не знаю, стоит поискать в интернете."))
        qa.search = FakeSearch([])

        answer = qa.answer("расскажи про фортуну 812")

        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "general")
        self.assertEqual(answer.full, "Не знаю, стоит поискать в интернете.")

    def test_unavailable_search_keeps_honest_answer(self) -> None:
        qa = make_service(llm=FakeLLM("Не знаю, стоит поискать в интернете."))
        qa.search = FakeSearch([self.result], available=False)

        answer = qa.answer("расскажи про фортуну 812")

        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "general")
        self.assertEqual(qa.search.calls, [])

    def test_disabled_fallback_does_not_search(self) -> None:
        llm = FakeLLM("Не знаю, стоит поискать в интернете.")
        qa = make_service(llm=llm)
        qa.search_fallback = False
        qa.search = FakeSearch([self.result])

        answer = qa.answer("расскажи про фортуну 812")

        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "general")
        self.assertEqual(qa.search.calls, [])
        self.assertEqual(len(llm.calls), 1)

    def test_followup_search_uses_previous_question(self) -> None:
        llm = FakeLLM(
            texts=[
                "Фотосинтез — это процесс, при котором растения получают энергию.",
                "Не уверена, стоит поискать в интернете.",
                "[1] Фотосинтез важен для кислорода.",
            ]
        )
        qa = make_service(llm=llm)
        qa.search = FakeSearch([self.result])

        qa.answer("что такое фотосинтез")
        answer = qa.answer("а почему это важно?")

        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "search")
        self.assertEqual(
            qa.search.calls,
            [("что такое фотосинтез а почему это важно?", False)],
        )


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

    def test_splits_facts_into_paragraphs(self) -> None:
        qa = make_service()
        answer = QAAnswer(
            True,
            "новости",
            "short",
            "[1] Первый факт.\n[2] Второй факт.",
            "search",
        )
        page = qa.render_html(answer)
        self.assertIn("<p>[1] Первый факт.</p>", page)
        self.assertIn("<p>[2] Второй факт.</p>", page)

    def test_show_opens_tab_only_for_success(self) -> None:
        browser = FakeBrowser()
        qa = make_service(browser=browser)
        self.assertTrue(qa.show(QAAnswer(True, "q", "s", "f", "general")))
        self.assertEqual(browser.calls[0][0], "render")
        self.assertFalse(qa.show(QAAnswer(False, "q", "s", "f", "error")))
        self.assertEqual(len(browser.calls), 1)

    def test_show_skips_datetime_answers(self) -> None:
        browser = FakeBrowser()
        qa = make_service(browser=browser)
        answer = QAAnswer(True, "который час", "Сейчас 20 часов.", "Сейчас 20 часов.", "time")
        self.assertFalse(qa.show(answer))
        self.assertEqual(browser.calls, [])

    def test_show_skips_empty_answers(self) -> None:
        browser = FakeBrowser()
        qa = make_service(browser=browser)
        answer = QAAnswer(True, "курс доллара", "Ничего не нашлось.", "Нет данных.", "empty")
        self.assertFalse(qa.show(answer))
        self.assertEqual(browser.calls, [])


if __name__ == "__main__":
    unittest.main()
