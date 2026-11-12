from __future__ import annotations

import unittest

from app.local_llm import LocalLLMError
from app.qa import (
    QAAnswer,
    QAService,
    QASource,
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


class CityExtractionTest(unittest.TestCase):
    def test_city_after_weather(self) -> None:
        self.assertEqual(extract_city("погода в Москве"), "Москве")
        self.assertEqual(extract_city("какая погода в Казани"), "Казани")
        self.assertEqual(
            extract_city("погода в Санкт-Петербурге"), "Санкт-Петербурге"
        )

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
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Текст новости")]
        )
        answer = qa.answer("какой курс доллара")
        self.assertTrue(answer.ok)
        self.assertEqual(answer.kind, "search")
        self.assertEqual(answer.full, "Курс вырос [1].")
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(qa.search.calls, [("какой курс доллара", False)])

    def test_search_answer_removes_meta_phrases(self) -> None:
        qa = make_service(
            llm=FakeLLM("В источнике [1] сообщается о том, что курс вырос [1].")
        )
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Курс вырос")]
        )
        answer = qa.answer("какой курс доллара")
        self.assertEqual(answer.full, "курс вырос [1].")

    def test_search_answer_removes_news_meta_phrases(self) -> None:
        qa = make_service(
            llm=FakeLLM("[1] В новости о курсе говорится, что курс вырос [1].")
        )
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Курс вырос")]
        )
        answer = qa.answer("какой курс доллара")
        self.assertEqual(answer.full, "[1] Курс вырос [1].")

    def test_search_answer_removes_meta_with_site_name(self) -> None:
        qa = make_service(
            llm=FakeLLM(
                "[1] В новости о пресс-секретаре Белого дома, Lenta.ru "
                "сообщает, что она показала фигуру [1]."
            )
        )
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Текст новости")]
        )
        answer = qa.answer("какие новости")
        self.assertEqual(answer.full, "[1] Она показала фигуру [1].")

    def test_search_answer_removes_trailing_dont_know(self) -> None:
        qa = make_service(llm=FakeLLM("Курс вырос [1]. Не знаю."))
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Курс вырос")]
        )
        answer = qa.answer("какой курс доллара")
        self.assertEqual(answer.full, "Курс вырос [1].")

    def test_search_answer_removes_refusal_sentences(self) -> None:
        qa = make_service(
            llm=FakeLLM(
                "[1] Курс вырос [1]. Не знаю. "
                "В представленных источниках нет информации."
            )
        )
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Курс вырос")]
        )
        answer = qa.answer("какой курс доллара")
        self.assertEqual(answer.full, "[1] Курс вырос [1].")

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

    def test_news_default_uses_news_style(self) -> None:
        llm = FakeLLM("Новость [1].")
        qa = make_service(llm=llm)
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Текст")]
        )
        qa.answer("какие новости")
        self.assertEqual(llm.kwargs[0]["style"], "news")
        self.assertEqual(llm.kwargs[0]["max_tokens"], 600)

    def test_news_short_overrides_default(self) -> None:
        llm = FakeLLM("Новость [1].")
        qa = make_service(llm=llm)
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Текст")]
        )
        qa.answer("новости кратко")
        self.assertEqual(llm.kwargs[0]["style"], "news_short")
        self.assertEqual(llm.kwargs[0]["max_tokens"], 220)
        self.assertEqual(
            qa.search.calls,
            [("главные новости России сегодня", True)],
        )

    def test_news_detailed_overrides_default(self) -> None:
        llm = FakeLLM("Новость [1].")
        qa = make_service(llm=llm)
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Текст")]
        )
        qa.answer("новости подробно")
        self.assertEqual(llm.kwargs[0]["style"], "news_detailed")
        self.assertEqual(
            qa.search.calls,
            [("главные новости России сегодня", True)],
        )

    def test_search_style_is_stripped_from_question(self) -> None:
        llm = FakeLLM("Курс вырос [1].")
        qa = make_service(llm=llm)
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Текст")]
        )
        qa.answer("какой курс доллара подробно")
        self.assertEqual(llm.calls[0][0], "какой курс доллара")
        self.assertEqual(llm.kwargs[0]["style"], "detailed")

    def test_local_style_is_passed(self) -> None:
        llm = FakeLLM("Фотосинтез — это процесс.")
        qa = make_service(llm=llm)
        qa.search = FakeSearch(available=False)
        qa.answer("расскажи про фотосинтез кратко")
        self.assertEqual(llm.calls[0][0], "расскажи про фотосинтез")
        self.assertEqual(llm.kwargs[0]["style"], "short")
        self.assertEqual(llm.kwargs[0]["max_tokens"], 220)

    def test_search_fallback_without_citations(self) -> None:
        qa = make_service(llm=FakeLLM("Курс вырос."))
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Текст новости")]
        )
        answer = qa.answer("какой курс доллара")
        self.assertTrue(answer.ok)
        self.assertIn("Модель не смогла пересказать источники", answer.full)
        self.assertIn("[1] Новость", answer.full)

    def test_search_fallback_for_citation_only_answer(self) -> None:
        qa = make_service(llm=FakeLLM("[1] [2]"))
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Текст новости")]
        )
        answer = qa.answer("какой курс доллара")
        self.assertTrue(answer.ok)
        self.assertIn("Модель не смогла пересказать источники", answer.full)
        self.assertIn("[1] Новость", answer.full)

    def test_search_fallback_for_refusal_answer(self) -> None:
        qa = make_service(llm=FakeLLM("[1] Не содержит актуального курса доллара."))
        qa.search = FakeSearch(
            [SearchResult("Новость", "https://example.com", "Текст новости")]
        )
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

    def test_local_answer_removes_cjk(self) -> None:
        qa = make_service(llm=FakeLLM("Вода и二氧化碳."))
        qa.search = FakeSearch(available=False)
        answer = qa.answer("расскажи про воду")
        self.assertNotIn("二氧化碳", answer.full)
        self.assertIn("Вода и", answer.full)

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
        self.assertTrue(qa.show(QAAnswer(True, "q", "s", "f", "local")))
        self.assertEqual(browser.calls[0][0], "render")
        self.assertFalse(qa.show(QAAnswer(False, "q", "s", "f", "error")))
        self.assertEqual(len(browser.calls), 1)


if __name__ == "__main__":
    unittest.main()
