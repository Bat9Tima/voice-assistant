from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from app.search import SearchError, TavilySearch


class FakeResponse:
    def __init__(self, body: bytes = b"", status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args) -> bool:
        return False

    def read(self) -> bytes:
        return self.body


def make_search(**overrides) -> TavilySearch:
    section = {"api_key": "tvly-test", "timeout_sec": 1}
    section.update(overrides)
    return TavilySearch({"qa": {"search": section}})


class TavilyConfigTest(unittest.TestCase):
    def test_missing_key_is_unavailable(self) -> None:
        client = TavilySearch({"qa": {"search": {"api_key": ""}}})
        self.assertFalse(client.available())
        with self.assertRaises(SearchError):
            client.search("новости")

    def test_disabled_is_unavailable(self) -> None:
        client = make_search(enabled=False)
        self.assertFalse(client.available())


class TavilyRequestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload: dict = {}
        self.client = make_search(
            news_domains=["rbc.ru", "lenta.ru"],
            news_days=3,
        )

    def fake_urlopen(self, request, timeout):
        self.payload = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            json.dumps(
                {
                    "results": [
                        {
                            "title": "Новость",
                            "url": "https://example.com/news",
                            "content": "Текст новости",
                            "score": 0.9,
                        }
                    ]
                }
            ).encode("utf-8")
        )

    def test_parses_results(self) -> None:
        with patch("app.search.urllib.request.urlopen", side_effect=self.fake_urlopen):
            results = self.client.search("что нового")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Новость")
        self.assertEqual(results[0].url, "https://example.com/news")
        self.assertEqual(results[0].content, "Текст новости")
        self.assertAlmostEqual(results[0].score, 0.9)

    def test_general_payload(self) -> None:
        with patch("app.search.urllib.request.urlopen", side_effect=self.fake_urlopen):
            self.client.search("что такое фотосинтез")
        self.assertEqual(self.payload["topic"], "general")
        self.assertNotIn("days", self.payload)
        self.assertNotIn("include_domains", self.payload)

    def test_news_payload_has_domains_and_days(self) -> None:
        with patch("app.search.urllib.request.urlopen", side_effect=self.fake_urlopen):
            self.client.search("новости", news=True)
        self.assertEqual(self.payload["topic"], "news")
        self.assertEqual(self.payload["days"], 3)
        self.assertEqual(self.payload["include_domains"], ["rbc.ru", "lenta.ru"])

    def test_news_uses_separate_result_limit(self) -> None:
        client = make_search(max_results=4, news_max_results=9)
        with patch("app.search.urllib.request.urlopen", side_effect=self.fake_urlopen):
            client.search("новости", news=True)
        self.assertEqual(self.payload["max_results"], 9)
        with patch("app.search.urllib.request.urlopen", side_effect=self.fake_urlopen):
            client.search("курс доллара")
        self.assertEqual(self.payload["max_results"], 4)

    def test_skips_results_without_url_or_content(self) -> None:
        body = json.dumps(
            {
                "results": [
                    {"title": "A", "url": "", "content": "x"},
                    {"title": "B", "url": "https://b", "content": ""},
                    {"title": "C", "url": "https://c", "content": "ok"},
                ]
            }
        ).encode("utf-8")
        with patch(
            "app.search.urllib.request.urlopen",
            return_value=FakeResponse(body),
        ):
            results = self.client.search("x")
        self.assertEqual([result.title for result in results], ["C"])

    def test_empty_results(self) -> None:
        with patch(
            "app.search.urllib.request.urlopen",
            return_value=FakeResponse(b'{"results": []}'),
        ):
            self.assertEqual(self.client.search("x"), [])


class TavilyExtractTest(unittest.TestCase):
    def test_extract_returns_raw_content(self) -> None:
        payload: dict = {}
        url = "https://example.com/news"

        def fake_urlopen(request, timeout):
            payload.update(json.loads(request.data.decode("utf-8")))
            self.assertEqual(request.full_url, "https://api.tavily.com/extract")
            return FakeResponse(
                json.dumps(
                    {"results": [{"url": url, "raw_content": "Полный текст статьи."}]}
                ).encode("utf-8")
            )

        client = make_search(extract_depth="advanced")
        with patch("app.search.urllib.request.urlopen", side_effect=fake_urlopen):
            text = client.extract(url)
        self.assertEqual(text, "Полный текст статьи.")
        self.assertEqual(payload["urls"], [url])
        self.assertEqual(payload["extract_depth"], "advanced")
        self.assertEqual(payload["format"], "text")

    def test_extract_without_content_raises(self) -> None:
        body = json.dumps({"results": [{"url": "https://example.com"}]}).encode("utf-8")
        with patch(
            "app.search.urllib.request.urlopen",
            return_value=FakeResponse(body),
        ):
            with self.assertRaises(SearchError):
                make_search().extract("https://example.com")

    def test_extract_empty_url_raises(self) -> None:
        with self.assertRaises(SearchError):
            make_search().extract("  ")


class TavilyArticleSearchTest(unittest.TestCase):
    def test_payload_uses_advanced_and_raw_content(self) -> None:
        payload: dict = {}

        def fake_urlopen(request, timeout):
            payload.update(json.loads(request.data.decode("utf-8")))
            return FakeResponse(b'{"results": []}')

        with patch("app.search.urllib.request.urlopen", side_effect=fake_urlopen):
            make_search().search_article("экспорт автомобилей", limit=3)
        self.assertEqual(payload["search_depth"], "advanced")
        self.assertEqual(payload["topic"], "general")
        self.assertTrue(payload["include_raw_content"])
        self.assertEqual(payload["max_results"], 3)

    def test_filters_unrelated_and_uses_raw_excerpt(self) -> None:
        query = "Экспорт легковых автомобилей из Китая вырос в октябре на 62%"
        relevant = (
            "Экспорт легковых автомобилей из Китая вырос в октябре на 62% "
            "год к году. " + "Подробности поставок. " * 20
        )
        unrelated = "Погода в Москве завтра будет солнечной и тёплой. " * 10
        raw = (
            "По данным ассоциации, экспорт легковых автомобилей из Китая "
            "вырос на 62% в октябре. " + "Детали рынка. " * 30
        )
        body = json.dumps(
            {
                "results": [
                    {
                        "title": "Разбор экспорта",
                        "url": "https://example.com/export",
                        "content": relevant,
                        "score": 0.5,
                    },
                    {
                        "title": "Погода",
                        "url": "https://example.com/weather",
                        "content": unrelated,
                        "score": 0.9,
                    },
                    {
                        "title": "Короткая заметка",
                        "url": "https://example.com/short",
                        "content": "Коротко.",
                        "raw_content": raw,
                        "score": 0.3,
                    },
                ]
            }
        ).encode("utf-8")
        with patch(
            "app.search.urllib.request.urlopen",
            return_value=FakeResponse(body),
        ):
            results = make_search().search_article(query, limit=3)
        urls = [result.url for result in results]
        self.assertIn("https://example.com/export", urls)
        self.assertIn("https://example.com/short", urls)
        self.assertNotIn("https://example.com/weather", urls)
        short = next(result for result in results if result.url.endswith("/short"))
        self.assertIn("ассоциации", short.content)

    def test_short_query_keeps_single_token_match(self) -> None:
        body = json.dumps(
            {
                "results": [
                    {
                        "title": "Курс",
                        "url": "https://example.com/rate",
                        "content": "Курс доллара вырос. " * 20,
                        "score": 0.8,
                    }
                ]
            }
        ).encode("utf-8")
        with patch(
            "app.search.urllib.request.urlopen",
            return_value=FakeResponse(body),
        ):
            results = make_search().search_article("курс", limit=3)
        self.assertEqual(len(results), 1)


class TavilyErrorTest(unittest.TestCase):
    def test_http_error(self) -> None:
        error = urllib.error.HTTPError(
            "https://api.tavily.com/search",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"detail": "bad key"}'),
        )
        with patch("app.search.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(SearchError) as ctx:
                make_search().search("x")
        self.assertIn("401", str(ctx.exception))

    def test_network_error(self) -> None:
        with patch(
            "app.search.urllib.request.urlopen",
            side_effect=urllib.error.URLError("offline"),
        ):
            with self.assertRaises(SearchError) as ctx:
                make_search().search("x")
        self.assertIn("недоступен", str(ctx.exception))

    def test_invalid_json(self) -> None:
        with patch(
            "app.search.urllib.request.urlopen",
            return_value=FakeResponse(b"not json"),
        ):
            with self.assertRaises(SearchError):
                make_search().search("x")


if __name__ == "__main__":
    unittest.main()
