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
