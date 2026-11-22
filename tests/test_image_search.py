from __future__ import annotations

import base64
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from app.image_search import (
    DEFAULT_TRIGGERS,
    ENDPOINT,
    ImageMatch,
    ImageSearchError,
    ImageSearchResult,
    YandexImageSearch,
    clean_answer,
    fallback_answer,
    format_results,
    match_request,
)


class FakeResponse:
    def __init__(self, payload: dict, headers: dict[str, str] | None = None) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class RawResponse(FakeResponse):
    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self._body = body
        self.headers = headers or {}


def http_error(status: int, body: dict) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        ENDPOINT,
        status,
        "error",
        {"x-request-id": "request-1"},
        io.BytesIO(json.dumps(body).encode("utf-8")),
    )


def make_search(**section: object) -> YandexImageSearch:
    cfg = {
        "image_search": {
            "enabled": True,
            "api_key": "test-key",
            "folder_id": "test-folder",
            "timeout_sec": 1,
            **section,
        }
    }
    return YandexImageSearch(cfg)


class MatchRequestTest(unittest.TestCase):
    def test_exact_phrase(self) -> None:
        self.assertEqual(match_request("поищи по фото", DEFAULT_TRIGGERS), (True, ""))

    def test_question_after_trigger(self) -> None:
        self.assertEqual(
            match_request("Поищи по фото, кто это?", DEFAULT_TRIGGERS),
            (True, "кто это"),
        )

    def test_photo_question_is_trigger(self) -> None:
        self.assertEqual(match_request("что это за фото", DEFAULT_TRIGGERS), (True, ""))
        self.assertEqual(match_request("что за фото", DEFAULT_TRIGGERS), (True, ""))

    def test_general_questions_are_not_image_search(self) -> None:
        for phrase in (
            "кто это",
            "кто это на фото",
            "что на экране",
            "найди видео",
            "поищи в интернете новости",
            "включи фото на экране",
            "посмотри на экран",
            "",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(match_request(phrase, DEFAULT_TRIGGERS), (False, ""))


class ParseAndFormatTest(unittest.TestCase):
    def test_camel_case_response(self) -> None:
        payload = {
            "images": [
                {
                    "url": "https://img.example/1.jpg",
                    "pageUrl": "https://page.example/1",
                    "pageTitle": "Илон Маск",
                    "passage": "<b>Илон</b> Маск — предприниматель.",
                    "host": "ru.wikipedia.org",
                    "width": "100",
                    "height": 200,
                    "format": "IMAGE_FORMAT_JPEG",
                }
            ],
            "page": "0",
            "id": "cbir-1",
        }
        with patch(
            "app.image_search.urllib.request.urlopen",
            return_value=FakeResponse(payload),
        ):
            result = make_search().search(b"jpeg")
        self.assertEqual(len(result.images), 1)
        item = result.images[0]
        self.assertEqual(item.page_url, "https://page.example/1")
        self.assertEqual(item.page_title, "Илон Маск")
        self.assertEqual(item.passage, "Илон Маск — предприниматель.")
        self.assertEqual(item.host, "ru.wikipedia.org")
        self.assertEqual(item.width, 100)
        self.assertEqual(item.height, 200)
        self.assertEqual(result.cbir_id, "cbir-1")

    def test_snake_case_and_missing_fields(self) -> None:
        payload = {
            "images": [
                {
                    "page_url": "https://page.example/2",
                    "page_title": "Заголовок",
                    "passage": "Текст",
                    "host": "example.com",
                }
            ]
        }
        with patch(
            "app.image_search.urllib.request.urlopen",
            return_value=FakeResponse(payload),
        ):
            result = make_search().search(b"jpeg")
        item = result.images[0]
        self.assertEqual(item.page_url, "https://page.example/2")
        self.assertEqual(item.page_title, "Заголовок")
        self.assertEqual(item.url, "")
        self.assertEqual(item.width, 0)

    def test_empty_images(self) -> None:
        with patch(
            "app.image_search.urllib.request.urlopen",
            return_value=FakeResponse({"images": []}),
        ):
            result = make_search().search(b"jpeg")
        self.assertEqual(result.images, ())
        self.assertEqual(fallback_answer(result), "Ничего похожего не нашёл")

    def test_html_is_cleaned_from_title_and_passage(self) -> None:
        payload = {
            "images": [
                {
                    "pageTitle": "<b>Илон</b> Маск",
                    "passage": "Предприниматель &amp; изобретатель.",
                    "host": "ru.wikipedia.org",
                }
            ]
        }
        with patch(
            "app.image_search.urllib.request.urlopen",
            return_value=FakeResponse(payload),
        ):
            result = make_search().search(b"jpeg")
        self.assertEqual(result.images[0].page_title, "Илон Маск")
        self.assertEqual(result.images[0].passage, "Предприниматель & изобретатель.")

    def test_urls_are_removed_from_snippets(self) -> None:
        payload = {
            "images": [
                {
                    "pageTitle": "Статья",
                    "passage": "Подробнее: https://example.com/page и дальше.",
                }
            ]
        }
        with patch(
            "app.image_search.urllib.request.urlopen",
            return_value=FakeResponse(payload),
        ):
            result = make_search().search(b"jpeg")
        self.assertEqual(result.images[0].passage, "Подробнее: и дальше.")

    def test_empty_result_is_skipped(self) -> None:
        with patch(
            "app.image_search.urllib.request.urlopen",
            return_value=FakeResponse({"images": [{}]}),
        ):
            result = make_search().search(b"jpeg")
        self.assertEqual(result.images, ())

    def test_clean_answer_strips_citations_and_links(self) -> None:
        text = "Это [Илон Маск](https://example.com) [1]. Источник: https://example.com"
        self.assertEqual(clean_answer(text), "Это Илон Маск. Источник:")

    def test_format_results_has_no_urls(self) -> None:
        result = ImageSearchResult(
            images=(
                ImageMatch(
                    page_title="Илон Маск",
                    host="ru.wikipedia.org",
                    passage="Предприниматель и изобретатель.",
                ),
                ImageMatch(
                    page_title="SpaceX",
                    host="spacex.com",
                    passage="Аэрокосмическая компания.",
                ),
            )
        )
        text = format_results(result, limit=1, max_passage_chars=10)
        self.assertIn("1. Илон Маск — ru.wikipedia.org", text)
        self.assertIn("Предприним…", text)
        self.assertNotIn("SpaceX", text)
        self.assertNotIn("https://", text)

    def test_fallback_uses_first_title(self) -> None:
        result = ImageSearchResult(images=(ImageMatch(page_title="Илон Маск"),))
        self.assertEqual(
            fallback_answer(result),
            "Нашёл похожие изображения. Первый результат: Илон Маск.",
        )


class SearchRequestTest(unittest.TestCase):
    def test_payload_and_headers(self) -> None:
        payload = {"images": [{"pageTitle": "Тест"}]}
        with patch(
            "app.image_search.urllib.request.urlopen",
            return_value=FakeResponse(payload),
        ) as urlopen:
            make_search().search(b"jpeg-bytes")
        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["folderId"], "test-folder")
        self.assertEqual(body["data"], base64.b64encode(b"jpeg-bytes").decode("ascii"))
        self.assertEqual(body["familyMode"], "FAMILY_MODE_MODERATE")
        self.assertEqual(body["page"], "0")
        self.assertNotIn("site", body)
        self.assertEqual(request.get_header("Authorization"), "Api-Key test-key")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(request.full_url, ENDPOINT)

    def test_site_is_sent_when_set(self) -> None:
        with patch(
            "app.image_search.urllib.request.urlopen",
            return_value=FakeResponse({"images": []}),
        ) as urlopen:
            make_search().search(b"jpeg", site="ya.ru")
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body["site"], "ya.ru")

    def test_cooldown_blocks_repeat(self) -> None:
        search = make_search(cooldown_sec=60)
        with patch(
            "app.image_search.urllib.request.urlopen",
            return_value=FakeResponse({"images": []}),
        ):
            search.search(b"jpeg")
            with self.assertRaises(ImageSearchError) as ctx:
                search.search(b"jpeg")
        self.assertIn("подождите", str(ctx.exception))

    def test_zero_cooldown_allows_repeat(self) -> None:
        search = make_search(cooldown_sec=0)
        with patch(
            "app.image_search.urllib.request.urlopen",
            return_value=FakeResponse({"images": []}),
        ) as urlopen:
            search.search(b"jpeg")
            search.search(b"jpeg")
        self.assertEqual(urlopen.call_count, 2)

    def test_missing_key(self) -> None:
        with self.assertRaises(ImageSearchError) as ctx:
            make_search(api_key="").search(b"jpeg")
        self.assertIn("ключ", str(ctx.exception))

    def test_missing_folder(self) -> None:
        with self.assertRaises(ImageSearchError) as ctx:
            make_search(folder_id="").search(b"jpeg")
        self.assertIn("folder_id", str(ctx.exception))

    def test_disabled(self) -> None:
        with self.assertRaises(ImageSearchError) as ctx:
            make_search(enabled=False).search(b"jpeg")
        self.assertIn("выключен", str(ctx.exception))

    def test_empty_image(self) -> None:
        with self.assertRaises(ImageSearchError) as ctx:
            make_search().search(b"")
        self.assertIn("Пустой", str(ctx.exception))

    def test_too_large_image(self) -> None:
        with self.assertRaises(ImageSearchError) as ctx:
            make_search().search(b"x" * (3 * 1024 * 1024 + 1))
        self.assertIn("3 МБ", str(ctx.exception))

    def test_401_maps_to_user_message(self) -> None:
        error = http_error(401, {"code": 16, "message": "Unknown api key"})
        with patch("app.image_search.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ImageSearchError) as ctx:
                make_search().search(b"jpeg")
        self.assertEqual(str(ctx.exception), "Яндекс не принял ключ поиска")
        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(ctx.exception.request_id, "request-1")

    def test_429_maps_to_user_message(self) -> None:
        error = http_error(429, {"code": 8, "message": "Quota exceeded"})
        with patch("app.image_search.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ImageSearchError) as ctx:
                make_search().search(b"jpeg")
        self.assertEqual(str(ctx.exception), "Лимит поиска по фото исчерпан")

    def test_network_error(self) -> None:
        with patch(
            "app.image_search.urllib.request.urlopen",
            side_effect=urllib.error.URLError("boom"),
        ):
            with self.assertRaises(ImageSearchError) as ctx:
                make_search().search(b"jpeg")
        self.assertEqual(str(ctx.exception), "Поиск по фото не ответил")

    def test_invalid_json(self) -> None:
        with patch(
            "app.image_search.urllib.request.urlopen",
            return_value=RawResponse(b"not-json"),
        ):
            with self.assertRaises(ImageSearchError) as ctx:
                make_search().search(b"jpeg")
        self.assertIn("непонятный ответ", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
