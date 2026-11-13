from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta, timezone

from app.datetime_answer import (
    answer_datetime,
    datetime_context,
    is_datetime_question,
    parse_datetime_question,
)

NOW = datetime(2026, 11, 13, 20, 47, tzinfo=UTC)


class DatetimeQuestionTest(unittest.TestCase):
    def test_time_questions(self) -> None:
        for text in (
            "который час",
            "который сейчас час",
            "сколько времени",
            "сколько сейчас времени",
            "сколько время",
            "какое сейчас время",
            "какое время",
            "сколько на часах",
            "текущее время",
            "скажи пожалуйста который час",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_datetime_question(text))

    def test_date_questions(self) -> None:
        for text in (
            "какое сегодня число",
            "какое число",
            "какое число сегодня",
            "какая сегодня дата",
            "какая дата",
            "какой сегодня день недели",
            "какой день недели",
            "какой сегодня день",
            "какое завтра число",
            "какое число будет завтра",
            "какое вчера было число",
            "какой вчера был день недели",
            "какой день недели был вчера",
            "сегодня какое число",
            "завтра какой день недели",
            "какое сегодняшнее число",
            "какое нынче число",
            "какой нынче день",
            "какое нынешнее число",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_datetime_question(text))

    def test_year_questions(self) -> None:
        for text in (
            "какой сейчас год",
            "какой год",
            "какой год сейчас",
            "сейчас какой год",
            "какой теперь год",
            "в каком году мы живем",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_datetime_question(text))

    def test_durations_and_other_questions_are_not_datetime(self) -> None:
        for text in (
            "сколько времени до нового года",
            "сколько дней до нового года",
            "через сколько дней новый год",
            "какое число в 2024 году",
            "какой день недели был 1 января 2020",
            "какой сегодня день недели по гороскопу",
            "новости за сегодня",
            "какая погода завтра",
            "что такое фотосинтез",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_datetime_question(text))

    def test_unrelated_day_and_year_phrases(self) -> None:
        for text in (
            "какой день выбрать",
            "какой день недели лучше для встречи",
            "какой год выбрать",
            "сколько дней до зарплаты",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_datetime_question(text))

    def test_parse_returns_structured_request(self) -> None:
        request = parse_datetime_question("какое завтра число")
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.kind, "date")
        self.assertEqual(request.shift, 1)
        self.assertIsNone(parse_datetime_question("что такое фотосинтез"))


class DatetimeAnswerTest(unittest.TestCase):
    def test_time_answer(self) -> None:
        self.assertEqual(
            answer_datetime("который час", now=NOW),
            "Сейчас 20 часов 47 минут.",
        )

    def test_time_plural_forms(self) -> None:
        cases = (
            (datetime(2026, 11, 13, 20, 1, tzinfo=UTC), "20 часов 1 минута"),
            (datetime(2026, 11, 13, 20, 2, tzinfo=UTC), "20 часов 2 минуты"),
            (datetime(2026, 11, 13, 20, 5, tzinfo=UTC), "20 часов 5 минут"),
        )
        for current, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    answer_datetime("сколько времени", now=current),
                    f"Сейчас {expected}.",
                )

    def test_time_hour_plural_forms(self) -> None:
        cases = (
            (datetime(2026, 11, 13, 1, 0, tzinfo=UTC), "1 час"),
            (datetime(2026, 11, 13, 2, 0, tzinfo=UTC), "2 часа"),
            (datetime(2026, 11, 13, 5, 0, tzinfo=UTC), "5 часов"),
            (datetime(2026, 11, 13, 21, 0, tzinfo=UTC), "21 час"),
            (datetime(2026, 11, 13, 22, 0, tzinfo=UTC), "22 часа"),
        )
        for current, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    answer_datetime("какое время", now=current),
                    f"Сейчас {expected} ровно.",
                )

    def test_midnight(self) -> None:
        current = datetime(2026, 11, 13, 0, 5, tzinfo=UTC)
        self.assertEqual(
            answer_datetime("который час", now=current),
            "Сейчас ноль часов 5 минут.",
        )

    def test_today_date(self) -> None:
        self.assertEqual(
            answer_datetime("какое сегодня число", now=NOW),
            "Сегодня 13 ноября 2026 года, пятница.",
        )

    def test_tomorrow_and_yesterday(self) -> None:
        self.assertEqual(
            answer_datetime("какое завтра число", now=NOW),
            "Завтра 14 ноября 2026 года, суббота.",
        )
        self.assertEqual(
            answer_datetime("какой вчера был день недели", now=NOW),
            "Вчера 12 ноября 2026 года, четверг.",
        )

    def test_current_year(self) -> None:
        self.assertEqual(
            answer_datetime("какой сейчас год", now=NOW),
            "Сейчас 2026 год.",
        )

    def test_year_rollover(self) -> None:
        current = datetime(2026, 12, 31, 23, 50, tzinfo=UTC)
        self.assertEqual(
            answer_datetime("какое завтра число", now=current),
            "Завтра 1 января 2027 года, пятница.",
        )

    def test_leap_day(self) -> None:
        before = datetime(2028, 2, 28, 22, 0, tzinfo=UTC)
        self.assertEqual(
            answer_datetime("какое завтра число", now=before),
            "Завтра 29 февраля 2028 года, вторник.",
        )
        on_leap_day = datetime(2028, 2, 29, 22, 0, tzinfo=UTC)
        self.assertEqual(
            answer_datetime("какое завтра число", now=on_leap_day),
            "Завтра 1 марта 2028 года, среда.",
        )

    def test_midnight_and_last_minute(self) -> None:
        midnight = datetime(2026, 11, 13, 0, 0, tzinfo=UTC)
        self.assertEqual(
            answer_datetime("который час", now=midnight),
            "Сейчас ноль часов ровно.",
        )
        late = datetime(2026, 11, 13, 23, 59, tzinfo=UTC)
        self.assertEqual(
            answer_datetime("сколько времени", now=late),
            "Сейчас 23 часа 59 минут.",
        )

    def test_other_offset_keeps_the_date(self) -> None:
        current = datetime(2026, 11, 13, 0, 10, tzinfo=timezone(timedelta(hours=10)))
        self.assertEqual(
            answer_datetime("какое сегодня число", now=current),
            "Сегодня 13 ноября 2026 года, пятница.",
        )

    def test_context_for_model(self) -> None:
        self.assertEqual(
            datetime_context(NOW),
            "Сегодня 13 ноября 2026 года, пятница. Сейчас 20:47.",
        )

    def test_not_datetime_returns_none(self) -> None:
        self.assertIsNone(answer_datetime("что такое фотосинтез", now=NOW))


if __name__ == "__main__":
    unittest.main()
