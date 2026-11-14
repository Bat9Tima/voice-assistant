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
            "какое послезавтра число",
            "какой позавчера был день недели",
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

    def test_relative_questions(self) -> None:
        for text in (
            "через 10 дней",
            "через одну неделю",
            "через две недели",
            "через месяц",
            "через полгода",
            "5 дней назад",
            "неделю назад",
            "год назад",
            "какое число будет через 10 дней",
            "какой день недели будет через 10 дней",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_datetime_question(text))

    def test_countdown_questions(self) -> None:
        for text in (
            "сколько дней до нового года",
            "сколько дней осталось до нового года",
            "через сколько дней новый год",
            "сколько дней до 8 марта",
            "сколько дней до лета",
            "сколько времени до нового года",
            "сколько недель до лета",
            "сколько часов до нового года",
            "сколько дней до конца месяца",
            "сколько дней до конца года",
            "сколько дней до конца февраля",
            "сколько дней до конца лета",
            "сколько дней до конца следующего года",
            "сколько дней до конца прошлого месяца",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_datetime_question(text))

    def test_difference_questions(self) -> None:
        for text in (
            "сколько дней между 1 января и 8 марта",
            "сколько дней с 1 января по 8 марта",
            "сколько дней от 20 декабря до 10 января",
            "сколько дней прошло с 1 января",
            "сколько дней прошло с 1 января 2020",
            "сколько недель между 1 января и 8 марта",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_datetime_question(text))

    def test_weekday_with_date_questions(self) -> None:
        for text in (
            "какой день недели был 1 января 2020",
            "какой день недели 1 января 2020",
            "какой день недели будет 8 марта",
            "какой день недели 29 февраля",
            "какой день недели 01.01.2020",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_datetime_question(text))

    def test_days_in_month_and_year_questions(self) -> None:
        for text in (
            "сколько дней в феврале",
            "сколько дней в феврале 2028",
            "сколько дней в январе",
            "сколько дней в месяце",
            "сколько дней в году",
            "сколько дней в 2026 году",
            "сколько дней в 2028 году",
            "сколько дней в високосном году",
            "сколько дней в следующем году",
            "сколько дней в прошлом году",
            "сколько дней в следующем месяце",
            "сколько дней в прошлом месяце",
            "сколько дней в следующем феврале",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_datetime_question(text))

    def test_unrelated_questions_are_not_datetime(self) -> None:
        for text in (
            "новости за сегодня",
            "какая погода завтра",
            "какая погода через 10 дней",
            "что такое фотосинтез",
            "какой день выбрать",
            "какой день недели лучше для встречи",
            "какой год выбрать",
            "какой день недели по гороскопу",
            "сколько дней до зарплаты",
            "какое число в 2024 году",
            "через полтора месяца",
            "полтора месяца назад",
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

    def test_parse_relative(self) -> None:
        request = parse_datetime_question("через две недели")
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.kind, "relative")
        self.assertEqual(request.amount, 2)
        self.assertEqual(request.unit, "week")
        self.assertEqual(request.direction, 1)

        past = parse_datetime_question("5 дней назад")
        assert past is not None
        self.assertEqual(past.amount, 5)
        self.assertEqual(past.unit, "day")
        self.assertEqual(past.direction, -1)

    def test_parse_countdown(self) -> None:
        request = parse_datetime_question("сколько дней до нового года")
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.kind, "countdown")
        self.assertEqual(request.unit, "day")
        assert request.target is not None
        self.assertEqual(request.target.event, "new_year")

    def test_parse_difference(self) -> None:
        request = parse_datetime_question("сколько дней между 1 января и 8 марта")
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.kind, "difference")
        self.assertEqual(request.unit, "day")
        assert request.start is not None
        assert request.end is not None
        self.assertEqual((request.start.day, request.start.month), (1, 1))
        self.assertEqual((request.end.day, request.end.month), (8, 3))

    def test_parse_weekday(self) -> None:
        request = parse_datetime_question("какой день недели был 1 января 2020")
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.kind, "weekday")
        self.assertEqual(request.hint, "past")
        assert request.target is not None
        self.assertEqual(request.target.year, 2020)

    def test_parse_days_in(self) -> None:
        month = parse_datetime_question("сколько дней в феврале 2028")
        self.assertIsNotNone(month)
        assert month is not None
        self.assertEqual(month.kind, "days_in_month")
        assert month.target is not None
        self.assertEqual(month.target.month, 2)
        self.assertEqual(month.target.year, 2028)

        year = parse_datetime_question("сколько дней в 2028 году")
        self.assertIsNotNone(year)
        assert year is not None
        self.assertEqual(year.kind, "days_in_year")
        assert year.target is not None
        self.assertEqual(year.target.year, 2028)


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

    def test_day_after_tomorrow_and_day_before_yesterday(self) -> None:
        self.assertEqual(
            answer_datetime("какое послезавтра число", now=NOW),
            "Послезавтра 15 ноября 2026 года, воскресенье.",
        )
        self.assertEqual(
            answer_datetime("какой позавчера был день недели", now=NOW),
            "Позавчера 11 ноября 2026 года, среда.",
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

    def test_relative_days(self) -> None:
        self.assertEqual(
            answer_datetime("через 10 дней", now=NOW),
            "Через 10 дней будет 23 ноября 2026 года, понедельник.",
        )
        self.assertEqual(
            answer_datetime("5 дней назад", now=NOW),
            "5 дней назад было 8 ноября 2026 года, воскресенье.",
        )

    def test_relative_nearest_days(self) -> None:
        self.assertEqual(
            answer_datetime("через 1 день", now=NOW),
            "Завтра 14 ноября 2026 года, суббота.",
        )
        self.assertEqual(
            answer_datetime("через 2 дня", now=NOW),
            "Послезавтра 15 ноября 2026 года, воскресенье.",
        )
        self.assertEqual(
            answer_datetime("2 дня назад", now=NOW),
            "Позавчера 11 ноября 2026 года, среда.",
        )

    def test_relative_weeks_and_months(self) -> None:
        self.assertEqual(
            answer_datetime("через две недели", now=NOW),
            "Через 2 недели будет 27 ноября 2026 года, пятница.",
        )
        self.assertEqual(
            answer_datetime("через месяц", now=NOW),
            "Через 1 месяц будет 13 декабря 2026 года, воскресенье.",
        )
        self.assertEqual(
            answer_datetime("год назад", now=NOW),
            "1 год назад было 13 ноября 2025 года, четверг.",
        )

    def test_relative_month_end_is_clamped(self) -> None:
        current = datetime(2027, 1, 31, 10, 0, tzinfo=UTC)
        self.assertEqual(
            answer_datetime("через месяц", now=current),
            "Через 1 месяц будет 28 февраля 2027 года, воскресенье.",
        )

    def test_relative_leap_day_is_clamped(self) -> None:
        current = datetime(2028, 2, 29, 10, 0, tzinfo=UTC)
        self.assertEqual(
            answer_datetime("через год", now=current),
            "Через 1 год будет 28 февраля 2029 года, среда.",
        )

    def test_countdown_to_new_year(self) -> None:
        for text in (
            "сколько дней до нового года",
            "сколько дней осталось до нового года",
            "через сколько дней новый год",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    answer_datetime(text, now=NOW),
                    "До нового года осталось 49 дней.",
                )

    def test_countdown_to_dates_and_seasons(self) -> None:
        self.assertEqual(
            answer_datetime("сколько дней до 8 марта", now=NOW),
            "До 8 марта 2027 года осталось 115 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней до лета", now=NOW),
            "До лета осталось 200 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней до 29 февраля", now=NOW),
            "До 29 февраля 2028 года осталось 473 дня.",
        )
        self.assertEqual(
            answer_datetime("сколько дней до 2027 года", now=NOW),
            "До 1 января 2027 года осталось 49 дней.",
        )

    def test_countdown_units(self) -> None:
        self.assertEqual(
            answer_datetime("сколько времени до нового года", now=NOW),
            "До нового года осталось 48 дней 3 часа.",
        )
        self.assertEqual(
            answer_datetime("сколько недель до лета", now=NOW),
            "До лета осталось 28 недель и 4 дня.",
        )

    def test_countdown_to_period_end(self) -> None:
        self.assertEqual(
            answer_datetime("сколько дней до конца месяца", now=NOW),
            "До конца месяца осталось 17 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней до конца года", now=NOW),
            "До конца года осталось 48 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней до конца февраля", now=NOW),
            "До конца февраля осталось 107 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней до конца лета", now=NOW),
            "До конца лета остался 291 день.",
        )

    def test_countdown_to_relative_period_end(self) -> None:
        self.assertEqual(
            answer_datetime("сколько дней до конца следующего года", now=NOW),
            "До конца следующего года осталось 413 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней до конца прошлого года", now=NOW),
            "Конец прошлого года уже в прошлом.",
        )
        self.assertEqual(
            answer_datetime("сколько дней до конца следующего месяца", now=NOW),
            "До конца следующего месяца осталось 48 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней до конца прошлого месяца", now=NOW),
            "Конец прошлого месяца уже в прошлом.",
        )
        # «Конец следующей зимы» неоднозначен: лучше не отвечать, чем соврать.
        self.assertIsNone(parse_datetime_question("сколько дней до конца следующей зимы"))

    def test_countdown_edge_cases(self) -> None:
        on_new_year = datetime(2027, 1, 1, 10, 0, tzinfo=UTC)
        self.assertEqual(
            answer_datetime("сколько дней до нового года", now=on_new_year),
            "Новый год сегодня.",
        )
        day_before = datetime(2026, 12, 31, 23, 0, tzinfo=UTC)
        self.assertEqual(
            answer_datetime("сколько дней до нового года", now=day_before),
            "До нового года остался 1 день.",
        )
        past = datetime(2026, 11, 13, 20, 47, tzinfo=UTC)
        self.assertEqual(
            answer_datetime("сколько дней до 1 января 2020", now=past),
            "1 января 2020 года уже в прошлом.",
        )
        self.assertEqual(
            answer_datetime("сколько дней до 30 февраля", now=NOW),
            "Такой даты нет: 30 февраля.",
        )

    def test_difference_between_dates(self) -> None:
        self.assertEqual(
            answer_datetime("сколько дней между 1 января и 8 марта", now=NOW),
            "От 1 января 2026 года до 8 марта 2026 года 66 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней с 1 января по 8 марта 2020", now=NOW),
            "От 1 января 2020 года до 8 марта 2020 года 67 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней от 20 декабря до 10 января", now=NOW),
            "От 20 декабря 2026 года до 10 января 2027 года 21 день.",
        )

    def test_difference_infers_year(self) -> None:
        self.assertEqual(
            answer_datetime("сколько дней между 20 декабря 2026 и 10 января", now=NOW),
            "От 20 декабря 2026 года до 10 января 2027 года 21 день.",
        )
        self.assertEqual(
            answer_datetime("сколько дней между 20 декабря и 10 января 2027", now=NOW),
            "От 20 декабря 2026 года до 10 января 2027 года 21 день.",
        )
        self.assertEqual(
            answer_datetime("сколько дней между 8 марта и 1 января", now=NOW),
            "От 8 марта 2026 года до 1 января 2027 года 299 дней.",
        )

    def test_difference_weeks(self) -> None:
        self.assertEqual(
            answer_datetime("сколько недель между 1 января и 8 марта 2020", now=NOW),
            "От 1 января 2020 года до 8 марта 2020 года 9 недель и 4 дня.",
        )

    def test_difference_since_date(self) -> None:
        self.assertEqual(
            answer_datetime("сколько дней прошло с 1 января", now=NOW),
            "С 1 января 2026 года прошло 316 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней прошло с 1 января 2020", now=NOW),
            "С 1 января 2020 года прошло 2508 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней прошло с 1 января 2027", now=NOW),
            "1 января 2027 года ещё не наступило.",
        )

    def test_difference_missing_date(self) -> None:
        self.assertEqual(
            answer_datetime("сколько дней между 1 января и 30 февраля", now=NOW),
            "Такой даты нет: 30 февраля.",
        )

    def test_weekday_for_date(self) -> None:
        self.assertEqual(
            answer_datetime("какой день недели 1 января 2020", now=NOW),
            "1 января 2020 года, среда.",
        )
        self.assertEqual(
            answer_datetime("какой день недели 01.01.2020", now=NOW),
            "1 января 2020 года, среда.",
        )
        self.assertEqual(
            answer_datetime("какой день недели будет 8 марта", now=NOW),
            "8 марта 2027 года, понедельник.",
        )
        self.assertEqual(
            answer_datetime("какой день недели 1 января", now=NOW),
            "1 января 2027 года, пятница.",
        )

    def test_weekday_direction_and_leap_day(self) -> None:
        self.assertEqual(
            answer_datetime("какой день недели был 8 марта", now=NOW),
            "8 марта 2026 года, воскресенье.",
        )
        self.assertEqual(
            answer_datetime("какой день недели 29 февраля", now=NOW),
            "29 февраля 2028 года, вторник.",
        )
        self.assertEqual(
            answer_datetime("какой день недели был 29 февраля", now=NOW),
            "29 февраля 2024 года, четверг.",
        )

    def test_weekday_missing_date(self) -> None:
        self.assertEqual(
            answer_datetime("какой день недели 30 февраля", now=NOW),
            "Такой даты нет: 30 февраля.",
        )

    def test_days_in_month(self) -> None:
        self.assertEqual(
            answer_datetime("сколько дней в феврале", now=NOW),
            "В феврале 2026 года 28 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней в феврале 2028", now=NOW),
            "В феврале 2028 года 29 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней в январе", now=NOW),
            "В январе 2026 года 31 день.",
        )
        self.assertEqual(
            answer_datetime("сколько дней в месяце", now=NOW),
            "В ноябре 2026 года 30 дней.",
        )

    def test_days_in_year(self) -> None:
        self.assertEqual(
            answer_datetime("сколько дней в году", now=NOW),
            "В 2026 году 365 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней в 2026 году", now=NOW),
            "В 2026 году 365 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней в 2028 году", now=NOW),
            "В 2028 году 366 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней в високосном году", now=NOW),
            "В високосном году 366 дней.",
        )

    def test_days_in_relative_periods(self) -> None:
        self.assertEqual(
            answer_datetime("сколько дней в следующем году", now=NOW),
            "В 2027 году 365 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней в прошлом году", now=NOW),
            "В 2025 году 365 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней в следующем месяце", now=NOW),
            "В декабре 2026 года 31 день.",
        )
        self.assertEqual(
            answer_datetime("сколько дней в прошлом месяце", now=NOW),
            "В октябре 2026 года 31 день.",
        )
        leap = datetime(2028, 1, 15, 10, 0, tzinfo=UTC)
        self.assertEqual(
            answer_datetime("сколько дней в следующем месяце", now=leap),
            "В феврале 2028 года 29 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней в следующем феврале", now=NOW),
            "В феврале 2027 года 28 дней.",
        )
        self.assertEqual(
            answer_datetime("сколько дней в прошлом феврале", now=NOW),
            "В феврале 2025 года 28 дней.",
        )
        before_leap_february = datetime(2027, 12, 15, 10, 0, tzinfo=UTC)
        self.assertEqual(
            answer_datetime("сколько дней в следующем феврале", now=before_leap_february),
            "В феврале 2028 года 29 дней.",
        )


if __name__ == "__main__":
    unittest.main()
