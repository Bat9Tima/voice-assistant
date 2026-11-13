from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from app.actions import CommandResult
from app.local_planner import LocalPlan, LocalPlannerError
from app.main import Assistant
from app.qa import QAAnswer, QAService, Route
from app.search import SearchError, SearchResult
from app.weather import DailyForecast, GeoPlace, WeatherError, WeatherReport


class FakeRouter:
    def __init__(self, result: CommandResult | None = None) -> None:
        self.result = result
        self.seen: list[str] = []
        self.plan_result = CommandResult(True, True, "Выполнено")
        self.plans: list[tuple[str, dict]] = []

    def handle(self, text: str) -> CommandResult | None:
        self.seen.append(text)
        return self.result

    def execute_plan(self, action: str, args: dict) -> CommandResult:
        self.plans.append((action, args))
        return self.plan_result


class FakeBrain:
    def __init__(self, answer: str = "ответ агента") -> None:
        self.answer = answer
        self.asked: list[str] = []

    def ask(self, text: str) -> str:
        self.asked.append(text)
        return self.answer


class FakeLocalPlanner:
    def __init__(self, plan: LocalPlan | None = None, error: Exception | None = None) -> None:
        self.plan_result = plan
        self.error = error
        self.seen: list[str] = []

    def plan(self, text: str) -> LocalPlan | None:
        self.seen.append(text)
        if self.error is not None:
            raise self.error
        return self.plan_result


class FakeTTS:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.phrases: list[str] = []
        self.said: list[str] = []

    def phrase(self, category: str, phrase_id: str | None = None) -> None:
        self.phrases.append(category)

    def say(self, text: str) -> None:
        self.said.append(text)


class FakeTray:
    def __init__(self, status: dict[str, Any]) -> None:
        self._status = status

    def status(self) -> dict[str, Any]:
        return self._status


class FakeQA:
    def __init__(self, answer: QAAnswer | None = None, kind: str = "general") -> None:
        self.result = answer or QAAnswer(
            True, "вопрос", "Короткий ответ", "Полный ответ", "general"
        )
        self.kind = kind
        self.routed: list[str] = []
        self.answered: list[str] = []
        self.shown: list[QAAnswer] = []

    def route(self, text: str) -> Route:
        self.routed.append(text)
        return Route(self.kind)

    def answer(self, text: str, *, route: Route | None = None) -> QAAnswer:
        self.answered.append(text)
        return self.result

    def show(self, answer: QAAnswer) -> bool:
        self.shown.append(answer)
        return True


class TrayStatusTest(unittest.TestCase):
    @staticmethod
    def make(tray: Any) -> Assistant:
        assistant = Assistant.__new__(Assistant)
        assistant.tray = tray
        assistant.cfg = {"wake_word": "хей луна", "hotkey": "ctrl+alt+space"}
        assistant.stt = SimpleNamespace(loaded=False)
        assistant.wake = None
        assistant._state_lock = threading.Lock()
        assistant._state = "starting"
        assistant._paused = False
        return assistant

    def test_status_exposes_tray_state(self) -> None:
        self.assertIsNone(self.make(None).status()["tray"])

        tray_status = {
            "available": True,
            "running": True,
            "error": None,
            "restarts": 2,
        }
        status = self.make(FakeTray(tray_status)).status()
        self.assertEqual(set(status["tray"]), {"available", "running", "error", "restarts"})
        self.assertEqual(status["tray"], tray_status)


class AgentTriggerRoutingTest(unittest.TestCase):
    def make(self, result: CommandResult | None = None) -> Assistant:
        assistant = Assistant.__new__(Assistant)
        assistant.router = FakeRouter(result)
        assistant.brain = FakeBrain()
        assistant.planner = FakeLocalPlanner()
        assistant.qa = None
        assistant.tts = FakeTTS()
        assistant.agent_triggers = ["спроси агента", "передай агенту", "агент"]
        assistant.notifications = []
        assistant.states = []
        assistant.notify = lambda message, title="Ассистент": assistant.notifications.append(
            message
        )
        assistant.set_state = lambda state: assistant.states.append(state)
        return assistant

    def test_unknown_command_does_not_call_agent(self) -> None:
        assistant = self.make()
        with patch("app.main.beep") as beep:
            message = assistant.run_text("сделай что-нибудь странное")
        self.assertEqual(message, "Не понял команду")
        self.assertEqual(assistant.brain.asked, [])
        self.assertEqual(assistant.router.seen, ["сделай что нибудь странное"])
        beep.assert_called_once_with(320, 120)

    def test_explicit_agent_call(self) -> None:
        assistant = self.make()
        message = assistant.run_text("Спроси агента, какая погода?")
        self.assertEqual(message, "ответ агента")
        self.assertEqual(assistant.brain.asked, ["какая погода"])
        self.assertEqual(assistant.router.seen, [])

    def test_agent_trigger_without_request(self) -> None:
        assistant = self.make()
        with patch("app.main.beep") as beep:
            message = assistant.run_text("агент")
        self.assertEqual(message, "Что спросить у агента?")
        self.assertEqual(assistant.brain.asked, [])
        beep.assert_not_called()

    def test_router_command_with_word_agent_is_not_trigger(self) -> None:
        assistant = self.make(CommandResult(True, True, "Открываю YouTube"))
        message = assistant.run_text("включи фильм агент 007")
        self.assertEqual(message, "Открываю YouTube")
        self.assertEqual(assistant.brain.asked, [])
        self.assertEqual(assistant.router.seen, ["включи фильм агент 007"])

    def test_canonicalizes_text_before_routing(self) -> None:
        assistant = self.make(CommandResult(True, True, "Открываю Google"))
        message = assistant.run_text("открой гугель")
        self.assertEqual(message, "Открываю Google")
        self.assertEqual(assistant.router.seen, ["открой гугл"])

    def test_free_form_command_uses_local_planner(self) -> None:
        assistant = self.make()
        assistant.planner = FakeLocalPlanner(LocalPlan("youtube", {"query": "lo-fi", "index": 1}))
        assistant.router.plan_result = CommandResult(True, True, "Включаю на YouTube: lo-fi")
        message = assistant.run_text("вруби лофай на ютубе")
        self.assertEqual(message, "Включаю на YouTube: lo-fi")
        self.assertEqual(assistant.planner.seen, ["вруби лоуфай на ютубе"])
        self.assertEqual(
            assistant.router.plans,
            [("youtube", {"query": "lo-fi", "index": 1})],
        )

    def test_local_planner_none_falls_back(self) -> None:
        assistant = self.make()
        with patch("app.main.beep") as beep:
            message = assistant.run_text("сделай что-нибудь странное")
        self.assertEqual(message, "Не понял команду")
        self.assertEqual(assistant.planner.seen, ["сделай что нибудь странное"])
        beep.assert_called_once_with(320, 120)

    def test_router_command_skips_local_planner(self) -> None:
        assistant = self.make(CommandResult(True, True, "Открываю YouTube"))
        assistant.run_text("открой ютуб")
        self.assertEqual(assistant.planner.seen, [])

    def test_local_planner_error_falls_back(self) -> None:
        assistant = self.make()
        assistant.planner = FakeLocalPlanner(error=LocalPlannerError("нет сервера"))
        with patch("app.main.beep") as beep:
            message = assistant.run_text("вруби музыку")
        self.assertEqual(message, "Не понял команду")
        beep.assert_called_once_with(320, 120)

    def test_agent_trigger_skips_local_planner(self) -> None:
        assistant = self.make()
        assistant.run_text("спроси агента, какая погода")
        self.assertEqual(assistant.planner.seen, [])

    def test_question_uses_qa_before_local_planner(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA()
        message = assistant.run_text("что такое фотосинтез")
        self.assertEqual(message, "Короткий ответ")
        self.assertEqual(assistant.planner.seen, [])
        self.assertEqual(assistant.qa.routed, ["что такое фотосинтез"])
        self.assertEqual(assistant.qa.answered, ["что такое фотосинтез"])
        self.assertEqual(len(assistant.qa.shown), 1)

    def test_qa_error_beeps_without_tab(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA(QAAnswer(False, "вопрос", "Нужен ключ", "Нужен ключ", "error"))
        with patch("app.main.beep") as beep:
            message = assistant.run_text("какие новости")
        self.assertEqual(message, "Нужен ключ")
        beep.assert_called_once_with(320, 120)
        self.assertEqual(assistant.qa.shown, [])
        self.assertEqual(assistant.tts.phrases, ["error"])
        self.assertEqual(assistant.tts.said, [])

    def test_no_model_reason_is_spoken(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA(
            QAAnswer(
                False,
                "вопрос",
                "Модель недоступна.",
                "Модель недоступна.",
                "no_model",
            )
        )
        with patch("app.main.beep") as beep:
            message = assistant.run_text("что такое фотосинтез")
        self.assertEqual(message, "Модель недоступна.")
        beep.assert_called_once_with(320, 120)
        self.assertEqual(assistant.tts.said, ["Модель недоступна."])
        self.assertEqual(assistant.tts.phrases, [])
        self.assertEqual(assistant.qa.shown, [])

    def test_router_command_does_not_use_qa(self) -> None:
        assistant = self.make(CommandResult(True, True, "Открываю YouTube"))
        assistant.qa = FakeQA()
        message = assistant.run_text("открой ютуб")
        self.assertEqual(message, "Открываю YouTube")
        self.assertEqual(assistant.qa.answered, [])
        self.assertEqual(assistant.qa.shown, [])

    def test_datetime_answer_skips_ack_phrase(self) -> None:
        assistant = self.make()
        assistant.qa = FakeQA(
            QAAnswer(
                True,
                "который час",
                "Сейчас 20 часов 47 минут.",
                "Сейчас 20 часов 47 минут.",
                "time",
            ),
            kind="datetime",
        )
        assistant.tts = FakeTTS(enabled=True)
        message = assistant.run_text("который час")
        self.assertEqual(message, "Сейчас 20 часов 47 минут.")
        self.assertEqual(assistant.tts.phrases, [])
        self.assertEqual(assistant.tts.said, ["Сейчас 20 часов 47 минут."])


class FakeLLM:
    def __init__(
        self,
        text: str = "Ответ модели.",
        available: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.is_available = available
        self.error = error
        self.calls: list[tuple[str, str]] = []
        self.kwargs: list[dict[str, Any]] = []

    def available(self) -> bool:
        return self.is_available

    def answer(self, question: str, context: str = "", **kwargs: Any) -> str:
        self.calls.append((question, context))
        self.kwargs.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return self.text


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
        self.article_calls: list[str] = []
        self.extract_calls: list[str] = []

    def available(self) -> bool:
        return self.is_available

    def search(self, query: str, *, news: bool = False) -> list[SearchResult]:
        self.calls.append((query, news))
        if self.error is not None:
            raise self.error
        return self.results

    def search_article(self, query: str, *, limit: int = 3) -> list[SearchResult]:
        self.article_calls.append(query)
        if self.error is not None:
            raise self.error
        return self.results[:limit]

    def extract(self, url: str) -> str:
        self.extract_calls.append(url)
        if self.error is not None:
            raise self.error
        return ""


class FakeWeather:
    def __init__(
        self,
        report: WeatherReport | None = None,
        error: Exception | None = None,
    ) -> None:
        self.report = report or make_e2e_report()
        self.error = error
        self.cities: list[str | None] = []

    def forecast(self, city: str | None = None) -> WeatherReport:
        self.cities.append(city)
        if self.error is not None:
            raise self.error
        return self.report


class FakeBrowser:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, action: str, **args: Any) -> dict[str, Any]:
        self.calls.append((action, args))
        return {"ok": True}


def make_e2e_report(place: str = "Москва") -> WeatherReport:
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


class CommandPipelineTest(unittest.TestCase):
    """Assistant._process → QAService → внешние сервисы, все внешние вызовы на фейках."""

    def make(
        self,
        *,
        llm: FakeLLM | None = None,
        search: FakeSearch | None = None,
        weather: FakeWeather | None = None,
        browser: FakeBrowser | None = None,
        router: FakeRouter | None = None,
    ) -> Assistant:
        assistant = Assistant.__new__(Assistant)
        assistant.router = router or FakeRouter()
        assistant.brain = FakeBrain()
        assistant.planner = FakeLocalPlanner()
        assistant.tts = FakeTTS(enabled=True)
        assistant.agent_triggers = ["спроси агента", "передай агенту", "агент"]
        assistant.notifications = []
        assistant.states = []

        def notify(message: str, title: str = "Ассистент") -> None:
            assistant.notifications.append(message)

        assistant.notify = notify
        assistant.set_state = assistant.states.append

        qa = QAService(
            {
                "qa": {
                    "enabled": True,
                    "weather": {"city": "Москва"},
                    "search": {"api_key": "test-key"},
                }
            },
            llm=llm,
            browser=browser or FakeBrowser(),
        )
        qa.search = search or FakeSearch()
        qa.weather = weather or FakeWeather()
        assistant.qa = qa
        return assistant

    def test_weather_answer_renders_tab_and_speaks(self) -> None:
        weather = FakeWeather(make_e2e_report())
        browser = FakeBrowser()
        assistant = self.make(weather=weather, browser=browser)

        with patch("app.main.beep") as beep:
            message = assistant.run_text("какая погода в Москве")

        self.assertIn("Москва", message)
        self.assertEqual(weather.cities, ["москве"])
        self.assertEqual(assistant.tts.phrases, ["weather"])
        self.assertEqual(assistant.tts.said, [message])
        self.assertEqual(assistant.states, ["thinking", "idle"])
        beep.assert_not_called()
        self.assertEqual(len(browser.calls), 1)
        action, args = browser.calls[0]
        self.assertEqual(action, "render")
        self.assertIn("Завтра", args["html"])

    def test_weather_followup_reuses_city(self) -> None:
        weather = FakeWeather(make_e2e_report())
        assistant = self.make(weather=weather)

        first = assistant.run_text("какая погода в Москве")
        second = assistant.run_text("а завтра?")

        self.assertIn("Москва", first)
        self.assertIn("завтра", second.lower())
        self.assertEqual(weather.cities, ["москве", "Москва"])
        self.assertEqual(assistant.tts.phrases, ["weather", "weather"])
        self.assertEqual(len(assistant.tts.said), 2)

    def test_general_answer_uses_llm_and_opens_tab(self) -> None:
        llm = FakeLLM("Фотосинтез — это процесс.")
        search = FakeSearch(available=False)
        browser = FakeBrowser()
        assistant = self.make(llm=llm, search=search, browser=browser)

        message = assistant.run_text("что такое фотосинтез")

        self.assertEqual(message, "Фотосинтез — это процесс.")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(search.calls, [])
        self.assertEqual(assistant.tts.phrases, ["wait"])
        self.assertEqual(assistant.tts.said, [message])
        self.assertEqual(len(browser.calls), 1)

    def test_general_followup_passes_history(self) -> None:
        llm = FakeLLM("Фотосинтез — это процесс.")
        assistant = self.make(llm=llm, search=FakeSearch(available=False))

        assistant.run_text("что такое фотосинтез")
        assistant.run_text("а почему это важно?")

        self.assertEqual(len(llm.calls), 2)
        self.assertIn("что такое фотосинтез", llm.kwargs[1]["history"])
        self.assertEqual(assistant.tts.phrases, ["wait", "wait"])

    def test_news_answer_opens_tab_and_remembers_items(self) -> None:
        search = FakeSearch(
            [
                SearchResult(
                    "Выборы",
                    "https://example.com/news",
                    "Свежая новость про выборы и курсы.",
                )
            ]
        )
        llm = FakeLLM("Главная новость про выборы [1].")
        browser = FakeBrowser()
        assistant = self.make(llm=llm, search=search, browser=browser)

        message = assistant.run_text("какие новости")

        self.assertTrue(message)
        self.assertTrue(search.calls[0][1])
        self.assertEqual(assistant.tts.phrases, ["news"])
        self.assertEqual(len(browser.calls), 1)
        self.assertTrue(assistant.qa.last_news)

    def test_empty_search_is_not_an_error(self) -> None:
        browser = FakeBrowser()
        assistant = self.make(search=FakeSearch(results=[]), browser=browser)

        with patch("app.main.beep") as beep:
            message = assistant.run_text("курс доллара")

        self.assertEqual(message, "Ничего не нашлось.")
        self.assertEqual(assistant.tts.phrases, ["wait"])
        self.assertEqual(assistant.tts.said, [message])
        self.assertEqual(browser.calls, [])
        beep.assert_not_called()

    def test_search_failure_speaks_generic_error(self) -> None:
        browser = FakeBrowser()
        assistant = self.make(search=FakeSearch(error=SearchError("boom")), browser=browser)

        with patch("app.main.beep") as beep:
            message = assistant.run_text("курс доллара")

        self.assertEqual(message, "Сервис не отвечает.")
        self.assertEqual(assistant.tts.phrases, ["wait", "error"])
        self.assertEqual(assistant.tts.said, [])
        self.assertEqual(browser.calls, [])
        beep.assert_called_once_with(320, 120)

    def test_no_model_reason_is_spoken(self) -> None:
        browser = FakeBrowser()
        assistant = self.make(
            llm=FakeLLM(available=False),
            search=FakeSearch(available=False),
            browser=browser,
        )

        with patch("app.main.beep") as beep:
            message = assistant.run_text("что такое фотосинтез")

        self.assertEqual(message, "Модель недоступна.")
        self.assertEqual(assistant.tts.phrases, ["wait"])
        self.assertEqual(assistant.tts.said, [message])
        self.assertEqual(browser.calls, [])
        beep.assert_called_once_with(320, 120)

    def test_missing_search_key_is_an_error(self) -> None:
        browser = FakeBrowser()
        assistant = self.make(search=FakeSearch(available=False), browser=browser)

        with patch("app.main.beep") as beep:
            message = assistant.run_text("какие новости")

        self.assertEqual(message, "Нужен ключ Tavily")
        self.assertEqual(assistant.tts.phrases, ["news", "error"])
        self.assertEqual(assistant.tts.said, [])
        self.assertEqual(browser.calls, [])
        beep.assert_called_once_with(320, 120)

    def test_weather_failure_speaks_generic_error(self) -> None:
        browser = FakeBrowser()
        assistant = self.make(weather=FakeWeather(error=WeatherError("down")), browser=browser)

        with patch("app.main.beep") as beep:
            message = assistant.run_text("какая погода")

        self.assertEqual(message, "Не смог узнать погоду.")
        self.assertEqual(assistant.tts.phrases, ["weather", "error"])
        self.assertEqual(assistant.tts.said, [])
        self.assertEqual(browser.calls, [])
        beep.assert_called_once_with(320, 120)

    def test_datetime_answer_is_local_and_silent_ack(self) -> None:
        llm = FakeLLM()
        search = FakeSearch()
        weather = FakeWeather()
        browser = FakeBrowser()
        assistant = self.make(llm=llm, search=search, weather=weather, browser=browser)

        message = assistant.run_text("какое сегодня число")

        self.assertTrue(message.startswith("Сегодня "))
        self.assertEqual(assistant.tts.phrases, [])
        self.assertEqual(assistant.tts.said, [message])
        self.assertEqual(llm.calls, [])
        self.assertEqual(search.calls, [])
        self.assertEqual(weather.cities, [])
        self.assertEqual(browser.calls, [])
        self.assertIsNone(assistant.qa._last_turn)

    def test_browser_command_bypasses_qa(self) -> None:
        router = FakeRouter(CommandResult(True, True, "Открываю YouTube"))
        llm = FakeLLM()
        search = FakeSearch()
        weather = FakeWeather()
        browser = FakeBrowser()
        assistant = self.make(
            llm=llm,
            search=search,
            weather=weather,
            browser=browser,
            router=router,
        )

        with patch("app.main.beep") as beep:
            message = assistant.run_text("открой ютуб")

        self.assertEqual(message, "Открываю YouTube")
        self.assertEqual(router.seen, ["открой ютуб"])
        self.assertEqual(assistant.tts.phrases, ["ack"])
        self.assertEqual(llm.calls, [])
        self.assertEqual(search.calls, [])
        self.assertEqual(weather.cities, [])
        self.assertEqual(browser.calls, [])
        beep.assert_not_called()

    def test_agent_trigger_bypasses_qa(self) -> None:
        llm = FakeLLM()
        search = FakeSearch()
        weather = FakeWeather()
        assistant = self.make(llm=llm, search=search, weather=weather)

        message = assistant.run_text("спроси агента, какая погода")

        self.assertEqual(message, "ответ агента")
        self.assertEqual(assistant.brain.asked, ["какая погода"])
        self.assertEqual(assistant.tts.phrases, ["agent"])
        self.assertEqual(assistant.tts.said, ["ответ агента"])
        self.assertEqual(llm.calls, [])
        self.assertEqual(search.calls, [])
        self.assertEqual(weather.cities, [])

    def test_unexpected_error_is_caught(self) -> None:
        assistant = self.make(llm=FakeLLM(error=RuntimeError("boom")))

        with patch("app.main.beep") as beep, patch("app.main.log.exception") as log_error:
            message = assistant.run_text("что такое фотосинтез")

        self.assertEqual(message, "Ошибка: boom")
        self.assertEqual(assistant.tts.phrases, ["wait", "error"])
        self.assertEqual(assistant.tts.said, [])
        self.assertEqual(assistant.states, ["thinking", "idle"])
        beep.assert_called_once_with(320, 120)
        log_error.assert_called_once()

    def test_unknown_command(self) -> None:
        assistant = self.make()

        with patch("app.main.beep") as beep:
            message = assistant.run_text("сделай что-нибудь странное")

        self.assertEqual(message, "Не понял команду")
        self.assertEqual(assistant.tts.phrases, ["ack", "unknown"])
        beep.assert_called_once_with(320, 120)


if __name__ == "__main__":
    unittest.main()
