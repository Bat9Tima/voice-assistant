from __future__ import annotations

import unittest

from PIL import Image

from app.frame_processor import ChangeResult
from app.screen_capture import Frame
from app.screen_watch import WatchEvent
from app.vision_events import (
    KIND_CHANGE,
    KIND_IDLE,
    KIND_START,
    KIND_WINDOW,
    EventTimeline,
)


def make_event(
    *,
    at: float = 0.0,
    changed: bool = False,
    window_changed: bool = False,
    title: str | None = None,
    baseline: bool = False,
) -> WatchEvent:
    result = None
    if not baseline:
        result = ChangeResult(
            changed=changed,
            score=0.2 if changed else 0.0,
            pixel_diff=0.1 if changed else 0.0,
            hash_distance=3 if changed else 0,
            hash_ratio=0.05 if changed else 0.0,
            reason="content" if changed else "same",
        )
    return WatchEvent(
        at=at,
        changed=changed or window_changed,
        window_changed=window_changed,
        result=result,
        width=320,
        height=180,
        window_title=title,
    )


def make_frame(*, at: float = 0.0, title: str | None = None) -> Frame:
    return Frame(
        image=Image.new("RGB", (32, 18), "white"),
        captured_at=at,
        window_title=title,
    )


class EventTimelineTest(unittest.TestCase):
    def make(self, **kwargs: float) -> EventTimeline:
        return EventTimeline(**kwargs)  # type: ignore[arg-type]

    def test_baseline_starts_timeline(self) -> None:
        timeline = self.make()

        recorded = timeline.record(make_event(baseline=True, title="Терминал"), now=100.0)

        self.assertIsNotNone(recorded)
        assert recorded is not None
        self.assertEqual(recorded.kind, KIND_START)
        self.assertEqual(recorded.window_title, "Терминал")
        self.assertEqual(timeline.events(), (recorded,))

    def test_window_change_is_separate_event(self) -> None:
        timeline = self.make()
        timeline.record(make_event(baseline=True, title="Терминал"), now=100.0)

        recorded = timeline.record(
            make_event(window_changed=True, title="Chrome — GitHub"),
            now=102.0,
        )

        self.assertIsNotNone(recorded)
        assert recorded is not None
        self.assertEqual(recorded.kind, KIND_WINDOW)
        self.assertEqual(recorded.previous_title, "Терминал")
        self.assertEqual(recorded.window_title, "Chrome — GitHub")

    def test_content_change_is_recorded(self) -> None:
        timeline = self.make()
        timeline.record(make_event(baseline=True, title="Терминал"), now=100.0)

        recorded = timeline.record(make_event(changed=True, title="Терминал"), now=102.0)

        self.assertIsNotNone(recorded)
        assert recorded is not None
        self.assertEqual(recorded.kind, KIND_CHANGE)
        self.assertEqual(recorded.count, 1)

    def test_rapid_changes_are_coalesced(self) -> None:
        timeline = self.make(coalesce_sec=10.0)
        timeline.record(make_event(baseline=True, title="Терминал"), now=100.0)
        timeline.record(make_event(changed=True, title="Терминал"), now=102.0)

        merged = timeline.record(make_event(changed=True, title="Терминал"), now=105.0)

        self.assertIsNotNone(merged)
        assert merged is not None
        self.assertEqual(merged.kind, KIND_CHANGE)
        self.assertEqual(merged.count, 2)
        self.assertEqual(merged.at, 105.0)
        self.assertEqual(len(timeline.events()), 2)

        separate = timeline.record(make_event(changed=True, title="Терминал"), now=120.0)

        self.assertIsNotNone(separate)
        assert separate is not None
        self.assertEqual(separate.count, 1)
        self.assertEqual(len(timeline.events()), 3)

    def test_changes_in_different_windows_are_not_coalesced(self) -> None:
        timeline = self.make(coalesce_sec=10.0)
        timeline.record(make_event(baseline=True, title="Терминал"), now=100.0)
        timeline.record(make_event(changed=True, title="Терминал"), now=102.0)

        recorded = timeline.record(make_event(changed=True, title="Chrome"), now=103.0)

        self.assertIsNotNone(recorded)
        assert recorded is not None
        self.assertEqual(recorded.kind, KIND_CHANGE)
        self.assertEqual(recorded.window_title, "Chrome")

    def test_idle_is_reported_once_until_activity(self) -> None:
        timeline = self.make(idle_after_sec=60.0)
        timeline.record(make_event(baseline=True, title="Терминал"), now=100.0)

        self.assertIsNone(timeline.record(make_event(title="Терминал"), now=150.0))
        idle = timeline.record(make_event(title="Терминал"), now=160.0)

        self.assertIsNotNone(idle)
        assert idle is not None
        self.assertEqual(idle.kind, KIND_IDLE)
        self.assertEqual(idle.detail, "1 минуту")
        self.assertIsNone(timeline.record(make_event(title="Терминал"), now=170.0))

        timeline.record(make_event(changed=True, title="Терминал"), now=175.0)
        again = timeline.record(make_event(title="Терминал"), now=240.0)

        self.assertIsNotNone(again)
        assert again is not None
        self.assertEqual(again.kind, KIND_IDLE)

    def test_zero_idle_after_disables_idle_events(self) -> None:
        timeline = self.make(idle_after_sec=0.0)
        timeline.record(make_event(baseline=True), now=100.0)

        self.assertIsNone(timeline.record(make_event(), now=10_000.0))
        self.assertEqual(len(timeline.events()), 1)

    def test_max_events_keeps_the_latest(self) -> None:
        timeline = self.make(max_events=3)
        timeline.record(make_event(baseline=True, title="A"), now=100.0)
        for index, title in enumerate(("B", "C", "D", "E"), start=1):
            timeline.record(
                make_event(window_changed=True, title=title),
                now=100.0 + index,
            )

        events = timeline.events()

        self.assertEqual(len(events), 3)
        self.assertEqual([event.window_title for event in events], ["C", "D", "E"])

    def test_reset_forgets_everything(self) -> None:
        timeline = self.make()
        timeline.record(
            make_event(baseline=True, title="Терминал"),
            now=100.0,
            frame=make_frame(title="Терминал"),
        )

        timeline.reset()

        self.assertEqual(timeline.events(), ())
        self.assertEqual(timeline.frames(), ())
        self.assertEqual(timeline.describe(now=100.0), "Событий пока нет.")

    def test_events_limit_returns_the_tail(self) -> None:
        timeline = self.make()
        timeline.record(make_event(baseline=True, title="A"), now=100.0)
        timeline.record(make_event(window_changed=True, title="B"), now=101.0)
        timeline.record(make_event(window_changed=True, title="C"), now=102.0)

        self.assertEqual([e.window_title for e in timeline.events(limit=2)], ["B", "C"])
        self.assertEqual(timeline.events(limit=0), ())

    def test_key_frames_store_start_and_window(self) -> None:
        timeline = self.make()
        timeline.record(
            make_event(baseline=True, title="A"),
            now=100.0,
            frame=make_frame(title="A"),
        )
        timeline.record(
            make_event(window_changed=True, title="B"),
            now=102.0,
            frame=make_frame(title="B"),
        )

        frames = timeline.frames()

        self.assertEqual([item.kind for item in frames], [KIND_START, KIND_WINDOW])
        self.assertEqual([item.frame.window_title for item in frames], ["A", "B"])

    def test_change_frames_respect_gap(self) -> None:
        timeline = self.make(coalesce_sec=0.0, frame_gap_sec=10.0)
        timeline.record(
            make_event(baseline=True, title="A"),
            now=100.0,
            frame=make_frame(title="A"),
        )
        timeline.record(
            make_event(changed=True, title="A"),
            now=105.0,
            frame=make_frame(title="A"),
        )

        self.assertEqual(len(timeline.frames()), 1)

        timeline.record(
            make_event(changed=True, title="A"),
            now=112.0,
            frame=make_frame(title="A"),
        )

        self.assertEqual(len(timeline.frames()), 2)

    def test_window_change_ignores_frame_gap(self) -> None:
        timeline = self.make(frame_gap_sec=60.0)
        timeline.record(
            make_event(baseline=True, title="A"),
            now=100.0,
            frame=make_frame(title="A"),
        )
        timeline.record(
            make_event(window_changed=True, title="B"),
            now=102.0,
            frame=make_frame(title="B"),
        )

        self.assertEqual([item.window_title for item in timeline.frames()], ["A", "B"])

    def test_key_frames_are_bounded(self) -> None:
        timeline = self.make(max_frames=2)
        for index, title in enumerate(("A", "B", "C"), start=1):
            timeline.record(
                make_event(window_changed=True, title=title),
                now=100.0 + index,
                frame=make_frame(title=title),
            )

        self.assertEqual([item.window_title for item in timeline.frames()], ["B", "C"])

    def test_zero_max_frames_disables_key_frames(self) -> None:
        timeline = self.make(max_frames=0)
        timeline.record(
            make_event(baseline=True, title="A"),
            now=100.0,
            frame=make_frame(title="A"),
        )

        self.assertEqual(timeline.frames(), ())

    def test_frames_limit_returns_the_tail(self) -> None:
        timeline = self.make(max_frames=4)
        for index, title in enumerate(("A", "B", "C"), start=1):
            timeline.record(
                make_event(window_changed=True, title=title),
                now=100.0 + index,
                frame=make_frame(title=title),
            )

        self.assertEqual([item.window_title for item in timeline.frames(limit=2)], ["B", "C"])
        self.assertEqual(timeline.frames(limit=0), ())

    def test_record_without_frame_does_not_break(self) -> None:
        timeline = self.make()
        timeline.record(make_event(baseline=True, title="A"), now=100.0)

        self.assertEqual(timeline.frames(), ())


class DescribeEventsTest(unittest.TestCase):
    def make_timeline(self) -> EventTimeline:
        timeline = EventTimeline()
        timeline.record(make_event(baseline=True, title="Терминал"), now=100.0)
        timeline.record(make_event(window_changed=True, title="Chrome — GitHub"), now=130.0)
        timeline.record(make_event(changed=True, title="Chrome — GitHub"), now=190.0)
        return timeline

    def test_empty_timeline(self) -> None:
        self.assertEqual(EventTimeline().describe(now=100.0), "Событий пока нет.")

    def test_describe_orders_events_and_uses_relative_time(self) -> None:
        text = self.make_timeline().describe(now=200.0)

        self.assertTrue(text.startswith("Что происходило: "))
        self.assertIn("1 минуту 40 секунд назад — наблюдение началось, окно «Терминал»", text)
        self.assertIn("1 минуту 10 секунд назад — активное окно — «Chrome — GitHub»", text)
        self.assertIn("10 секунд назад — экран изменился, окно «Chrome — GitHub»", text)

    def test_recent_event_is_marked_as_just_now(self) -> None:
        timeline = EventTimeline()
        timeline.record(make_event(baseline=True, title="Терминал"), now=100.0)

        text = timeline.describe(now=102.0)

        self.assertIn("только что — наблюдение началось", text)

    def test_describe_truncates_to_limit(self) -> None:
        text = self.make_timeline().describe(now=200.0, limit=2)

        self.assertTrue(text.startswith("Последние 2 события: "))
        self.assertNotIn("наблюдение началось", text)
        self.assertIn("активное окно", text)
        self.assertIn("экран изменился", text)

    def test_coalesced_change_reports_count(self) -> None:
        timeline = EventTimeline(coalesce_sec=10.0)
        timeline.record(make_event(baseline=True, title="Терминал"), now=100.0)
        timeline.record(make_event(changed=True, title="Терминал"), now=102.0)
        timeline.record(make_event(changed=True, title="Терминал"), now=104.0)

        text = timeline.describe(now=110.0)

        self.assertIn("экран изменился 2 раза, окно «Терминал»", text)

    def test_idle_event_is_described(self) -> None:
        timeline = EventTimeline(idle_after_sec=60.0)
        timeline.record(make_event(baseline=True, title="Терминал"), now=100.0)
        timeline.record(make_event(title="Терминал"), now=160.0)

        text = timeline.describe(now=170.0)

        self.assertIn("экран не менялся 1 минуту", text)

    def test_long_title_is_shortened(self) -> None:
        timeline = EventTimeline()
        timeline.record(
            make_event(baseline=True, title="Очень длинный заголовок " * 5),
            now=100.0,
        )

        text = timeline.describe(now=101.0)

        title = text[text.index("«") + 1 : text.index("»")]
        self.assertLessEqual(len(title), 60)
        self.assertTrue(title.endswith("…"))


if __name__ == "__main__":
    unittest.main()
