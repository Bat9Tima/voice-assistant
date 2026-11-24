"""Диагностика наблюдения: захват → сравнение → событие.

В v1.0 модуль не подключён к ассистенту и не вызывает модели. Он нужен,
чтобы проверить захват и пороги на реальном экране:

    python -m app.screen_watch --seconds 15 --interval 2

Кадры остаются в памяти, на диск ничего не пишется.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from .frame_processor import (
    DEFAULT_HASH_THRESHOLD,
    DEFAULT_PIXEL_THRESHOLD,
    ChangeResult,
    FrameProcessor,
)
from .screen_capture import Frame, ScreenCapture, ScreenCaptureError
from .window_info import active_window_title

DEFAULT_DURATION_SEC = 15.0
DEFAULT_INTERVAL_SEC = 2.0


@dataclass(frozen=True)
class WatchEvent:
    """Один шаг наблюдения: кадр и решение по нему."""

    at: float
    changed: bool
    window_changed: bool
    result: ChangeResult | None
    width: int
    height: int
    window_title: str | None = None


class ScreenWatcher:
    """Держит последний кадр и сообщает, изменился ли следующий."""

    def __init__(self, capture: ScreenCapture, processor: FrameProcessor) -> None:
        self.capture = capture
        self.processor = processor
        self._previous: Frame | None = None

    def poll(self) -> WatchEvent:
        """Снимает кадр и сравнивает его с предыдущим."""
        frame = self.capture.grab()
        previous = self._previous
        result = None
        window_changed = False
        if previous is not None:
            result = self.processor.compare(previous.image, frame.image)
            window_changed = previous.window_title != frame.window_title
        self._previous = frame
        return WatchEvent(
            at=frame.captured_at,
            changed=bool((result and result.changed) or window_changed),
            window_changed=window_changed,
            result=result,
            width=frame.width,
            height=frame.height,
            window_title=frame.window_title,
        )

    def reset(self) -> None:
        """Забывает предыдущий кадр: следующий шаг снова будет стартовым."""
        self._previous = None


def _short_title(title: str, limit: int = 60) -> str:
    title = " ".join(title.split())
    if len(title) <= limit:
        return title
    return title[: limit - 1].rstrip() + "…"


def format_event(event: WatchEvent, elapsed: float) -> str:
    """Строка для консольной диагностики."""
    suffix = f"  окно: {_short_title(event.window_title)}" if event.window_title else ""
    if event.result is None:
        return f"+{elapsed:5.1f} с  старт      {event.width}×{event.height}{suffix}"
    result = event.result
    reasons = []
    if result.changed:
        reasons.append(result.reason)
    if event.window_changed:
        reasons.append("window")
    reason = "+".join(reasons) or "same"
    status = "изменение" if event.changed else "без изм."
    return (
        f"+{elapsed:5.1f} с  {status:9}  score={result.score:.3f}  "
        f"pixel={result.pixel_diff:.4f}  hash={result.hash_distance}  ({reason}){suffix}"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Проверка захвата экрана и порогов обнаружения изменений.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_DURATION_SEC,
        help=f"сколько секунд наблюдать (по умолчанию {DEFAULT_DURATION_SEC:g})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SEC,
        help=f"интервал между кадрами (по умолчанию {DEFAULT_INTERVAL_SEC:g})",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="ширина кадра после сжатия; 0 — без сжатия (по умолчанию 1280)",
    )
    parser.add_argument(
        "--pixel-threshold",
        type=float,
        default=DEFAULT_PIXEL_THRESHOLD,
        help=f"порог разницы яркости (по умолчанию {DEFAULT_PIXEL_THRESHOLD:g})",
    )
    parser.add_argument(
        "--hash-threshold",
        type=float,
        default=DEFAULT_HASH_THRESHOLD,
        help=f"порог различия структуры (по умолчанию {DEFAULT_HASH_THRESHOLD:g})",
    )
    args = parser.parse_args(argv)
    if args.seconds < 0:
        parser.error("--seconds не может быть отрицательным")
    if args.interval <= 0:
        parser.error("--interval должен быть больше нуля")
    return args


def main(argv: list[str] | None = None) -> int:
    """Запускает диагностический цикл и печатает события."""
    args = _parse_args(argv)
    capture = ScreenCapture(max_width=args.max_width, title_provider=active_window_title)
    watcher = ScreenWatcher(
        capture,
        FrameProcessor(
            pixel_threshold=args.pixel_threshold,
            hash_threshold=args.hash_threshold,
        ),
    )
    started = time.monotonic()
    deadline = started + args.seconds
    print(
        f"Наблюдение {args.seconds:g} с, интервал {args.interval:g} с, "
        f"пороги pixel={args.pixel_threshold:g} hash={args.hash_threshold:g}"
    )
    try:
        while True:
            event = watcher.poll()
            print(format_event(event, event.at - started))
            now = time.monotonic()
            if now >= deadline:
                break
            time.sleep(min(args.interval, deadline - now))
    except KeyboardInterrupt:
        print("Остановлено.")
    except ScreenCaptureError as exc:
        print(f"Ошибка захвата: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
