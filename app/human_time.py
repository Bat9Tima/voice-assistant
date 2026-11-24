"""Человеческие формулировки времени и числительных для голоса.

Один набор правил склонения, чтобы ответы «5 минут», «2 раза» и «3 события»
не расходились по разным модулям.
"""

from __future__ import annotations


def plural(count: int, one: str, few: str, many: str) -> str:
    """Русская форма слова после числа."""
    number = abs(int(count))
    if number % 10 == 1 and number % 100 != 11:
        return one
    if 2 <= number % 10 <= 4 and not 12 <= number % 100 <= 14:
        return few
    return many


def format_duration(seconds: float) -> str:
    """«5 минут 3 секунды» — без таблиц склонений в вызывающем коде."""
    total = int(max(0.0, seconds))
    minutes, rest = divmod(total, 60)
    parts: list[str] = []
    if minutes:
        parts.append(f"{minutes} {plural(minutes, 'минуту', 'минуты', 'минут')}")
    if rest or not parts:
        parts.append(f"{rest} {plural(rest, 'секунду', 'секунды', 'секунд')}")
    return " ".join(parts)


__all__ = ["format_duration", "plural"]
