from __future__ import annotations

import unittest
from unittest.mock import patch

from app.window_info import active_window_title, normalize_title


class NormalizeTitleTest(unittest.TestCase):
    def test_collapses_spaces(self) -> None:
        self.assertEqual(normalize_title("  Chrome   —  GitHub  "), "Chrome — GitHub")

    def test_empty_becomes_none(self) -> None:
        self.assertIsNone(normalize_title(""))
        self.assertIsNone(normalize_title("   "))
        self.assertIsNone(normalize_title(None))


class ActiveWindowTitleTest(unittest.TestCase):
    def test_reads_and_normalizes_title(self) -> None:
        with patch(
            "app.window_info._read_foreground_title",
            return_value="  Chrome   —  GitHub ",
        ):
            self.assertEqual(active_window_title(), "Chrome — GitHub")

    def test_empty_title_becomes_none(self) -> None:
        with patch("app.window_info._read_foreground_title", return_value="   "):
            self.assertIsNone(active_window_title())

    def test_read_error_returns_none(self) -> None:
        with patch(
            "app.window_info._read_foreground_title",
            side_effect=OSError("нет доступа"),
        ):
            self.assertIsNone(active_window_title())


if __name__ == "__main__":
    unittest.main()
