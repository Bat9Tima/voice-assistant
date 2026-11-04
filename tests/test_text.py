from __future__ import annotations

import unittest

from app.main import to_pynput
from app.text import normalize, strip_wake_word


class TextTest(unittest.TestCase):
    def test_normalize_keeps_domain(self) -> None:
        self.assertEqual(normalize("Открой Example.COM!"), "открой example.com")

    def test_normalize_removes_edge_dots(self) -> None:
        self.assertEqual(normalize("...привет..."), "привет")

    def test_strip_exact_wake_word(self) -> None:
        self.assertEqual(
            strip_wake_word("Окей ассистент, открой ютуб!", ["окей ассистент"]),
            "открой ютуб",
        )

    def test_strip_fuzzy_wake_word(self) -> None:
        self.assertEqual(
            strip_wake_word("окей асистент открой ютуб", ["окей ассистент"]),
            "открой ютуб",
        )

    def test_keeps_text_without_wake_word(self) -> None:
        self.assertEqual(strip_wake_word("включи музыку", ["окей ассистент"]), "включи музыку")

    def test_strips_partial_wake_word(self) -> None:
        self.assertEqual(
            strip_wake_word("окей открой ютуб", ["окей ассистент"]),
            "открой ютуб",
        )
        self.assertEqual(
            strip_wake_word("ассистент включи музыку", ["окей ассистент"]),
            "включи музыку",
        )

    def test_strips_merged_wake_phrase(self) -> None:
        aliases = ["хей луна", "эй луна"]
        self.assertEqual(strip_wake_word("хейлуна открой ютуб", aliases), "открой ютуб")
        self.assertEqual(strip_wake_word("эйлона громкость пять", aliases), "громкость пять")
        self.assertEqual(strip_wake_word("хайлу на следующий трек", aliases), "следующий трек")
        self.assertEqual(strip_wake_word("эйна громкость пять", aliases), "громкость пять")

    def test_strips_standalone_wake_filler(self) -> None:
        self.assertEqual(
            strip_wake_word("хэй открой ютуб", ["хей луна", "эй луна"]),
            "открой ютуб",
        )

    def test_hotkey_space(self) -> None:
        self.assertEqual(
            to_pynput("ctrl+alt+space"), "<ctrl>+<alt>+<space>"
        )

    def test_hotkey_letter(self) -> None:
        self.assertEqual(to_pynput("ctrl+shift+k"), "<ctrl>+<shift>+k")


if __name__ == "__main__":
    unittest.main()
