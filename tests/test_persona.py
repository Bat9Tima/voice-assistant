from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from app.persona import PERSONA_HEADER, PersonaMemory, parse_facts


class PersonaMemoryTest(unittest.TestCase):
    def make(self, text: str | None = None, **kwargs) -> PersonaMemory:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "luna.txt"
        if text is not None:
            path.write_text(text, encoding="utf-8")
        return PersonaMemory(path, **kwargs)

    def test_missing_file_is_empty(self) -> None:
        memory = self.make()
        self.assertEqual(memory.facts(), [])
        self.assertEqual(memory.text(), "")
        self.assertEqual(memory.block(), "")

    def test_facts_are_read_without_bullets(self) -> None:
        memory = self.make("- Мне 23 года.\n- Живу в Москве.\n")
        self.assertEqual(memory.facts(), ["Мне 23 года.", "Живу в Москве."])

    def test_merge_adds_new_facts(self) -> None:
        memory = self.make("Мне 23 года.\n")
        self.assertEqual(memory.merge(["Люблю кофе.", "Живу в Москве."]), 2)
        self.assertEqual(
            memory.facts(),
            ["Мне 23 года.", "Люблю кофе.", "Живу в Москве."],
        )

    def test_merge_skips_exact_duplicates(self) -> None:
        memory = self.make("Мне 23 года.\n")
        self.assertEqual(memory.merge(["мне 23 года", "- Мне 23 года."]), 0)
        self.assertEqual(memory.facts(), ["Мне 23 года."])

    def test_merge_skips_empty_and_negative_answer(self) -> None:
        memory = self.make()
        self.assertEqual(memory.merge(["  ", "НЕТ", "нет фактов."]), 0)
        self.assertEqual(memory.facts(), [])

    def test_merge_without_path_does_nothing(self) -> None:
        memory = PersonaMemory(None)
        self.assertEqual(memory.merge(["Люблю кофе."]), 0)
        self.assertEqual(memory.facts(), [])

    def test_block_has_header_and_facts(self) -> None:
        memory = self.make("Люблю кофе.\n")
        block = memory.block()
        self.assertTrue(block.startswith(PERSONA_HEADER))
        self.assertIn("- Люблю кофе.", block)

    def test_text_keeps_last_facts_when_limited(self) -> None:
        memory = self.make("Первое.\nВторое.\nТретье.\n", limit=20)
        text = memory.text()
        self.assertNotIn("Первое", text)
        self.assertIn("Третье", text)
        self.assertLessEqual(len(text), 20)

    def test_write_leaves_no_temp_file(self) -> None:
        memory = self.make()
        memory.merge(["Люблю кофе."])
        self.assertFalse(memory.path.with_name(memory.path.name + ".tmp").exists())

    def test_file_is_utf8(self) -> None:
        memory = self.make()
        memory.merge(["Люблю кофе и книги."])
        self.assertIn("Люблю кофе", memory.path.read_text(encoding="utf-8"))

    def test_parallel_merges_keep_all_facts(self) -> None:
        memory = self.make()

        def add(start: int) -> None:
            for index in range(20):
                memory.merge([f"Факт {start}-{index}."])

        threads = [threading.Thread(target=add, args=(number,)) for number in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(memory.facts()), 100)


class ParseFactsTest(unittest.TestCase):
    def test_strips_bullets_and_numbers(self) -> None:
        answer = "- Мне 23 года.\n2. Живу в Москве.\n• Люблю кофе.\n"
        self.assertEqual(
            parse_facts(answer),
            ["Мне 23 года.", "Живу в Москве.", "Люблю кофе."],
        )

    def test_skips_negative_answer(self) -> None:
        self.assertEqual(parse_facts("НЕТ"), [])
        self.assertEqual(parse_facts("нет фактов."), [])

    def test_skips_heading(self) -> None:
        self.assertEqual(parse_facts("Факты о Луне:\n- Люблю кофе."), ["Люблю кофе."])

    def test_empty_answer(self) -> None:
        self.assertEqual(parse_facts(""), [])
        self.assertEqual(parse_facts("   \n\n"), [])


if __name__ == "__main__":
    unittest.main()
