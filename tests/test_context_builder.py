from __future__ import annotations

import unittest

from app.context_builder import ContextBuilder, ContextPart


def part(name: str, size: int, **kwargs) -> ContextPart:
    return ContextPart(name, "x" * size, **kwargs)


class ContextBuilderTest(unittest.TestCase):
    def make(self, budget: int = 1000) -> ContextBuilder:
        # Один символ = один токен: так арифметику видно без округлений.
        return ContextBuilder(max_tokens=budget, chars_per_token=1.0)

    def test_system_and_messages_order(self) -> None:
        built = self.make().build(
            parts=[
                ContextPart("rules", "правила", required=True),
                ContextPart("profile", "профиль"),
            ],
            history=[{"role": "user", "content": "старое"}],
            user_text="новое",
        )

        self.assertEqual(built.system, "правила\n\nпрофиль")
        self.assertEqual(
            built.messages,
            [
                {"role": "user", "content": "старое"},
                {"role": "user", "content": "новое"},
            ],
        )
        self.assertEqual(built.stats.parts, {"rules": 7, "profile": 7})
        self.assertEqual(built.stats.history_messages, 1)
        self.assertEqual(built.stats.trimmed, [])

    def test_empty_parts_are_skipped(self) -> None:
        built = self.make().build(
            parts=[ContextPart("rules", "правила", required=True), ContextPart("profile", "  ")],
            history=[],
            user_text="привет",
            memory="   ",
        )

        self.assertEqual(built.system, "правила")
        self.assertEqual(set(built.stats.parts), {"rules"})

    def test_estimate_rounds_up(self) -> None:
        builder = ContextBuilder(chars_per_token=2.5)
        self.assertEqual(builder.estimate(""), 0)
        self.assertEqual(builder.estimate("аб"), 1)
        self.assertEqual(builder.estimate("абв"), 2)

    def test_memory_is_trimmed_first(self) -> None:
        built = self.make(300).build(
            parts=[part("rules", 100, required=True), part("profile", 100)],
            history=[],
            user_text="u" * 100,
            memory="m" * 100,
        )

        self.assertNotIn("m" * 10, built.system)
        self.assertIn("x" * 100, built.system)
        self.assertEqual(built.stats.trimmed, ["memory"])
        self.assertEqual(built.stats.tokens, 300)

    def test_persona_is_trimmed_before_profile(self) -> None:
        persona = "a" * 50 + "b" * 50
        built = self.make(330).build(
            parts=[
                part("rules", 100, required=True),
                part("profile", 100),
                ContextPart("persona", persona, keep_tail=True),
            ],
            history=[],
            user_text="u" * 50,
        )

        blocks = built.system.split("\n\n")
        self.assertEqual(blocks[1], "x" * 100)
        self.assertTrue(blocks[2].startswith("…"))
        self.assertTrue(blocks[2].endswith("b" * 50))
        self.assertEqual(len(blocks[2]), 80)
        self.assertEqual(built.stats.trimmed, ["persona"])
        self.assertEqual(built.stats.tokens, 330)

    def test_short_part_is_dropped_instead_of_stub(self) -> None:
        built = self.make(300).build(
            parts=[part("rules", 100, required=True), part("profile", 100), part("persona", 100)],
            history=[],
            user_text="u" * 50,
        )

        self.assertNotIn("persona", built.stats.parts)
        self.assertEqual(built.stats.trimmed, ["persona"])

    def test_history_is_dropped_oldest_first(self) -> None:
        built = self.make(250).build(
            parts=[part("rules", 100, required=True)],
            history=[
                {"role": "user", "content": "h" * 100},
                {"role": "assistant", "content": "i" * 100},
            ],
            user_text="u" * 50,
        )

        self.assertEqual(built.stats.history_dropped, 1)
        self.assertEqual(built.stats.history_messages, 1)
        self.assertEqual(built.messages[0]["content"], "i" * 100)
        self.assertEqual(built.stats.trimmed, ["history"])
        self.assertEqual(built.stats.tokens, 250)

    def test_required_overflow_is_kept(self) -> None:
        with self.assertLogs("app.context_builder", level="WARNING"):
            built = self.make(100).build(
                parts=[part("rules", 500, required=True)],
                history=[],
                user_text="u" * 100,
            )

        self.assertIn("x" * 500, built.system)
        self.assertEqual(built.messages[-1]["content"], "u" * 100)
        self.assertEqual(built.stats.tokens, 600)

    def test_tail_follows_memory_and_is_kept(self) -> None:
        built = self.make(350).build(
            parts=[part("rules", 100, required=True), part("profile", 100)],
            history=[],
            user_text="u" * 50,
            memory="m" * 100,
            tail=[ContextPart("character", "c" * 100, required=True)],
        )

        blocks = built.system.split("\n\n")
        self.assertEqual(blocks[-1], "c" * 100)
        self.assertNotIn("memory", built.stats.parts)
        self.assertIn("character", built.stats.parts)
        self.assertEqual(built.stats.trimmed, ["memory"])
        self.assertEqual(built.stats.tokens, 350)

    def test_zero_budget_disables_trimming(self) -> None:
        built = self.make(0).build(
            parts=[part("rules", 500, required=True)],
            history=[{"role": "user", "content": "h" * 100}],
            user_text="u" * 100,
            memory="m" * 100,
        )

        self.assertIn("m" * 100, built.system)
        self.assertEqual(built.stats.trimmed, [])
        self.assertEqual(built.stats.tokens, 800)


if __name__ == "__main__":
    unittest.main()
