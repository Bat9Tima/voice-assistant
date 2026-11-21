from __future__ import annotations

import unittest

from app.durability import EVENT, STATE, TRAIT, classify_scope


class ScopeTest(unittest.TestCase):
    def test_preference_is_trait(self) -> None:
        self.assertEqual(classify_scope("Пользователь любит кофе"), TRAIT)
        self.assertEqual(classify_scope("Пользователь сегодня любит кофе"), TRAIT)
        self.assertEqual(classify_scope("Пользователь работает программистом"), TRAIT)

    def test_past_action_is_event(self) -> None:
        self.assertEqual(classify_scope("Пользователь вчера сдал экзамен"), EVENT)
        self.assertEqual(classify_scope("Пользователь ходил в кино"), EVENT)
        self.assertEqual(classify_scope("Пользователь был на встрече"), EVENT)

    def test_current_situation_is_state(self) -> None:
        self.assertEqual(classify_scope("Сосед ужинает"), STATE)
        self.assertEqual(classify_scope("Пользователь сегодня устал"), STATE)
        self.assertEqual(classify_scope("Пользователь болеет"), STATE)


if __name__ == "__main__":
    unittest.main()
