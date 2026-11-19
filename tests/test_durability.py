from __future__ import annotations

import unittest

from app.durability import PERMANENT, TEMPORARY, classify_durability


class DurabilityTest(unittest.TestCase):
    def test_health_is_temporary(self) -> None:
        self.assertEqual(classify_durability("Пользователь болеет"), TEMPORARY)
        self.assertEqual(classify_durability("Пользователь уже выздоровел"), TEMPORARY)
        self.assertEqual(classify_durability("Пользователю очень плохо"), TEMPORARY)
        self.assertEqual(classify_durability("Пользователь сегодня устал"), TEMPORARY)

    def test_preference_is_permanent(self) -> None:
        self.assertEqual(classify_durability("Пользователь любит кофе"), PERMANENT)
        self.assertEqual(classify_durability("Пользователь не любит кофе"), PERMANENT)
        self.assertEqual(classify_durability("Пользователь увлекается футболом"), PERMANENT)

    def test_preference_wins_over_short_period(self) -> None:
        self.assertEqual(classify_durability("Пользователь сегодня любит кофе"), PERMANENT)

    def test_current_situation_is_temporary(self) -> None:
        self.assertEqual(classify_durability("Сосед ужинает"), TEMPORARY)
        self.assertEqual(
            classify_durability("Пользователь ещё не лёг спать и сидит дома"),
            TEMPORARY,
        )
        self.assertEqual(
            classify_durability("Рядом с пользователем включён звук"),
            TEMPORARY,
        )
        self.assertEqual(classify_durability("Пользователь расстроен"), TEMPORARY)

    def test_relationship_detail_stays_permanent(self) -> None:
        self.assertEqual(classify_durability("Соседа зовут Матвей, а не Дмитрий"), PERMANENT)
        self.assertEqual(classify_durability("У пользователя есть сосед"), PERMANENT)

    def test_unclear_fact_defaults_to_permanent(self) -> None:
        self.assertEqual(classify_durability("Пользователь работает программистом"), PERMANENT)
        self.assertEqual(classify_durability(""), PERMANENT)


if __name__ == "__main__":
    unittest.main()
