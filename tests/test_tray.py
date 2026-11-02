from __future__ import annotations

import unittest
from unittest.mock import Mock, call

from app.tray import TrayUI


class TrayNotifyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = TrayUI(app=Mock())
        self.ui.icon = Mock()
        self.ui._ready.set()

    def test_new_notification_replaces_previous(self) -> None:
        self.ui.notify("Готово")
        self.assertEqual(
            self.ui.icon.method_calls[:2],
            [
                call.remove_notification(),
                call.notify("Готово", "Ассистент"),
            ],
        )

    def test_clear_removes_current_notification(self) -> None:
        self.ui.clear()
        self.ui.icon.remove_notification.assert_called_once_with()

    def test_empty_message_is_ignored(self) -> None:
        self.ui.notify("   ")
        self.ui.icon.remove_notification.assert_not_called()
        self.ui.icon.notify.assert_not_called()

    def test_clear_without_icon_does_nothing(self) -> None:
        self.ui.icon = None
        self.ui.clear()  # не должно бросить исключение


if __name__ == "__main__":
    unittest.main()
