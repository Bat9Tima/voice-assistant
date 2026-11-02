from __future__ import annotations

import unittest
from unittest.mock import Mock, call

from app.tray import TrayUI


class FakeToast:
    def __init__(self, text_fields, **kwargs) -> None:
        self.text_fields = text_fields
        self.group = kwargs.get("group")
        self.tag = ""


class FakeToaster:
    def __init__(self) -> None:
        self.cleared = 0
        self.shown: list[FakeToast] = []

    def clear_toasts(self) -> None:
        self.cleared += 1

    def show_toast(self, toast: FakeToast) -> None:
        self.shown.append(toast)


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

    def test_windows_toast_replaces_previous(self) -> None:
        toaster = FakeToaster()
        self.ui._toaster = toaster
        self.ui._toast_class = FakeToast
        self.ui.notify("Готово", "Успех")
        self.assertEqual(toaster.cleared, 1)
        self.assertEqual(len(toaster.shown), 1)
        toast = toaster.shown[0]
        self.assertEqual(toast.text_fields, ["Успех", "Готово"])
        self.assertEqual(toast.group, "voice-assistant")
        self.assertEqual(toast.tag, "assistant")
        self.ui.icon.notify.assert_not_called()

    def test_clear_uses_toast_history(self) -> None:
        toaster = FakeToaster()
        self.ui._toaster = toaster
        self.ui._toast_class = FakeToast
        self.ui.clear()
        self.assertEqual(toaster.cleared, 1)
        self.ui.icon.remove_notification.assert_not_called()


if __name__ == "__main__":
    unittest.main()
