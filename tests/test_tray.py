from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from unittest.mock import Mock, call, patch

import app.tray as tray_module
from app.tray import TrayUI

# Имена потоков, которые создаёт TrayUI: после тестов их быть не должно.
_TRAY_THREAD_NAMES = {"tray", "tray-watchdog"}


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


@unittest.skipUnless(sys.platform == "win32", "только Windows")
class TrayMessageFilterTest(unittest.TestCase):
    """Guard'ы над ctypes-функциями pystray (реальный модуль, без окон)."""

    def setUp(self) -> None:
        from pystray._util import win32

        self.win32 = win32
        self._original_filter = win32.ChangeWindowMessageFilterEx
        self._original_notify = win32.Shell_NotifyIcon

    def tearDown(self) -> None:
        # Обязательно возвращаем настоящие функции pystray.
        self.win32.ChangeWindowMessageFilterEx = self._original_filter
        self.win32.Shell_NotifyIcon = self._original_notify

    # ------------------------------------------------- ChangeWindowMessageFilterEx
    def test_message_filter_guard_delegates_and_is_idempotent(self) -> None:
        calls = []

        def fake_filter(hwnd, message, action, change_filter_struct):
            calls.append((hwnd, message, action, change_filter_struct))
            return True

        self.win32.ChangeWindowMessageFilterEx = fake_filter
        TrayUI._guard_message_filter()
        wrapper = self.win32.ChangeWindowMessageFilterEx
        self.assertIsNot(wrapper, fake_filter)
        self.assertEqual(wrapper.__name__, "_allow_message_filter")
        self.assertIs(getattr(wrapper, tray_module._GUARD_MARK, None), True)

        self.assertTrue(wrapper(1, 2, 3, None))
        self.assertEqual(calls, [(1, 2, 3, None)])

        # Повторный вызов не заворачивает обёртку в обёртку.
        TrayUI._guard_message_filter()
        self.assertIs(self.win32.ChangeWindowMessageFilterEx, wrapper)
        self.assertTrue(wrapper(4, 5, 6, None))
        self.assertEqual(calls, [(1, 2, 3, None), (4, 5, 6, None)])

    def test_message_filter_guard_swallows_access_denied(self) -> None:
        for error in (OSError(5, "Access is denied"), PermissionError(13, "Access denied")):
            with self.subTest(error=type(error).__name__):
                fake_filter = Mock(side_effect=error)
                self.win32.ChangeWindowMessageFilterEx = fake_filter
                TrayUI._guard_message_filter()
                wrapper = self.win32.ChangeWindowMessageFilterEx
                with self.assertLogs("app.tray", level="DEBUG"):
                    # WinError 5 не должен рвать создание окна трея.
                    self.assertTrue(wrapper(0, 0, 0, None))

    def test_message_filter_guard_without_pystray_is_noop(self) -> None:
        with patch.object(tray_module, "_pystray_win32", return_value=None):
            TrayUI._guard_message_filter()  # не должно бросить исключение
        self.win32.ChangeWindowMessageFilterEx = self._original_filter

    # ------------------------------------------------------- Shell_NotifyIcon
    def test_shell_notify_guard_wraps_and_is_idempotent(self) -> None:
        original = Mock(return_value=True)
        self.win32.Shell_NotifyIcon = original
        TrayUI._guard_shell_notify_icon()
        wrapper = self.win32.Shell_NotifyIcon
        self.assertIsNot(wrapper, original)
        self.assertEqual(wrapper.__name__, "_guarded_shell_notify_icon")
        self.assertIs(getattr(wrapper, tray_module._GUARD_MARK, None), True)

        self.assertTrue(wrapper(tray_module._NIM_ADD, None))
        original.assert_called_once_with(tray_module._NIM_ADD, None)

        TrayUI._guard_shell_notify_icon()
        self.assertIs(self.win32.Shell_NotifyIcon, wrapper)
        self.assertTrue(wrapper(tray_module._NIM_ADD, None))
        self.assertEqual(original.call_count, 2)

    def test_shell_notify_add_failure_raises_winerror(self) -> None:
        self.win32.Shell_NotifyIcon = Mock(return_value=False)
        TrayUI._guard_shell_notify_icon()
        with patch.object(tray_module, "_last_win_error", return_value=5):
            with self.assertRaises(OSError) as ctx:
                self.win32.Shell_NotifyIcon(tray_module._NIM_ADD, None)
        self.assertEqual(ctx.exception.errno, 5)
        self.assertIn("WinError 5", str(ctx.exception))
        self.assertIn("NIM_ADD", str(ctx.exception))

    def test_shell_notify_modify_failure_returns_false_without_raising(self) -> None:
        self.win32.Shell_NotifyIcon = Mock(return_value=False)
        TrayUI._guard_shell_notify_icon()
        with patch.object(tray_module, "_last_win_error", return_value=5):
            # NIM_MODIFY (1): сбой обновления не фатален.
            self.assertIs(self.win32.Shell_NotifyIcon(1, None), False)

    def test_shell_notify_success_returns_true(self) -> None:
        self.win32.Shell_NotifyIcon = Mock(return_value=True)
        TrayUI._guard_shell_notify_icon()
        self.assertIs(self.win32.Shell_NotifyIcon(tray_module._NIM_ADD, None), True)


class FakeMenu:
    SEPARATOR = object()

    def __init__(self, *items: object) -> None:
        self.items = items


class FakeMenuItem:
    def __init__(self, text: str, action=None, default: bool = False) -> None:
        self.text = text
        self.action = action
        self.default = default


class FakeIcon:
    """Заменитель ``pystray.Icon``: без окон и без настоящего pystray."""

    def __init__(self, factory: FakePystray, name: str, image, title, menu) -> None:
        self.factory = factory
        self.name = name
        self.image = image
        self.title = title
        self.menu = menu
        self.visible = False
        self.run_calls = 0
        self.stop_calls = 0
        self._stopped = threading.Event()

    def run(self, setup=None) -> None:
        self.run_calls += 1
        behavior = self.factory.behavior
        if behavior == "raise":
            raise RuntimeError(self.factory.error_text)
        if behavior == "nim_add":
            # pystray зовёт Shell_NotifyIcon(NIM_ADD); guard поднимет OSError.
            self.factory.win32.Shell_NotifyIcon(tray_module._NIM_ADD, None)
            return
        if setup is not None:
            setup(self)
        # Успешная иконка живёт, пока её не остановят.
        self._stopped.wait(timeout=5.0)

    def stop(self) -> None:
        self.stop_calls += 1
        self._stopped.set()


class FakePystray:
    """Модуль-заменитель ``pystray`` целиком."""

    def __init__(self, behavior: str = "block", error_text: str = "boom") -> None:
        self.behavior = behavior
        self.error_text = error_text
        self.icons: list[FakeIcon] = []
        self.win32: types.SimpleNamespace | None = None
        self.Menu = FakeMenu
        self.MenuItem = FakeMenuItem

    def Icon(self, name, image, title, menu) -> FakeIcon:
        icon = FakeIcon(self, name, image, title, menu)
        self.icons.append(icon)
        return icon


def _make_fake_win32(shell_notify=None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        ChangeWindowMessageFilterEx=Mock(return_value=True),
        Shell_NotifyIcon=shell_notify if shell_notify is not None else Mock(return_value=True),
    )


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


class TrayLifecycleTest(unittest.TestCase):
    """start/stop/status/watchdog на фейковом pystray."""

    def setUp(self) -> None:
        self.ui = TrayUI(app=Mock())
        self.ui._init_toasts = Mock()
        self.ui._show_toast = Mock(return_value=False)
        self.ui._startup_timeout = 1.0
        self.ui._thread_join_timeout = 1.0
        self.ui._watchdog_join_timeout = 1.0
        self._patchers: list[object] = []

    def tearDown(self) -> None:
        self.ui.stop()
        for patcher in reversed(self._patchers):
            patcher.stop()
        self._assert_no_tray_threads()

    def _assert_no_tray_threads(self) -> None:
        def alive() -> list[threading.Thread]:
            return [
                thread
                for thread in threading.enumerate()
                if thread.name in _TRAY_THREAD_NAMES and thread.is_alive()
            ]

        if not _wait_for(lambda: not alive(), timeout=1.0):
            self.fail(f"После теста остались живые потоки трея: {alive()}")

    def _install_fakes(
        self,
        *,
        behavior: str = "block",
        error_text: str = "boom",
        shell_notify=None,
    ) -> FakePystray:
        factory = FakePystray(behavior=behavior, error_text=error_text)
        factory.win32 = _make_fake_win32(shell_notify)
        patchers = [
            patch.dict(sys.modules, {"pystray": factory}),
            patch.object(tray_module, "_pystray_win32", return_value=factory.win32),
            patch.object(tray_module, "_last_win_error", return_value=5),
        ]
        for patcher in patchers:
            patcher.start()
            self._patchers.append(patcher)
        return factory

    # ------------------------------------------------------------- start
    def test_start_success_sets_status_and_logs(self) -> None:
        factory = self._install_fakes()

        with self.assertLogs("app.tray", level="INFO") as logs:
            self.ui.start()

        status = self.ui.status()
        self.assertTrue(status["running"])
        self.assertTrue(status["available"])
        self.assertIsNone(status["error"])
        self.assertEqual(status["restarts"], 0)
        self.assertEqual(len(factory.icons), 1)
        self.assertTrue(factory.icons[0].visible)
        self.assertIn("Трей запущен", "\n".join(logs.output))
        self.assertIsNotNone(self.ui._watchdog_thread)

    def test_start_icon_run_failure_is_reported(self) -> None:
        factory = self._install_fakes(behavior="raise", error_text="boom: run упал")

        with self.assertLogs("app.tray", level="INFO") as logs:
            self.ui.start()

        output = "\n".join(logs.output)
        self.assertNotIn("Трей запущен", output)
        status = self.ui.status()
        self.assertFalse(status["running"])
        self.assertFalse(status["available"])
        self.assertIsNotNone(status["error"])
        self.assertIn("boom: run упал", status["error"])
        self.assertEqual(len(factory.icons), 1)

    def test_start_nim_add_failure_does_not_report_success(self) -> None:
        factory = self._install_fakes(
            behavior="nim_add", shell_notify=Mock(return_value=False)
        )

        with self.assertLogs("app.tray", level="INFO") as logs:
            self.ui.start()

        output = "\n".join(logs.output)
        self.assertNotIn("Трей запущен", output)
        self.assertIn("NIM_ADD", output)
        status = self.ui.status()
        self.assertFalse(status["running"])
        self.assertFalse(status["available"])
        self.assertIsNotNone(status["error"])
        self.assertIn("NIM_ADD", status["error"])
        # Guard действительно установлен поверх fake-функции.
        self.assertIs(
            getattr(factory.win32.Shell_NotifyIcon, tray_module._GUARD_MARK, None), True
        )

    # ----------------------------------------------------------- watchdog
    def test_watchdog_restarts_dead_icon(self) -> None:
        factory = self._install_fakes()
        self.ui._watchdog_interval = 0.02
        self.ui._restart_initial_delay = 0.01
        self.ui._restart_max_delay = 0.01

        with self.assertLogs("app.tray", level="INFO"):
            self.ui.start()
        self.assertEqual(len(factory.icons), 1)

        # Убиваем фейковый трей-поток: icon.run() возвращается.
        factory.icons[0].stop()

        self.assertTrue(
            _wait_for(
                lambda: self.ui.status()["restarts"] >= 1 and len(factory.icons) >= 2,
                timeout=2.0,
            ),
            "watchdog не пересоздал иконку",
        )
        self.assertGreaterEqual(self.ui.status()["restarts"], 1)
        self.assertEqual(len(factory.icons), 2)
        self.assertTrue(
            _wait_for(lambda: factory.icons[1].visible, timeout=1.0),
            "новая иконка не была показана",
        )

    def test_stop_prevents_restarts(self) -> None:
        factory = self._install_fakes()
        self.ui._watchdog_interval = 0.02
        self.ui._restart_initial_delay = 0.01

        with self.assertLogs("app.tray", level="INFO"):
            self.ui.start()
        icon = factory.icons[0]
        self.assertTrue(icon.visible)

        self.ui.stop()

        self.assertEqual(icon.stop_calls, 1)
        restarts = self.ui.status()["restarts"]
        # Даём watchdog'у несколько интервалов: новых перезапусков быть не должно.
        time.sleep(0.15)
        self.assertEqual(self.ui.status()["restarts"], restarts)
        self.assertFalse(self.ui.status()["running"])
        self.assertEqual(len(factory.icons), 1)
        self.assertIsNone(self.ui._watchdog_thread)

    def test_unavailable_notification_sent_only_once(self) -> None:
        self._install_fakes(behavior="raise", error_text="нет трея")
        self.ui._watchdog_interval = 0.02
        self.ui._restart_initial_delay = 0.01
        self.ui._restart_max_delay = 0.02
        self.ui._show_toast = Mock(return_value=True)

        with self.assertLogs("app.tray", level="INFO"):
            self.ui.start()

        self.assertTrue(
            _wait_for(lambda: self.ui.status()["restarts"] >= 2, timeout=2.0),
            "watchdog не сделал повторных попыток",
        )
        # Сколько бы раз трей ни падал, уведомление показывается максимум один раз.
        self.assertEqual(self.ui._show_toast.call_count, 1)


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
