from __future__ import annotations

import logging
import os
import queue
import random
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from .config import STATE_DIR
from .text import normalize

log = logging.getLogger(__name__)

# Клик по дорожке SoundCloud смещает значение примерно на 4%: компенсируем,
# чтобы попадать в цель с первого раза.
SOUNDCLOUD_SLIDER_OFFSET = 0.04


class BrowserError(RuntimeError):
    pass


class BrowserController(threading.Thread):
    """Единый поток, владеющий Playwright и подключением к Chrome по CDP.

    Все операции выполняются через очередь, поэтому вызывать методы можно
    из любого потока (основной цикл, HTTP-сервер, tray).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(daemon=True, name="browser")
        self.cfg = cfg["browser"]
        self.port = int(self.cfg.get("port", 9222))
        self.profile_dir = Path(self.cfg["profile_dir"])
        self._requests: queue.Queue[tuple[str, dict, Future]] = queue.Queue()
        self._ready = threading.Event()
        self._stop_event = threading.Event()
        self._launched = False
        self._pw = None
        self._browser = None
        self._context = None
        self._assistant_page = None
        self.last_error: str | None = None

    # ------------------------------------------------------------------ life
    def start(self) -> BrowserController:
        if not self._launched:
            self._launched = True
            super().start()
        if not self._ready.wait(timeout=20):
            raise BrowserError("Поток браузера не запустился за 20 секунд")
        if self.last_error:
            raise BrowserError(self.last_error)
        return self

    def stop(self) -> None:
        if not self._launched or not self.is_alive():
            return
        self._stop_event.set()
        future: Future = Future()
        try:
            self._requests.put(("__stop__", {}, future))
            future.result(timeout=3)
        except Exception:
            pass
        self.join(timeout=5)

    def run(self) -> None:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as pw:
                self._pw = pw
                self._ready.set()
                while not self._stop_event.is_set():
                    try:
                        action, args, future = self._requests.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if action == "__stop__":
                        future.set_result({"ok": True})
                        break
                    try:
                        data = self._dispatch(action, args)
                        future.set_result({"ok": True, "data": data})
                    except Exception as exc:  # noqa: BLE001
                        log.exception("Ошибка браузера в действии %s", action)
                        future.set_result({"ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"Не удалось запустить Playwright: {exc}"
            log.exception("Поток браузера упал")
        finally:
            self._ready.set()

    def call(self, action: str, timeout: float = 90, **args: Any) -> dict[str, Any]:
        if not self._launched:
            self.start()
        if not self.is_alive():
            raise BrowserError(self.last_error or "Поток браузера остановлен")
        future: Future = Future()
        self._requests.put((action, args, future))
        try:
            return future.result(timeout=timeout)
        except TimeoutError as exc:
            raise BrowserError(f"Браузер не ответил за {timeout:.0f} с") from exc

    # ------------------------------------------------------------- connection
    def _cdp_alive(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/json/version", timeout=1.5
            ) as response:
                return response.status == 200
        except Exception:
            return False

    def _find_browser(self) -> str:
        candidates = [
            self.cfg.get("chrome_path", ""),
            os.path.join(
                os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                "Google",
                "Chrome",
                "Application",
                "chrome.exe",
            ),
            os.path.join(
                os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
                "Google",
                "Chrome",
                "Application",
                "chrome.exe",
            ),
            os.path.join(
                os.environ.get("LOCALAPPDATA", ""),
                "Google",
                "Chrome",
                "Application",
                "chrome.exe",
            ),
            os.path.join(
                os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
                "Microsoft",
                "Edge",
                "Application",
                "msedge.exe",
            ),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        found = shutil.which("chrome") or shutil.which("msedge")
        if found:
            return found
        raise BrowserError("Chrome не найден. Укажите путь в config.json → browser.chrome_path")

    def _launch_chrome(self) -> None:
        browser = self._find_browser()
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        args = [
            browser,
            f"--remote-debugging-port={self.port}",
            f"--user-data-dir={self.profile_dir}",
            "--profile-directory=Default",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-search-engine-choice-screen",
            "--disable-session-crashed-bubble",
            "--hide-crash-restore-bubble",
            "--disable-features=Translate,ProfilePicker",
            "--autoplay-policy=no-user-gesture-required",
            str(self.cfg.get("start_url") or "https://www.google.com"),
        ]
        log.info("Запускаю браузер с CDP на порту %s: %s", self.port, browser)
        try:
            subprocess.Popen(args, close_fds=True)
        except OSError as exc:
            raise BrowserError(f"Не удалось запустить Chrome: {exc}") from exc

        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            if self._cdp_alive():
                log.info("CDP доступен на порту %s", self.port)
                return
            time.sleep(0.4)
        raise BrowserError(
            "Chrome запущен, но CDP-порт не открылся. "
            "Закройте окно Chrome, созданное ассистентом, и повторите команду."
        )

    def _ensure_connection(self) -> None:
        if self._browser is not None and self._browser.is_connected():
            return
        if not self._cdp_alive():
            self._launch_chrome()
        endpoint = f"http://127.0.0.1:{self.port}"
        log.info("Подключаюсь к Chrome по CDP: %s", endpoint)
        self._browser = self._pw.chromium.connect_over_cdp(endpoint, timeout=30000)
        contexts = self._browser.contexts
        self._context = contexts[0] if contexts else self._browser.new_context()
        self._assistant_page = None

    def _pages(self) -> list:
        self._ensure_connection()
        assert self._context is not None
        return [page for page in self._context.pages if not page.is_closed()]

    def _assistant_tab(self) -> Any:
        self._ensure_connection()
        if self._assistant_page is not None and not self._assistant_page.is_closed():
            return self._assistant_page
        pages = self._pages()
        self._assistant_page = pages[-1] if pages else self._context.new_page()
        return self._assistant_page

    def _soundcloud_page(self) -> Any | None:
        """Вкладка SoundCloud с собственным плеером (у него нет тега audio)."""
        try:
            pages = self._pages()
        except Exception:
            return None
        for page in reversed(pages):
            try:
                if (
                    "soundcloud.com" in page.url
                    and page.locator(".playControls__play").count()
                ):
                    return page
            except Exception:
                continue
        return None

    def _media_page(self) -> Any:
        """Ищет вкладку, где реально есть видео/аудио, иначе — вкладку ассистента."""
        for page in reversed(self._pages()):
            try:
                has_media = page.evaluate(
                    "() => [...document.querySelectorAll('video, audio')]"
                    ".some(m => m.readyState > 0)"
                )
                if has_media:
                    return page
            except Exception:
                continue
        return self._assistant_tab()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _clean_url(url: str) -> str:
        url = (url or "").strip()
        if not url:
            raise BrowserError("Пустой URL")
        lowered = url.lower()
        for scheme in ("javascript:", "data:", "file:", "chrome:"):
            if lowered.startswith(scheme):
                raise BrowserError(f"Схема {scheme} запрещена")
        if "://" not in url:
            url = "https://" + url
        return url

    @staticmethod
    def _safe_title(page: Any) -> str:
        try:
            return page.title()
        except Exception:
            return ""

    @staticmethod
    def _dismiss_consent(page: Any) -> None:
        selectors = [
            "#onetrust-accept-btn-handler",
            "button:has-text('Accept all')",
            "button:has-text('I accept')",
            "button:has-text('Принять все')",
            "button:has-text('Принять')",
            "button:has-text('I agree')",
            "button:has-text('Согласен')",
            "button:has-text('Reject all')",
            "button:has-text('Отклонить все')",
            "tp-yt-paper-button:has-text('Accept all')",
            "tp-yt-paper-button:has-text('Принять все')",
        ]
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                if locator.count() and locator.is_visible():
                    locator.click(timeout=2000)
                    page.wait_for_timeout(500)
                    return
            except Exception:
                continue

    @staticmethod
    def _close_soundcloud_auth(page: Any) -> None:
        """Закрывает окно входа SoundCloud, если оно перекрывает плеер."""
        for selector in (
            ".modal.auth-modal .modal__closeButton",
            ".webAuthContainerWrapper .modal__closeButton",
        ):
            try:
                locator = page.locator(selector).first
                if locator.count() and locator.is_visible():
                    locator.click(timeout=2000)
                    page.wait_for_timeout(400)
                    return
            except Exception:
                continue

    @staticmethod
    def _ensure_playing(page: Any) -> None:
        try:
            page.wait_for_timeout(1200)
            page.evaluate(
                """() => {
                    const video = document.querySelector('video');
                    if (video) {
                        video.muted = false;
                        video.play().catch(() => {});
                    }
                }"""
            )
        except Exception:
            pass

    def _tab_info(self, index: int, page: Any) -> dict[str, Any]:
        return {
            "index": index,
            "url": page.url,
            "title": self._safe_title(page),
        }

    def _tabs(self) -> list[dict[str, Any]]:
        return [self._tab_info(i + 1, p) for i, p in enumerate(self._pages())]

    # ---------------------------------------------------------------- actions
    def _dispatch(self, action: str, args: dict[str, Any]) -> Any:
        action = (action or "").strip().lower()
        if action == "ping":
            return {"pong": True, "cdp": self._cdp_alive()}
        if action == "goto":
            return self._act_goto(args)
        if action == "open":
            return self._act_goto({"url": args.get("url")})
        if action == "youtube":
            return self._act_youtube(args)
        if action == "soundcloud":
            return self._act_soundcloud(args)
        if action == "soundcloud_volume":
            return self._act_soundcloud_volume(args)
        if action == "soundcloud_likes":
            return self._act_soundcloud_likes(args)
        if action == "google":
            return self._act_google(args)
        if action == "search_result":
            return self._act_search_result(args)
        if action == "newtab":
            return self._act_new_tab(args)
        if action == "closetab":
            return self._act_close_tab()
        if action == "tabs":
            return {"tabs": self._tabs()}
        if action == "tab":
            return self._act_switch_tab(args)
        if action in {"back", "forward", "reload"}:
            return self._act_nav(action)
        if action in {"play", "pause", "toggle", "play_pause"}:
            return self._act_play(action)
        if action in {"next", "prev"}:
            return self._act_media_skip(action)
        if action == "press":
            page = self._assistant_tab()
            page.keyboard.press(str(args.get("key", "")))
            return {"pressed": args.get("key")}
        if action == "click":
            page = self._assistant_tab()
            page.click(str(args["selector"]), timeout=10000)
            return {"clicked": args["selector"], "url": page.url}
        if action == "fill":
            page = self._assistant_tab()
            page.fill(str(args["selector"]), str(args.get("text", "")), timeout=10000)
            return {"filled": args["selector"]}
        if action == "type":
            page = self._assistant_tab()
            page.keyboard.type(str(args.get("text", "")), delay=15)
            return {"typed": True}
        if action == "eval":
            page = self._assistant_tab()
            return {"result": page.evaluate(str(args["script"]))}
        if action == "text":
            page = self._assistant_tab()
            return {"text": page.inner_text("body", timeout=5000)[:4000]}
        if action == "scroll":
            page = self._assistant_tab()
            delta = -800 if str(args.get("direction", "down")).lower() == "up" else 800
            page.mouse.wheel(0, delta)
            return {"scrolled": delta}
        if action == "wait":
            page = self._assistant_tab()
            page.wait_for_timeout(int(args.get("ms", 1000)))
            return {"waited": args.get("ms", 1000)}
        if action == "screenshot":
            return self._act_screenshot(args)
        if action == "status":
            page = self._assistant_tab()
            return {
                "url": page.url,
                "title": self._safe_title(page),
                "tabs": self._tabs(),
            }
        raise BrowserError(f"Неизвестное действие: {action}")

    def _act_goto(self, args: dict[str, Any]) -> dict[str, Any]:
        url = self._clean_url(str(args.get("url", "")))
        page = self._assistant_tab()
        page.bring_to_front()
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
        self._dismiss_consent(page)
        return {"url": page.url, "title": self._safe_title(page)}

    def _act_youtube(self, args: dict[str, Any]) -> dict[str, Any]:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        query = str(args.get("query", "")).strip()
        if not query:
            raise BrowserError("Пустой поисковый запрос для YouTube")
        try:
            index = max(1, int(args.get("index", 1) or 1))
        except (TypeError, ValueError):
            index = 1
        page = self._assistant_tab()
        page.bring_to_front()
        search_url = "https://www.youtube.com/results?search_query=" + quote_plus(query)
        page.goto(search_url, wait_until="domcontentloaded", timeout=45000)
        self._dismiss_consent(page)

        href = ""
        available = 0
        selectors = [
            "ytd-video-renderer a#video-title",
            "a#video-title",
            "a[href*='/watch?v=']",
        ]
        last_error: Exception | None = None
        for selector in selectors:
            try:
                page.wait_for_selector(selector, timeout=15000)
                hrefs = page.eval_on_selector_all(
                    selector,
                    "els => els.map(el => el.href).filter(Boolean)",
                ) or []
                available = max(available, len(hrefs))
                if len(hrefs) >= index:
                    href = hrefs[index - 1]
                    break
            except PlaywrightTimeout as exc:
                last_error = exc
                continue
        if not href:
            if available:
                raise BrowserError(f"В результатах YouTube только {available} видео")
            raise BrowserError("YouTube не показал результаты поиска") from last_error
        page.goto(href, wait_until="domcontentloaded", timeout=45000)
        self._ensure_playing(page)
        return {
            "url": page.url,
            "title": self._safe_title(page),
            "query": query,
            "index": index,
        }

    def _act_soundcloud(self, args: dict[str, Any]) -> dict[str, Any]:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        query = str(args.get("query", "")).strip()
        if not query:
            raise BrowserError("Пустой поисковый запрос для SoundCloud")
        try:
            index = max(1, int(args.get("index", 1) or 1))
        except (TypeError, ValueError):
            index = 1
        page = self._assistant_tab()
        page.bring_to_front()
        search_url = "https://soundcloud.com/search/sounds?q=" + quote_plus(query)
        page.goto(search_url, wait_until="domcontentloaded", timeout=45000)
        self._dismiss_consent(page)
        self._close_soundcloud_auth(page)

        play_selector = "a.sc-button-play.playButton[title='Play']:not(.sc-button-disabled)"
        try:
            page.wait_for_selector(play_selector, timeout=15000)
        except PlaywrightTimeout as exc:
            raise BrowserError("SoundCloud не показал результаты поиска") from exc

        buttons = page.locator(play_selector)
        count = buttons.count()
        if count < index:
            raise BrowserError(f"В поиске SoundCloud только {count} треков")
        button = buttons.nth(index - 1)
        try:
            button.click(timeout=5000)
        except Exception:
            # Иногда SoundCloud показывает окно входа уже после клика.
            self._close_soundcloud_auth(page)
            button.click(timeout=10000)

        try:
            page.wait_for_function(
                """() => {
                    const button = document.querySelector('.playControls__play');
                    return button && /pause/i.test(button.getAttribute('title') || '');
                }""",
                timeout=10000,
            )
        except PlaywrightTimeout as exc:
            auth_visible = False
            try:
                auth_visible = page.locator(".modal.auth-modal").first.is_visible()
            except Exception:
                pass
            if auth_visible:
                raise BrowserError(
                    "SoundCloud просит войти. Войдите в Chrome ассистента один раз и повторите."
                ) from exc
            raise BrowserError("SoundCloud не начал воспроизведение") from exc

        page.wait_for_timeout(500)
        return {
            "url": page.url,
            "title": self._safe_title(page),
            "query": query,
            "index": index,
        }

    # ------------------------------------------------------- soundcloud volume
    @staticmethod
    def _soundcloud_volume_value(page: Any) -> float:
        try:
            raw = page.eval_on_selector(
                ".volume__sliderWrapper",
                "el => el.getAttribute('aria-valuenow')",
            )
            return max(0.0, min(1.0, float(raw)))
        except Exception:
            return 0.0

    def _hover_soundcloud_volume(self, page: Any) -> bool:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        wrapper = page.locator(".volume__sliderWrapper").first
        icon = page.locator(".volume__speakerIcon").first
        if not icon.count():
            return False
        for _ in range(3):
            page.mouse.move(2, 2)
            page.wait_for_timeout(250)
            try:
                icon.hover(timeout=5000)
                page.wait_for_function(
                    "() => { const el = document.querySelector('.volume__sliderWrapper');"
                    " return !!el && el.getBoundingClientRect().height > 100; }",
                    timeout=2000,
                )
            except PlaywrightTimeout:
                continue
            box = wrapper.bounding_box()
            if box and box["height"] > 100:
                return True
        return False

    def _set_soundcloud_volume(self, page: Any, target: float) -> tuple[float, int]:
        target = max(0.0, min(1.0, target))
        actual = self._soundcloud_volume_value(page)
        if abs(actual - target) <= 0.01:
            return actual, 0

        correction = 0.0
        attempts = 0
        # Обычно хватает одного клика. Второй — если попали мимо, третий —
        # только на редкий случай, когда клик ушёл в самое начало шкалы.
        while attempts < 3:
            if attempts == 2 and not (actual <= 0.001 and target > 0.05):
                break
            if not self._hover_soundcloud_volume(page):
                raise BrowserError("Не удалось открыть регулятор громкости SoundCloud")

            background = page.locator(".volume__sliderBackground").first.bounding_box()
            wrapper = page.locator(".volume__sliderWrapper").first.bounding_box()
            if not background or not wrapper or wrapper["height"] <= 100:
                raise BrowserError("Не удалось открыть регулятор громкости SoundCloud")

            x = background["x"] + background["width"] / 2
            if target >= 0.995:
                y = background["y"] + 1
            else:
                y = (
                    background["y"]
                    + (1.0 - target + SOUNDCLOUD_SLIDER_OFFSET) * background["height"]
                    + correction
                )
            y = max(wrapper["y"] + 6, min(wrapper["y"] + wrapper["height"] - 6, y))
            page.mouse.click(x, y)
            attempts += 1
            page.wait_for_timeout(300)
            actual = self._soundcloud_volume_value(page)
            if abs(actual - target) <= 0.02:
                break
            correction += (actual - target) * background["height"]

        page.mouse.move(2, 2)
        page.wait_for_timeout(150)
        log.debug(
            "Громкость SoundCloud: цель %.2f, факт %.2f, кликов %d",
            target,
            actual,
            attempts,
        )
        return actual, attempts

    def _act_soundcloud_volume(self, args: dict[str, Any]) -> dict[str, Any]:
        page = self._soundcloud_page()
        if page is None:
            raise BrowserError("SoundCloud не открыт. Скажите «открой саундклауд».")
        page.bring_to_front()
        if not page.locator(".volume__sliderWrapper").count():
            raise BrowserError("На странице SoundCloud не найден регулятор громкости")

        mode = str(args.get("mode", "up")).strip().lower()
        current = self._soundcloud_volume_value(page)
        if mode in {"mute", "unmute"}:
            want_mute = mode == "mute"
            muted = current <= 0.001
            clicked = False
            if muted != want_mute:
                page.locator(".volume__speakerIcon").first.click(timeout=5000)
                clicked = True
                page.wait_for_timeout(500)
                current = self._soundcloud_volume_value(page)
            return {
                "mode": mode,
                "volume": round(current * 100),
                "muted": current <= 0.001,
                "attempts": int(clicked),
                "url": page.url,
            }

        if mode == "set":
            try:
                target = float(args.get("value", 0)) / 100.0
            except (TypeError, ValueError) as exc:
                raise BrowserError("Неверное значение громкости SoundCloud") from exc
        elif mode in {"up", "down"}:
            try:
                step = float(args.get("value", 10)) / 100.0
            except (TypeError, ValueError) as exc:
                raise BrowserError("Неверный шаг громкости SoundCloud") from exc
            target = current + step if mode == "up" else current - step
        else:
            raise BrowserError(f"Неизвестный режим громкости SoundCloud: {mode}")

        target = max(0.0, min(1.0, target))
        actual, attempts = self._set_soundcloud_volume(page, target)
        return {
            "mode": mode,
            "volume": round(actual * 100),
            "muted": actual <= 0.001,
            "attempts": attempts,
            "url": page.url,
        }

    # -------------------------------------------------------- soundcloud likes
    @staticmethod
    def _soundcloud_like_items(page: Any) -> list[dict[str, Any]]:
        return page.evaluate(
            """() => [...document.querySelectorAll('li.badgeList__item')].map((li, index) => {
                const link = li.querySelector(
                    'a.playableTile__mainHeading, a.playableTile__artworkLink'
                );
                const artwork = li.querySelector('.sc-artwork[aria-label]');
                const play = li.querySelector(
                    '.playableTile__playButton a.sc-button-play, a.sc-button-play.playButton'
                );
                return {
                    index,
                    title: ((artwork && artwork.getAttribute('aria-label'))
                        || (link && link.innerText) || '').trim(),
                    href: (link && link.getAttribute('href')) || '',
                    playable: !!play
                        && (play.getAttribute('title') || '').toLowerCase() === 'play'
                };
            }).filter(item => item.title)"""
        )

    def _load_soundcloud_likes(
        self, page: Any, minimum: int, rounds: int
    ) -> list[dict[str, Any]]:
        items = self._soundcloud_like_items(page)
        for _ in range(rounds):
            if len(items) >= minimum:
                break
            before = len(items)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(1200)
            items = self._soundcloud_like_items(page)
            if len(items) == before:
                page.mouse.move(400, 400)
                page.mouse.wheel(0, 2500)
                page.wait_for_timeout(1000)
                items = self._soundcloud_like_items(page)
                if len(items) == before:
                    break
        return items

    @staticmethod
    def _find_soundcloud_like(
        items: list[dict[str, Any]], query: str
    ) -> dict[str, Any] | None:
        needle = normalize(query)
        if not needle:
            return None
        tokens = [token for token in needle.split() if len(token) > 1]
        for item in items:
            haystack = normalize(str(item.get("title", "")))
            if not haystack:
                continue
            if needle == haystack or needle in haystack or haystack in needle:
                return item
            if tokens and all(token in haystack for token in tokens):
                return item
        return None

    @staticmethod
    def _soundcloud_login_visible(page: Any) -> bool:
        for selector in (".webAuthContainerWrapper", ".modal.auth-modal"):
            try:
                if page.locator(selector).first.is_visible():
                    return True
            except Exception:
                continue
        return False

    def _require_soundcloud_login(self, page: Any, label: str) -> None:
        if self._soundcloud_login_visible(page) or "/signin" in page.url:
            raise BrowserError(
                f"SoundCloud просит войти, чтобы открыть {label}. "
                "Войдите в Chrome ассистента и повторите."
            )

    def _act_soundcloud_likes(self, args: dict[str, Any]) -> dict[str, Any]:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        query = str(args.get("query", "")).strip()
        want_random = bool(args.get("random"))
        play = bool(args.get("play", True))
        if play and not query:
            want_random = True

        page = self._soundcloud_page()
        if page is None:
            self._ensure_connection()
            page = self._context.new_page()
            self._assistant_page = page
        page.bring_to_front()

        if "/you/likes" not in page.url:
            page.goto(
                "https://soundcloud.com/you/likes",
                wait_until="domcontentloaded",
                timeout=45000,
            )
        self._dismiss_consent(page)
        self._close_soundcloud_auth(page)

        try:
            page.wait_for_selector("li.badgeList__item", timeout=15000)
        except PlaywrightTimeout as exc:
            self._require_soundcloud_login(page, "лайкнутые треки")
            raise BrowserError("SoundCloud не показал лайкнутые треки") from exc

        if not play:
            items = self._soundcloud_like_items(page)
            return {
                "mode": "open",
                "url": page.url,
                "title": self._safe_title(page),
                "count": len(items),
            }

        minimum = 120 if query else 80
        rounds = 8 if query else 5
        items = self._soundcloud_like_items(page)
        item: dict[str, Any] | None = None
        if not want_random and query:
            playable = [entry for entry in items if entry.get("playable")]
            item = self._find_soundcloud_like(playable, query)
        if item is None:
            items = self._load_soundcloud_likes(page, minimum=minimum, rounds=rounds)
            playable = [entry for entry in items if entry.get("playable")]
            if not playable:
                self._require_soundcloud_login(page, "лайкнутые треки")
                raise BrowserError("В лайкнутых треках нет доступных для воспроизведения")
            if want_random:
                item = random.choice(playable)
            else:
                item = self._find_soundcloud_like(playable, query)
                if item is None:
                    raise BrowserError(f"Не нашёл «{query}» среди лайкнутых треков")

        row = page.locator("li.badgeList__item").nth(int(item["index"]))
        button = row.locator("a.sc-button-play").first
        try:
            row.evaluate("el => el.scrollIntoView({block: 'center', inline: 'nearest'})")
            page.wait_for_timeout(400)
            row.hover(timeout=5000)
            page.wait_for_timeout(300)
            button.click(timeout=7000)
        except Exception:
            self._close_soundcloud_auth(page)
            try:
                row.evaluate("el => el.scrollIntoView({block: 'center', inline: 'nearest'})")
                page.wait_for_timeout(400)
                row.hover(timeout=5000)
                page.wait_for_timeout(300)
                button.click(timeout=7000)
            except Exception:
                button.evaluate("el => el.click()")

        try:
            page.wait_for_function(
                """() => {
                    const button = document.querySelector('.playControls__play');
                    return button && /pause/i.test(button.getAttribute('title') || '');
                }""",
                timeout=10000,
            )
        except PlaywrightTimeout as exc:
            self._require_soundcloud_login(page, "воспроизведение")
            raise BrowserError("SoundCloud не начал воспроизведение выбранного трека") from exc

        page.wait_for_timeout(500)
        return {
            "mode": "random" if want_random else "track",
            "url": page.url,
            "title": str(item.get("title") or self._safe_title(page)),
            "query": query,
            "count": len(items),
        }

    def _act_google(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        if not query:
            raise BrowserError("Пустой поисковый запрос")
        page = self._assistant_tab()
        page.bring_to_front()
        page.goto(
            "https://www.google.com/search?q=" + quote_plus(query),
            wait_until="domcontentloaded",
            timeout=45000,
        )
        self._dismiss_consent(page)
        return {"url": page.url, "title": self._safe_title(page), "query": query}

    def _act_search_result(self, args: dict[str, Any]) -> dict[str, Any]:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        try:
            index = max(1, int(args.get("index", 1) or 1))
        except (TypeError, ValueError):
            index = 1

        page = self._assistant_tab()
        page.bring_to_front()
        url = page.url or ""
        if "google." not in url or "/search" not in url:
            query = str(args.get("query") or "").strip()
            if not query:
                raise BrowserError(
                    "Сейчас открыта не выдача Google. Сначала скажите «открой …»"
                )
            page.goto(
                "https://www.google.com/search?q=" + quote_plus(query),
                wait_until="domcontentloaded",
                timeout=45000,
            )
        self._dismiss_consent(page)

        try:
            page.wait_for_selector("#rso a h3, #search a h3", timeout=10000)
        except PlaywrightTimeout:
            pass

        items = page.evaluate(
            """() => {
                const seen = new Set();
                const out = [];
                for (const a of document.querySelectorAll('#rso a, #search a')) {
                    const h3 = a.querySelector('h3');
                    if (!h3) continue;
                    if (a.closest('#tads, #bottomads, #rhs')) continue;
                    const href = a.href || '';
                    if (!/^https?:/i.test(href)) continue;
                    try {
                        const parsed = new URL(href);
                        if (/(^|\\.)google\\.[^.]*$/i.test(parsed.hostname)
                            && parsed.pathname !== '/url'
                            && parsed.pathname !== '/goto') continue;
                    } catch (e) {
                        continue;
                    }
                    if (seen.has(href)) continue;
                    seen.add(href);
                    out.push({href, title: (h3.innerText || a.innerText || '').trim()});
                }
                return out;
            }"""
        ) or []

        if not items:
            if page.locator("form#captcha-form, #recaptcha, iframe[src*='recaptcha']").count():
                raise BrowserError("Google показал капчу — откройте выдачу вручную")
            raise BrowserError("В выдаче нет обычных ссылок")
        if index > len(items):
            raise BrowserError(f"В выдаче только {len(items)} ссылок")

        item = items[index - 1]
        page.goto(str(item["href"]), wait_until="domcontentloaded", timeout=45000)
        self._dismiss_consent(page)
        return {
            "url": page.url,
            "title": self._safe_title(page),
            "index": index,
            "result_title": item["title"],
            "count": len(items),
        }

    def _act_new_tab(self, args: dict[str, Any]) -> dict[str, Any]:
        self._ensure_connection()
        page = self._context.new_page()
        url = args.get("url")
        if url:
            page.goto(self._clean_url(str(url)), wait_until="domcontentloaded", timeout=45000)
        self._assistant_page = page
        return {"url": page.url, "title": self._safe_title(page)}

    def _act_close_tab(self) -> dict[str, Any]:
        page = self._assistant_page
        pages = self._pages()
        if page is None or page.is_closed():
            page = pages[-1] if pages else None
        if page is not None:
            page.close()
        self._assistant_page = None
        remaining = self._pages()
        if not remaining:
            self._assistant_page = self._context.new_page()
        return {"tabs": self._tabs()}

    def _act_switch_tab(self, args: dict[str, Any]) -> dict[str, Any]:
        pages = self._pages()
        index = int(args.get("index", 1)) - 1
        if index < 0 or index >= len(pages):
            raise BrowserError(f"Нет вкладки с номером {index + 1}")
        page = pages[index]
        page.bring_to_front()
        self._assistant_page = page
        return self._tab_info(index + 1, page)

    def _act_nav(self, action: str) -> dict[str, Any]:
        page = self._assistant_tab()
        if action == "back":
            page.go_back(wait_until="domcontentloaded", timeout=30000)
        elif action == "forward":
            page.go_forward(wait_until="domcontentloaded", timeout=30000)
        else:
            page.reload(wait_until="domcontentloaded", timeout=30000)
        return {"url": page.url, "title": self._safe_title(page)}

    def _act_play(self, action: str) -> dict[str, Any]:
        page = self._soundcloud_page()
        if page is None:
            page = self._media_page()
        if "soundcloud.com" in page.url:
            player = page.locator(".playControls__play").first
            try:
                has_player = player.count() > 0
            except Exception:
                has_player = False
            if has_player:
                mode = "play" if action == "play" else "pause" if action == "pause" else "toggle"
                before = (player.get_attribute("title") or "").lower()
                playing = "pause" in before
                should_click = (
                    mode == "toggle"
                    or (mode == "play" and not playing)
                    or (mode == "pause" and playing)
                )
                if should_click:
                    player.click(timeout=5000)
                    page.wait_for_timeout(600)
                after = (player.get_attribute("title") or "").lower()
                return {
                    "state": "playing" if "pause" in after else "paused",
                    "url": page.url,
                }

        script = """(mode) => {
            const media = [...document.querySelectorAll('video, audio')]
                .filter(m => m.readyState > 0);
            if (!media.length) return null;
            const m = media[0];
            if (mode === 'pause' || (mode === 'toggle' && !m.paused)) {
                m.pause();
                return 'paused';
            }
            m.muted = false;
            m.play().catch(() => {});
            return 'playing';
        }"""
        mode = "play" if action == "play" else "pause" if action == "pause" else "toggle"
        result = page.evaluate(script, mode)
        if result is None:
            raise BrowserError("На странице нет видео или аудио")
        return {"state": result, "url": page.url}

    def _act_media_skip(self, action: str) -> dict[str, Any]:
        page = self._soundcloud_page()
        if page is None:
            page = self._media_page()
        page.bring_to_front()
        if "soundcloud.com" in page.url:
            selector = ".playControls__next" if action == "next" else ".playControls__prev"
            button = page.locator(selector).first
            if button.count():
                button.click(timeout=5000)
                page.wait_for_timeout(1200)
                return {"url": page.url, "title": self._safe_title(page)}
        if "youtube.com" in page.url:
            page.keyboard.press("Shift+N" if action == "next" else "Shift+P")
        else:
            page.keyboard.press("MediaTrackNext" if action == "next" else "MediaTrackPrevious")
        page.wait_for_timeout(800)
        return {"url": page.url, "title": self._safe_title(page)}

    def _act_screenshot(self, args: dict[str, Any]) -> dict[str, Any]:
        page = self._assistant_tab()
        raw_path = str(args.get("path") or "")
        if raw_path:
            target = Path(raw_path)
            if not target.is_absolute():
                target = STATE_DIR / "screenshots" / target
        else:
            target = STATE_DIR / "screenshots" / f"screen-{int(time.time())}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            target = target.with_suffix(".png")
        page.screenshot(path=str(target))
        return {"path": str(target)}
