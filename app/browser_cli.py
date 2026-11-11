from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from typing import Any

from .config import load_config


def _post(payload: dict[str, Any], timeout: float = 120) -> dict[str, Any]:
    cfg = load_config()["control"]
    url = f"http://{cfg['host']}:{cfg['port']}/browser"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _daemon_alive() -> bool:
    cfg = load_config()["control"]
    url = f"http://{cfg['host']}:{cfg['port']}/ping"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            return response.status == 200
    except Exception:
        return False


def _run_direct(payload: dict[str, Any]) -> dict[str, Any]:
    from .browser import BrowserController

    controller = BrowserController(load_config())
    try:
        controller.start()
        action = payload.pop("action")
        return controller.call(action, timeout=120, **payload)
    finally:
        controller.stop()


def _print_result(data: Any, action: str) -> None:
    if action == "text":
        print(data.get("text", ""))
        return
    if action == "tabs":
        for tab in data.get("tabs", []):
            print(f"{tab['index']}. {tab['title']} — {tab['url']}")
        return
    if action in {"goto", "open", "youtube", "google", "soundcloud", "search_result"}:
        print(f"{data.get('title', '')} — {data.get('url', '')}".strip(" —"))
        return
    if action == "soundcloud_volume":
        mode = data.get("mode")
        if mode == "mute":
            print("Звук SoundCloud выключен")
        elif mode == "unmute":
            print("Звук SoundCloud включён")
        else:
            print(f"Громкость SoundCloud: {data.get('volume', 0)}%")
        return
    if action == "soundcloud_likes":
        if data.get("mode") == "open":
            print(f"Лайкнутые треки: {data.get('count', 0)} шт. — {data.get('url', '')}")
        else:
            print(f"{data.get('title', '')} — {data.get('url', '')}".strip(" —"))
        return
    if action == "status":
        print(f"{data.get('title', '')} — {data.get('url', '')}")
        return
    if action == "screenshot":
        print(data.get("path", ""))
        return
    print(json.dumps(data, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="browser.cmd",
        description="Управление браузером ассистента (через демон или напрямую).",
    )
    parser.add_argument("--direct", action="store_true", help="Не использовать демон")
    parser.add_argument("--json", action="store_true", help="Вывод в JSON")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод ошибок")
    parser.add_argument("action", help="goto, youtube, google, tabs, click, ...")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    namespace = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if namespace.verbose else logging.CRITICAL,
        format="%(message)s",
    )

    action = namespace.action.lower()
    rest = [part for part in namespace.args if part != "--"]

    payload: dict[str, Any] = {"action": action}
    if action in {"goto", "open", "newtab"}:
        payload["url"] = " ".join(rest)
    elif action in {"youtube", "google"}:
        payload["query"] = " ".join(rest)
    elif action == "search_result":
        try:
            payload["index"] = int(rest[0]) if rest else 1
        except ValueError:
            print("Использование: search_result [номер]", file=sys.stderr)
            return 2
    elif action == "soundcloud":
        payload["query"] = " ".join(rest)
    elif action == "soundcloud_volume":
        mode = rest[0].lower() if rest else "up"
        payload["mode"] = mode
        if len(rest) > 1:
            try:
                payload["value"] = int(rest[1])
            except ValueError:
                print("Использование: soundcloud_volume <up|down|set|mute|unmute> [0-100]", file=sys.stderr)
                return 2
    elif action == "soundcloud_likes":
        words: list[str] = []
        want_random = False
        want_play = True
        for part in rest:
            if part in {"--random", "-r"}:
                want_random = True
            elif part in {"--open", "-o"}:
                want_play = False
            else:
                words.append(part)
        payload["query"] = " ".join(words)
        payload["random"] = want_random
        payload["play"] = want_play
    elif action == "click":
        payload["selector"] = " ".join(rest)
    elif action == "fill":
        if len(rest) < 2:
            print("Использование: fill <css-селектор> <текст>", file=sys.stderr)
            return 2
        payload["selector"] = rest[0]
        payload["text"] = " ".join(rest[1:])
    elif action == "press":
        payload["key"] = rest[0] if rest else ""
    elif action in {"play", "pause", "stop"}:
        payload["site"] = rest[0].lower() if rest else ""
    elif action == "seek":
        try:
            payload["seconds"] = int(rest[0])
        except (IndexError, ValueError):
            print(
                "Использование: seek <секунды> [youtube|soundcloud]",
                file=sys.stderr,
            )
            return 2
        payload["site"] = rest[1].lower() if len(rest) > 1 else ""
    elif action == "tab":
        try:
            payload["index"] = int(rest[0])
        except (IndexError, ValueError):
            print("Использование: tab <номер>", file=sys.stderr)
            return 2
    elif action == "scroll":
        payload["direction"] = rest[0] if rest else "down"
    elif action == "wait":
        try:
            payload["ms"] = int(rest[0])
        except (IndexError, ValueError):
            payload["ms"] = 1000
    elif action == "screenshot":
        payload["path"] = rest[0] if rest else ""
    elif action == "eval":
        payload["script"] = " ".join(rest)
    elif action == "type":
        payload["text"] = " ".join(rest)
    elif action in {"status", "tabs", "back", "forward", "reload", "play", "pause", "next", "prev", "closetab"}:
        pass
    else:
        print(f"Неизвестное действие: {action}", file=sys.stderr)
        return 2

    try:
        if not namespace.direct and _daemon_alive():
            result = _post(payload)
        else:
            result = _run_direct(dict(payload))
    except (urllib.error.URLError, ConnectionError):
        try:
            result = _run_direct(dict(payload))
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка браузера: {exc}", file=sys.stderr)
            return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка браузера: {exc}", file=sys.stderr)
        return 1

    if not result.get("ok"):
        print(f"Ошибка: {result.get('error', 'неизвестная')}", file=sys.stderr)
        return 1

    data = result.get("data")
    if namespace.json:
        print(json.dumps(data, ensure_ascii=False))
    else:
        _print_result(data, action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
