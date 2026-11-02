from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

log = logging.getLogger(__name__)


class ControlServer(threading.Thread):
    """Локальный HTTP-мост между OpenCode-агентом и браузером ассистента."""

    def __init__(
        self,
        cfg: dict[str, Any],
        browser: Any,
        status_provider: Callable[[], dict[str, Any]],
        command_sink: Callable[[str], None],
    ) -> None:
        super().__init__(daemon=True, name="control-server")
        self.host = cfg["control"]["host"]
        self.port = int(cfg["control"]["port"])
        self.browser = browser
        self.status_provider = status_provider
        self.command_sink = command_sink
        self.httpd: ThreadingHTTPServer | None = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        assistant = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "VoiceAssistant/0.1"
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                log.debug("HTTP: " + fmt, *args)

            def _send(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    return {}
                if length <= 0 or length > 1_000_000:
                    return {}
                try:
                    data = json.loads(self.rfile.read(length).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return {}
                return data if isinstance(data, dict) else {}

            def do_GET(self) -> None:  # noqa: N802
                if self.path in {"/ping", "/status"}:
                    status = assistant.status_provider()
                    if self.path == "/ping":
                        self._send(200, {"ok": True, "status": status})
                    else:
                        self._send(200, {"ok": True, **status})
                    return
                self._send(404, {"ok": False, "error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                if self.path == "/browser":
                    payload = self._read_json()
                    action = str(payload.pop("action", "")).strip()
                    if not action:
                        self._send(400, {"ok": False, "error": "action is required"})
                        return
                    try:
                        result = assistant.browser.call(action, timeout=120, **payload)
                    except Exception as exc:  # noqa: BLE001
                        log.exception("Ошибка выполнения действия браузера")
                        self._send(500, {"ok": False, "error": str(exc)})
                        return
                    self._send(200, result)
                    return

                if self.path == "/command":
                    payload = self._read_json()
                    text = str(payload.get("text", "")).strip()
                    if not text:
                        self._send(400, {"ok": False, "error": "text is required"})
                        return
                    assistant.command_sink(text)
                    self._send(202, {"ok": True, "queued": text})
                    return

                self._send(404, {"ok": False, "error": "not found"})

        try:
            self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
            self.httpd.daemon_threads = True
            log.info("HTTP-мост слушает http://%s:%s", self.host, self.port)
            self.httpd.serve_forever(poll_interval=0.3)
        except OSError as exc:
            log.error("Не удалось запустить HTTP-мост на порту %s: %s", self.port, exc)
        finally:
            if self.httpd is not None:
                self.httpd.server_close()

    def stop(self) -> None:
        self._stop_event.set()
        if self.httpd is not None:
            self.httpd.shutdown()
