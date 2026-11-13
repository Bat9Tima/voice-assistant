from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from .config import ROOT, STATE_DIR

log = logging.getLogger(__name__)

SESSION_FILE = STATE_DIR / "opencode_session.json"


class OpenCodeError(RuntimeError):
    pass


class OpenCodeBrain:
    """Отправляет свободные команды агенту OpenCode и помнит сессию."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg["opencode"]
        self.agent = str(self.cfg.get("agent") or "voice")
        self.model = str(self.cfg.get("model") or "").strip()
        self.timeout = float(self.cfg.get("timeout_sec") or 180)
        self.standalone = bool(self.cfg.get("standalone"))
        self.cli = self._find_cli()
        self._lock = threading.Lock()
        self.signature = self._signature()
        self.session_id = self._load_session()

    # ------------------------------------------------------------------- setup
    def _find_cli(self) -> str | None:
        candidates: list[str] = []
        configured = str(self.cfg.get("cli") or "").strip()
        if configured:
            candidates.append(configured)
        env_cli = os.environ.get("OPENCODE_CLI", "").strip()
        if env_cli:
            candidates.append(env_cli)
        for name in ("opencode", "opencode-cli"):
            found = shutil.which(name)
            if found:
                candidates.append(found)
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        if local_appdata:
            candidates.append(
                os.path.join(
                    local_appdata,
                    "Programs",
                    "ai.opencode.desktop",
                    "resources",
                    "opencode-cli.exe",
                )
            )
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                log.info("OpenCode CLI: %s", candidate)
                return candidate
        log.warning("OpenCode CLI не найден")
        return None

    def _signature(self) -> str:
        """Отпечаток промпта и модели.

        Сессия OpenCode хранит историю прошлых команд. После правки промпта
        старые примеры могут перебивать новые правила, поэтому начинаем
        новую сессию. Если файла промпта нет, отпечаток пустой и сессия
        сохраняется как раньше.
        """
        prompt = ROOT / ".opencode" / "agents" / f"{self.agent}.md"
        digest = hashlib.sha256()
        try:
            digest.update(prompt.read_bytes())
        except OSError:
            return ""
        digest.update(b"\0")
        digest.update(self.agent.encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.model.encode("utf-8"))
        digest.update(b"\0")
        # Сессия OpenCode привязана к папке, в которой создана. После переноса
        # проекта старую сессию открыть нельзя (Location not found), поэтому
        # корень входит в отпечаток и автоматически сбрасывает её.
        digest.update(str(ROOT).encode("utf-8"))
        return digest.hexdigest()

    def _load_session(self) -> str | None:
        try:
            data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None
        session_id = data.get("session_id")
        stored = str(data.get("signature") or "")
        stored_root = str(data.get("root") or "")
        if session_id and stored_root and stored_root != str(ROOT):
            log.info("Папка проекта изменилась — начинаю новую сессию OpenCode")
            return None
        if session_id and self.signature and stored != self.signature:
            log.info("Промпт агента изменился — начинаю новую сессию OpenCode")
            return None
        return session_id

    def _save_session(self, session_id: str | None) -> None:
        self.session_id = session_id
        payload = {"session_id": session_id, "root": str(ROOT)}
        if self.signature:
            payload["signature"] = self.signature
        try:
            SESSION_FILE.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("Не удалось сохранить сессию OpenCode: %s", exc)

    def reset_session(self) -> None:
        self._save_session(None)

    # -------------------------------------------------------------------- call
    def ask(self, text: str) -> str:
        if not self.cfg.get("enabled", True):
            return ""
        if not self.cli:
            raise OpenCodeError("OpenCode CLI не найден. Укажите путь в config.json → opencode.cli")
        with self._lock:
            answer = self._run(text, self.session_id)
            if answer is None:
                # Сессия могла быть удалена — пробуем новую.
                self._save_session(None)
                answer = self._run(text, None)
            return answer or ""

    def _command(self, text: str, session_id: str | None) -> list[str]:
        cmd = [self.cli, "run", "--format", "json", "--auto"]
        if self.standalone:
            cmd.append("--standalone")
        if self.agent:
            cmd.extend(["--agent", self.agent])
        if self.model:
            cmd.extend(["--model", self.model])
        if session_id:
            cmd.extend(["--session", session_id])
        cmd.append(text)
        return cmd

    @staticmethod
    def _stale_session_error(text: str, session_id: str | None) -> bool:
        """Сессия существует, но привязана к другой папке?"""
        if not session_id:
            return False
        lowered = (text or "").lower()
        if "not found" not in lowered:
            return False
        return "session" in lowered or "location" in lowered

    def _run(self, text: str, session_id: str | None) -> str | None:
        cmd = self._command(text, session_id)

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["NO_COLOR"] = "1"
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        log.info("OpenCode: %s", text)
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                env=env,
                creationflags=creationflags,
            )
        except subprocess.TimeoutExpired as exc:
            raise OpenCodeError(f"OpenCode не ответил за {self.timeout:.0f} секунд") from exc
        except OSError as exc:
            raise OpenCodeError(f"Не удалось запустить OpenCode: {exc}") from exc

        texts: list[str] = []
        errors: list[str] = []
        new_session = session_id
        for line in (completed.stdout or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            new_session = event.get("sessionID") or new_session
            if event.get("type") == "error":
                error = event.get("error") or {}
                message = str(error.get("message") or error.get("type") or "").strip()
                if message:
                    errors.append(message)
                continue
            if event.get("type") != "text":
                continue
            part = event.get("part") or {}
            if part.get("type") == "text" and part.get("text"):
                texts.append(part["text"].strip())

        if new_session:
            self._save_session(new_session)

        if texts:
            return texts[-1]

        stderr = (completed.stderr or "").strip()
        details = "\n".join([*errors, stderr])
        if self._stale_session_error(details, session_id):
            log.warning("Сессия OpenCode не найдена в текущей папке, создаю новую")
            return None
        if completed.returncode != 0:
            if errors:
                detail = errors[-1][:500]
            elif stderr:
                detail = stderr.splitlines()[-1][:500]
            else:
                detail = f"код {completed.returncode}"
            raise OpenCodeError(f"OpenCode завершился с ошибкой: {detail}")

        if errors:
            raise OpenCodeError(f"OpenCode: {errors[-1][:500]}")
        return ""
