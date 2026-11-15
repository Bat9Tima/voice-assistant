"""Контекст пользователя из простого текстового файла.

Файл пишет сам пользователь: персона ассистента, факты о себе, предпочтения.
Читается перед каждым ответом, поэтому правки применяются без перезапуска.
История реплик живёт отдельно, в ``app.session``.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_CONTEXT_LIMIT = 4000


class Profile:
    """Читает текстовый контекст и отдаёт блок для системного промпта."""

    def __init__(self, path: Path | None = None, limit: int = DEFAULT_CONTEXT_LIMIT) -> None:
        self.path = Path(path) if path is not None else None
        self.limit = max(0, int(limit))
        self._warned = False

    def text(self) -> str:
        """Содержимое файла без хвостовых пробелов. Ошибки не роняют диалог."""
        if self.path is None or not self.path.exists():
            return ""
        try:
            data = self.path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            if not self._warned:
                log.warning("Контекст %s не прочитан: %s", self.path, exc)
                self._warned = True
            return ""
        self._warned = False

        text = "\n".join(line.rstrip() for line in data.splitlines()).strip()
        if self.limit and len(text) > self.limit:
            text = text[: self.limit].rsplit("\n", 1)[0].rstrip() + "\n…"
        return text

    def block(self) -> str:
        """Готовый кусок системного промпта или пустая строка."""
        text = self.text()
        if not text:
            return ""
        return "Контекст от пользователя (персона и факты):\n" + text
