from __future__ import annotations

import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "coverage_html",
    "logs",
    "models",
    "state",
    "VoiceAssistantTools",
    "VoiceDataset_backups",
}
TEXT_SUFFIXES = {
    ".bat",
    ".cfg",
    ".cmd",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_NAMES = {".editorconfig", ".gitattributes", ".gitignore", "requirements.lock"}
UTF8_BOM = b"\xef\xbb\xbf"


def text_files() -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        relative = Path(dirpath).relative_to(ROOT)
        if relative.parts[:2] == ("wake_model", "data"):
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() in TEXT_SUFFIXES or name in TEXT_NAMES:
                files.append(path)
    return files


class EncodingTest(unittest.TestCase):
    def test_text_files_are_utf8(self) -> None:
        for path in text_files():
            with self.subTest(path=path.relative_to(ROOT)):
                path.read_bytes().decode("utf-8")

    def test_powershell_scripts_have_bom(self) -> None:
        for path in text_files():
            if path.suffix.lower() != ".ps1":
                continue
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertTrue(
                    path.read_bytes().startswith(UTF8_BOM),
                    "PowerShell 5.1 читает UTF-8 без BOM как ANSI",
                )

    def test_batch_files_are_ascii(self) -> None:
        for path in text_files():
            if path.suffix.lower() not in {".bat", ".cmd"}:
                continue
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertTrue(
                    path.read_bytes().isascii(),
                    "cmd.exe ненадёжно читает не-ASCII batch-файлы",
                )


if __name__ == "__main__":
    unittest.main()
