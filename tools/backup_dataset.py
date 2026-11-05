#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Архивная копия датасета VoiceDataset в соседний zip.

Только читает датасет; ничего в нём не меняет и не удаляет. Пишет
%USERPROFILE%\\Documents\\VoiceDataset_backups\\VoiceDataset-<дата-время>.zip
и проверяет целостность архива (testzip). Повторный запуск делает новый
снимок, старые не трогает.

Запуск: .venv\\Scripts\\python.exe tools\\backup_dataset.py [--keep N]
--keep N — после создания оставить только N свежих архивов (по умолчанию 6).
"""

from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

SOURCE = Path.home() / "Documents" / "VoiceDataset"
DEST_DIR = Path.home() / "Documents" / "VoiceDataset_backups"
KEEP = 6


def main() -> int:
    parser = argparse.ArgumentParser(description="Бэкап датасета в zip")
    parser.add_argument("--keep", type=int, default=KEEP,
                        help=f"сколько свежих архивов хранить (по умолчанию {KEEP})")
    args = parser.parse_args()

    if not SOURCE.exists():
        print(f"Нет датасета: {SOURCE}")
        return 2
    if not (SOURCE / "labels.csv").exists():
        print(f"В {SOURCE} нет labels.csv — это датасет?")
        return 2
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = DEST_DIR / f"VoiceDataset-{stamp}.zip"
    files = sorted(
        p for p in SOURCE.rglob("*")
        if p.is_file() and not p.name.endswith((".zip", ".part"))
    )
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, arcname=str(path.relative_to(SOURCE)))

    with zipfile.ZipFile(target) as z:
        bad = z.testzip()
        if bad is not None:
            print(f"Архив повреждён: {bad}. Файл: {target}")
            return 1
        stored = z.namelist()

    size_mb = target.stat().st_size / 1024 / 1024
    print(f"Архив: {target}")
    print(f"Файлов: {len(stored)} · размер {size_mb:.1f} MB · целостность ok")

    archives = sorted(DEST_DIR.glob("VoiceDataset-*.zip"))
    for old in archives[:-max(args.keep, 1)]:
        old.unlink()
        print(f"Удалён старый архив: {old.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
