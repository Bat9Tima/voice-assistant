"""Генерация предзаписанных фраз через ProxyAPI (Fish Audio).

Читает каталог фраз, синтезирует по несколько дублей на каждую, обрезает
тишину по краям, выравнивает громкость и сохраняет WAV 44.1 кГц моно.

Пример:
    set PROXYAPI_KEY=...
    python tools/generate_phrases.py --only done --takes 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.tts import SAMPLE_RATE, normalize_pcm, save_wav  # noqa: E402

API_URL = "https://api.proxyapi.ru/v1/audio/speech"
DEFAULT_MODEL = "fish-audio/drama-3"
DEFAULT_VOICE = "09f3f59b50254dc29f419fde998fd474"


def synth(text: str, model: str, voice: str, api_key: str, retries: int = 4) -> bytes:
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "pcm",
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code not in (429, 500, 502, 503, 504):
                raise last_error from exc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        if attempt + 1 < retries:
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"не удалось синтезировать за {retries} попыток: {last_error}")


def process(pcm: bytes):
    return normalize_pcm(pcm)


def write_wav(path: Path, data) -> float:
    save_wav(path, data)
    return len(data) / SAMPLE_RATE


def build_tasks(catalog: dict, takes_override: int | None, only: str | None) -> list[dict]:
    tasks: list[dict] = []
    for category in catalog["categories"]:
        if only and category["id"] != only:
            continue
        takes = (
            takes_override
            or int(catalog.get("takes_override", {}).get(category["id"], 0))
            or int(catalog.get("takes_default", 5))
        )
        for phrase_id, text in category["phrases"]:
            for take in range(1, takes + 1):
                tasks.append(
                    {
                        "id": phrase_id,
                        "category": category["id"],
                        "text": text,
                        "take": take,
                        "file": f"{phrase_id}_{take:02d}.wav",
                    }
                )
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=ROOT / "assets" / "phrases" / "catalog.json")
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "phrases")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--takes", type=int, default=None, help="переопределить число дублей")
    parser.add_argument("--only", default=None, help="только одна категория")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="перегенерировать существующие")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("PROXYAPI_KEY", "").strip()
    if not api_key:
        print("Нужна переменная окружения PROXYAPI_KEY", file=sys.stderr)
        return 2

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    tasks = build_tasks(catalog, args.takes, args.only)
    args.out.mkdir(parents=True, exist_ok=True)

    pending = [task for task in tasks if args.force or not (args.out / task["file"]).exists()]
    print(f"Всего задач: {len(tasks)}, к генерации: {len(pending)}")
    if args.dry_run:
        for task in pending[:20]:
            print(f"  {task['file']}: {task['text']}")
        return 0

    durations: dict[str, float] = {}
    errors: list[str] = []
    started = time.monotonic()
    done = 0

    def work(task: dict) -> tuple[dict, float]:
        pcm = synth(task["text"], args.model, args.voice, api_key)
        data = process(pcm)
        if data is None:
            raise RuntimeError("пустое аудио")
        return task, write_wav(args.out / task["file"], data)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            done += 1
            try:
                result, duration = future.result()
                durations[result["file"]] = round(duration, 3)
                print(f"[{done}/{len(pending)}] {result['file']:16s} {duration:.2f}s  {result['text']}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{task['file']}: {exc}")
                print(f"[{done}/{len(pending)}] {task['file']:16s} ОШИБКА {exc}", file=sys.stderr)

    manifest_path = args.out / "manifest.json"
    previous: dict = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            previous = {}
    known = previous.get("durations", {}) if isinstance(previous, dict) else {}
    known.update(durations)

    phrases = []
    for category in catalog["categories"]:
        for phrase_id, text in category["phrases"]:
            files = sorted(
                path.name
                for path in args.out.glob(f"{phrase_id}_*.wav")
                if path.name in known
            )
            if files:
                phrases.append(
                    {
                        "id": phrase_id,
                        "category": category["id"],
                        "text": text,
                        "takes": [
                            {"file": name, "duration": known[name]} for name in files
                        ],
                    }
                )

    manifest = {
        "version": 1,
        "model": args.model,
        "voice": args.voice,
        "sample_rate": SAMPLE_RATE,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "durations": known,
        "phrases": phrases,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    elapsed = time.monotonic() - started
    total_audio = sum(durations.values())
    print(
        f"\nГотово: {len(durations)} файлов, {total_audio:.1f} с аудио, "
        f"{elapsed:.0f} с работы. Ошибок: {len(errors)}"
    )
    if errors:
        for line in errors[:10]:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
