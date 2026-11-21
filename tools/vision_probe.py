"""Проверка зрения: один скриншот → ProxyAPI → текстовый ответ.

Скриншот не сохраняется на диск: он живёт в памяти, пока идёт запрос.
Модель берётся из config.json (`cloud_llm.model`), поэтому пробник проверяет
ровно ту модель, с которой работает ассистент.

Примеры:
    python tools/vision_probe.py
    python tools/vision_probe.py --prompt "Что за ошибка на экране?"
    python tools/vision_probe.py --image state/sample.png --dry-run
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402

DEFAULT_PROMPT = (
    "Опиши, что видно на экране, в двух-трёх предложениях. "
    "Если на экране текст, код или ошибка — назови главное. "
    "Если экран чёрный или пустой, так и скажи."
)
MIN_COMPLETION_TOKENS = 2000


def capture_screen(
    max_width: int, quality: int
) -> tuple[bytes, str, tuple[int, int], tuple[int, int]]:
    """Снимает основной экран и возвращает JPEG в памяти."""
    from PIL import Image, ImageGrab

    image = ImageGrab.grab()
    original = image.size
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max_width > 0 and image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue(), "image/jpeg", original, image.size


def encode_image(
    path: Path, max_width: int, quality: int
) -> tuple[bytes, str, tuple[int, int], tuple[int, int]]:
    """Читает готовую картинку и приводит её к тому же виду, что и скриншот."""
    from PIL import Image

    image = Image.open(path)
    original = image.size
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max_width > 0 and image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue(), "image/jpeg", original, image.size


def build_payload(
    cfg: dict[str, Any], model: str, prompt: str, image_b64: str, mime: str
) -> dict[str, Any]:
    section = cfg.get("cloud_llm") or {}
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_completion_tokens": max(
            int(section.get("max_completion_tokens") or MIN_COMPLETION_TOKENS),
            MIN_COMPLETION_TOKENS,
        ),
        "temperature": float(section.get("temperature", 0.2)),
        "stream": False,
    }
    if section.get("reasoning_effort"):
        payload["reasoning_effort"] = section["reasoning_effort"]
    params = (section.get("model_params") or {}).get(model) or {}
    for key, value in params.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Один скриншот → ProxyAPI → ответ")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image", type=Path, help="вместо экрана взять файл")
    parser.add_argument("--model", default="", help="по умолчанию cloud_llm.model")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--quality", type=int, default=75)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true", help="только снимок и кодирование")
    args = parser.parse_args()

    cfg = load_config()
    section = cfg.get("cloud_llm") or {}
    model = (args.model or str(section.get("model") or "")).strip()
    base_url = str(section.get("base_url") or "https://api.proxyapi.ru/v1").rstrip("/")
    api_key = (
        str(section.get("api_key") or "").strip()
        or str((cfg.get("tts") or {}).get("api_key") or "").strip()
    )

    try:
        if args.image:
            raw, mime, original, sent = encode_image(args.image, args.max_width, args.quality)
            source = str(args.image)
        else:
            raw, mime, original, sent = capture_screen(args.max_width, args.quality)
            source = "экран"
    except Exception as exc:  # noqa: BLE001
        print(f"Не удалось получить изображение: {exc}", file=sys.stderr)
        return 1

    print(f"Источник: {source}")
    print(
        f"Размер: {original[0]}x{original[1]} → {sent[0]}x{sent[1]}, JPEG {len(raw) / 1024:.0f} КБ"
    )
    print(f"Модель: {model or 'не задана'}")
    if args.dry_run:
        print("dry-run: запрос не отправлялся")
        return 0
    if not model:
        print("Не задана модель: cloud_llm.model", file=sys.stderr)
        return 1
    if not api_key:
        print("Не задан ключ: cloud_llm.api_key или tts.api_key", file=sys.stderr)
        return 1

    payload = build_payload(cfg, model, args.prompt, base64.b64encode(raw).decode("ascii"), mime)
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Сервер недоступен: {exc}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        print(f"Ответ не JSON: {body[:300]}", file=sys.stderr)
        return 1
    choices = data.get("choices") or []
    message = (choices[0] or {}).get("message") if choices else {}
    content = (message or {}).get("content")
    if isinstance(content, list):
        content = "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        print(f"Пустой ответ: {body[:500]}", file=sys.stderr)
        return 1

    usage = data.get("usage") or {}
    print(
        f"Ответ за {elapsed:.2f} с, токены {usage.get('prompt_tokens')}→"
        f"{usage.get('completion_tokens')}"
    )
    print("---")
    print(content.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
