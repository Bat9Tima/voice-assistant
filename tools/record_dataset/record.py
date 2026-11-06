#!/usr/bin/env python
"""Гид по записи датасета для голосового ассистента.

Пишет WAV 16 кГц / моно / 16 бит через тот же путь захвата, что и ассистент
(app.audio.Microphone), режет тишину по краям и ведёт labels.csv.

Запуск:  start.cmd            — запись
         start.cmd --plan     — только показать/создать план
         start.cmd --check    — проверить микрофон и фон
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
import wave
from pathlib import Path

import numpy as np

try:
    import msvcrt
except ImportError:  # pragma: no cover - Windows only
    msvcrt = None

RATE = 16000
PROJECT = Path(r"C:\Users\TimAS\Documents\Проект по умолчанию")
BASE = Path(__file__).resolve().parent
DEFAULT_OUT = Path.home() / "Documents" / "VoiceDataset"

FIELDS = [
    "index", "file", "block", "kind", "text", "action", "level", "rate",
    "distance_cm", "condition", "speaker", "room", "session", "take",
    "max_sec", "trim", "status", "peak_db", "rms_db", "dur_s", "notes",
]

DEFAULT_CONDITION = "silence"
DEFAULT_SPEAKER = "self"
DEFAULT_ROOM = "комната"

COMMANDS = [
    ("громкость пять", "volume:50"),
    ("громкость десять", "volume:100"),
    ("громкость саундклауда десять", "volume:soundcloud:100"),
    ("громче на два", "volume:+20"),
    ("тише", "volume:-10"),
    ("выключи звук", "mute"),
    ("новая вкладка", "newtab"),
    ("закрой вкладку", "closetab"),
    ("следующий трек", "media:next"),
    ("предыдущий трек", "media:prev"),
    ("открой ютуб", "goto:youtube"),
    ("найди на ютубе лоуфай", "youtube:лоуфай"),
]

QUIET_SUBSET = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]
LOUD_SUBSET = [0, 2, 8, 10, 11]
FAST_SLOW = [0, 2, 3, 8, 10, 11]
WHISPER_SUBSET = [0, 2, 4, 8, 10, 11]

WAKE_COMMANDS = [
    ("хей луна громкость пять", "volume:50"),
    ("хей луна следующий трек", "media:next"),
    ("хей луна открой ютуб", "goto:youtube"),
    ("хей луна громкость саундклауда десять", "volume:soundcloud:100"),
    ("хей луна тише", "volume:-10"),
    ("хей луна найди на ютубе лоуфай", "youtube:лоуфай"),
    ("хей луна предыдущий трек", "media:prev"),
    ("хей луна новая вкладка", "newtab"),
    ("хей луна громче на два", "volume:+20"),
    ("хей луна выключи звук", "mute"),
]

NEGATIVES = [
    ("хей люда", "похоже на wake, но не оно"),
    ("хей луна-парк", "лишнее слово"),
    ("лей луну", "похожие звуки"),
    ("хей ну на", "похожий ритм"),
    ("эй лужа", "похожие звуки"),
    ("хей юля", "похожий ритм"),
    ("луна", "одно слово"),
    ("завтра надо позвонить в поликлинику", "бытовая речь"),
    ("включи телевизор", "неподдерживаемая команда"),
    ("громкость", "команда без числа"),
    ("следующий", "одно слово"),
    ("ФОН: музыка или телевизор", "включите фон и молчите"),
]

# ------------------------------------------------- второй круг (индексы 118+)
ROUND2_WAKE_2M = 6

ROUND2_DIGITS = [
    ("громкость ноль", "volume:0"),
    ("громкость один", "volume:10"),
    ("громкость два", "volume:20"),
    ("громче на один", "volume:+10"),
    ("тише на два", "volume:-20"),
    ("громче на три", "volume:+30"),
]

ROUND2_COMMANDS = [
    ("пауза", "media:pause"),
    ("продолжи", "media:play"),
    ("следующее видео", "media:next"),
    ("открой гугл", "goto:google"),
    ("включи на саундклауде лоуфай", "soundcloud:лоуфай"),
    ("громче в саундклауде", "soundcloud_volume:up:10"),
    ("тише на один в саундклауде", "soundcloud_volume:down:10"),
    ("выключи звук в саундклауде", "soundcloud_volume:mute"),
    ("какие вкладки", "tabs"),
    ("переключись на вкладку два", "tab:2"),
    ("что играет", "status"),
    ("включи звук", "unmute"),
]

ROUND2_NEGATIVES = [
    ("луна сегодня красивая", "слово «луна» в предложении"),
    ("пошли в луна-парк", "«луна-парк» в обычной речи"),
    ("эй ты, иди сюда", "похоже на wake"),
    ("какая завтра погода", "бытовая речь"),
    ("включи телевизор", "неподдерживаемая команда"),
    ("хей люда", "похоже на wake, но не оно"),
]

# ------------------------------------------------- третий круг (M/N/O)
# M — wake word в разных условиях: дистанция, фон, вполоборота.
# (дистанция, условие, уровень, сколько дублей, пометка)
ROUND3_WAKE = [
    (100, "silence", "normal", 3, "1 м"),
    (200, "silence", "normal", 4, "2 м"),
    (300, "silence", "normal", 4, "3 м"),
    (200, "silence", "normal", 3, "2 м, вполоборота"),
    (150, "silence", "normal", 4, "1.5 м, тишина"),
    (150, "fan", "normal", 4, "1.5 м, вентилятор"),
    (150, "music", "normal", 4, "1.5 м, музыка"),
    (150, "tv", "normal", 4, "1.5 м, телевизор"),
]

# N — сложные негативы: похожие фразы, «луна» в речи, фон, бытовые фразы.
ROUND3_NEGATIVES = [
    ("хей люда", "silence", "похоже на wake"),
    ("хей юля", "silence", "похожий ритм"),
    ("эй лужа", "silence", "похожие звуки"),
    ("лей луна", "silence", "похожие звуки"),
    ("хей ну на", "silence", "похожий ритм"),
    ("хей луна-парк", "silence", "лишнее слово"),
    ("хей луно", "silence", "другое окончание"),
    ("эй луно", "silence", "другое окончание"),
    ("хей лупа", "silence", "похожее слово"),
    ("эй руна", "silence", "похожее слово"),
    ("луна сегодня красивая", "silence", "«луна» в предложении"),
    ("пошли в луна-парк", "silence", "«луна-парк» в речи"),
    ("включи песню про луну", "silence", "«луну» в команде"),
    ("на небе полная луна", "silence", "«луна» в предложении"),
    ("луна светит в окно", "silence", "«луна» в предложении"),
    ("расскажи про луну", "silence", "«луну» в команде"),
    ("ФОН: музыка", "music", "фон без речи"),
    ("ФОН: телевизор", "tv", "фон без речи"),
    ("ФОН: вентилятор", "fan", "фон без речи"),
    ("ФОН: улица", "street", "фон без речи"),
    ("завтра надо позвонить в поликлинику", "silence", "бытовая речь"),
    ("какая завтра погода", "silence", "бытовая речь"),
    ("включи телевизор", "silence", "неподдерживаемая команда"),
    ("громкость", "silence", "команда без числа"),
]

# O — команды в фоне и на расстоянии.
ROUND3_COMMANDS = [
    ("громкость пять", "volume:50"),
    ("громкость саундклауда десять", "volume:soundcloud:100"),
    ("громче на два", "volume:+20"),
    ("тише", "volume:-10"),
    ("пауза", "media:pause"),
    ("следующий трек", "media:next"),
    ("открой ютуб", "goto:youtube"),
    ("включи на саундклауде лоуфай", "soundcloud:лоуфай"),
    ("какие вкладки", "tabs"),
    ("что играет", "status"),
]

ROUND3_COMMAND_CONDITIONS = [
    ("music", 150, "музыка"),
    ("tv", 200, "телевизор"),
]

HINTS = {
    "A": "Только фраза «хей луна», без команды.",
    "A2": "Только «эй луна», без команды.",
    "B": "Вся фраза на одном выдохе, без паузы после «хей луна».",
    "C": "Обычный голос, как всегда.",
    "D": "Тише обычного, но голосом — не шёпотом.",
    "E": "Громко, но не крик.",
    "G": "Шёпотом.",
    "H": "Это НЕ должно срабатывать. Говорите как обычно.",
    "I": "Фраза «хей луна» с 2 метров, вполоборота, как в жизни.",
    "J": "Голосом, обычная громкость. Цифру называйте словом.",
    "K": "Обычный голос. Это команды из повседневного набора.",
    "L": "Это НЕ должно срабатывать. Говорите как обычно.",
    "M": "Wake word в разных условиях. Стоять на месте, лицом к микрофону; "
         "оба слова одинаково громко. Соблюдайте дистанцию и фон из пометки.",
    "N": "Это НЕ должно срабатывать. Говорите как обычно; для «ФОН» — просто включите фон.",
    "O": "Команды в фоне и на расстоянии. Говорите как всегда.",
}

LEVEL_RANGES = {
    "normal": (-24.0, -6.0),
    "quiet": (-42.0, -18.0),
    "loud": (-12.0, -3.0),
    "whisper": (-60.0, -18.0),
}

OUT = DEFAULT_OUT


# --------------------------------------------------------------------- план
def build_plan() -> list[dict]:
    rows: list[dict] = []

    def add(name, block, kind, text, action, level, rate, distance, session,
            take, max_sec, notes, trim="1", condition=DEFAULT_CONDITION,
            speaker=DEFAULT_SPEAKER, room=DEFAULT_ROOM):
        index = len(rows) + 1
        rows.append({
            "index": index,
            "file": f"{index:03d}_{name}.wav",
            "block": block,
            "kind": kind,
            "text": text,
            "action": action,
            "level": level,
            "rate": rate,
            "distance_cm": distance,
            "condition": condition,
            "speaker": speaker,
            "room": room,
            "session": session,
            "take": take,
            "max_sec": max_sec,
            "trim": trim,
            "status": "pending",
            "peak_db": "",
            "rms_db": "",
            "dur_s": "",
            "notes": notes,
        })

    # A — только wake word
    for take in range(1, 9):
        add(f"A_wake_{take:02d}", "A", "wake", "хей луна", "", "normal",
            "normal", 60, "s1", take, 3.0, "обычный голос")
    for take in (9, 10):
        add(f"A_wake_{take:02d}", "A", "wake", "хей луна", "", "quiet",
            "normal", 60, "s1", take, 3.0, "тихий голос, не шёпот")
    for take in (11, 12):
        add(f"A_wake_{take:02d}", "A", "wake", "хей луна", "", "normal",
            "normal", 200, "s1", take, 3.0, "около 2 м, вполоборота")

    # A2 — алиас
    for take in range(1, 5):
        add(f"A2_alias_{take:02d}", "A2", "wake_alias", "эй луна", "",
            "normal", "normal", 60, "s1", take, 3.0, "алиас")

    # B — wake + команда одной фразой
    for take, (text, action) in enumerate(WAKE_COMMANDS, 1):
        add(f"B_wc{take:02d}", "B", "wake_command", text, action, "normal",
            "normal", 60, "s3", take, 6.0, "одной фразой, без паузы")

    # C — команды, обычный голос, 3 дубля
    for take in (1, 2, 3):
        for num, (text, action) in enumerate(COMMANDS, 1):
            add(f"C_c{num:02d}_t{take}", "C", "command", text, action,
                "normal", "normal", 60, f"s{take}", take, 5.0,
                f"команда {num} из 12")

    # D — тихо, 2 дубля
    for take in (1, 2):
        for num in QUIET_SUBSET:
            text, action = COMMANDS[num]
            add(f"D_c{num + 1:02d}_t{take}", "D", "command", text, action,
                "quiet", "normal", 60, f"s{take + 1}", take, 5.5,
                f"тихо, команда {num + 1} из 12")

    # E — громко
    for take, num in enumerate(LOUD_SUBSET, 1):
        text, action = COMMANDS[num]
        add(f"E_c{num + 1:02d}", "E", "command", text, action, "loud",
            "normal", 60, "s1", take, 5.0,
            f"громко, команда {num + 1} из 12")

    # F — быстро / медленно
    for take, num in enumerate(FAST_SLOW, 1):
        text, action = COMMANDS[num]
        add(f"F_c{num + 1:02d}_fast", "F", "command", text, action, "normal",
            "fast", 60, "s2", take, 4.5, "быстро, слитно")
    for take, num in enumerate(FAST_SLOW, 1):
        text, action = COMMANDS[num]
        add(f"F_c{num + 1:02d}_slow", "F", "command", text, action, "normal",
            "slow", 60, "s2", take + 6, 6.5, "паузы 0.3-0.5 с между словами")

    # G — шёпот
    for take, num in enumerate(WHISPER_SUBSET, 1):
        text, action = COMMANDS[num]
        add(f"G_c{num + 1:02d}_whisper", "G", "command", text, action,
            "whisper", "normal", 60, "s2", take, 5.0, "шёпотом")

    # H — негативы
    for take, (text, notes) in enumerate(NEGATIVES, 1):
        trim = "0" if take == len(NEGATIVES) else "1"
        max_sec = 6.0 if trim == "0" else 5.0
        add(f"H_neg_{take:02d}", "H", "negative", text, "", "normal",
            "normal", 60, "s3", take, max_sec, notes, trim=trim)

    assert len(rows) == 117, len(rows)
    return rows


def build_round2(start: int = 118) -> list[dict]:
    """Второй круг: дальний микрофон, цифры и команды вне первого набора."""
    rows: list[dict] = []

    def add(name, block, kind, text, action, level, rate, distance, session,
            take, max_sec, notes, trim="1", condition=DEFAULT_CONDITION,
            speaker=DEFAULT_SPEAKER, room=DEFAULT_ROOM):
        index = start + len(rows)
        rows.append({
            "index": index,
            "file": f"{index:03d}_{name}.wav",
            "block": block,
            "kind": kind,
            "text": text,
            "action": action,
            "level": level,
            "rate": rate,
            "distance_cm": distance,
            "condition": condition,
            "speaker": speaker,
            "room": room,
            "session": session,
            "take": take,
            "max_sec": max_sec,
            "trim": trim,
            "status": "pending",
            "peak_db": "",
            "rms_db": "",
            "dur_s": "",
            "notes": notes,
        })

    for take in range(1, ROUND2_WAKE_2M + 1):
        add(f"I_wake_2m_{take:02d}", "I", "wake", "хей луна", "", "normal",
            "normal", 200, "s4", take, 3.5, "2 м, вполоборота")

    for take in (1, 2):
        for num, (text, action) in enumerate(ROUND2_DIGITS, 1):
            add(f"J_vol{num:02d}_t{take}", "J", "command", text, action,
                "normal", "normal", 60, "s4", take, 5.0,
                f"цифры, {num} из {len(ROUND2_DIGITS)}")

    for num, (text, action) in enumerate(ROUND2_COMMANDS, 1):
        add(f"K_cmd{num:02d}", "K", "command", text, action, "normal",
            "normal", 60, "s4", num, 6.0, f"команда {num} из {len(ROUND2_COMMANDS)}")

    for num, (text, notes) in enumerate(ROUND2_NEGATIVES, 1):
        add(f"L_neg{num:02d}", "L", "negative", text, "", "normal",
            "normal", 60, "s4", num, 5.0, notes)

    return rows


def build_round3(start: int = 154) -> list[dict]:
    """Третий круг: wake в условиях, сложные негативы и команды в фоне."""
    rows: list[dict] = []

    def add(name, block, kind, text, action, level, rate, distance, session,
            take, max_sec, notes, trim="1", condition=DEFAULT_CONDITION,
            speaker=DEFAULT_SPEAKER, room=DEFAULT_ROOM):
        index = start + len(rows)
        rows.append({
            "index": index,
            "file": f"{index:03d}_{name}.wav",
            "block": block,
            "kind": kind,
            "text": text,
            "action": action,
            "level": level,
            "rate": rate,
            "distance_cm": distance,
            "condition": condition,
            "speaker": speaker,
            "room": room,
            "session": session,
            "take": take,
            "max_sec": max_sec,
            "trim": trim,
            "status": "pending",
            "peak_db": "",
            "rms_db": "",
            "dur_s": "",
            "notes": notes,
        })

    # M — wake word: дистанция, фон, вполоборота.
    take = 0
    for distance, condition, level, count, note in ROUND3_WAKE:
        for _ in range(count):
            take += 1
            add(f"M_wake_{take:02d}", "M", "wake", "хей луна", "",
                level, "normal", distance, "s5", take, 4.0, note,
                condition=condition)

    # N — сложные негативы.
    for take, (text, condition, notes) in enumerate(ROUND3_NEGATIVES, 1):
        background = text.startswith("ФОН:")
        add(f"N_neg_{take:02d}", "N", "negative", text, "", "normal",
            "normal", 150, "s6", take, 6.0 if background else 5.0, notes,
            trim="0" if background else "1", condition=condition)

    # O — команды в фоне.
    take = 0
    for condition, distance, label in ROUND3_COMMAND_CONDITIONS:
        for num, (text, action) in enumerate(ROUND3_COMMANDS, 1):
            take += 1
            add(f"O_cmd{num:02d}_{condition}", "O", "command", text, action,
                "normal", "normal", distance, "s7", take, 5.0,
                f"{label}, {distance} см", condition=condition)

    return rows


def build_full_plan() -> list[dict]:
    """Все круги: A–H, I–L и M–O."""
    first = build_plan()
    second = build_round2(start=first[-1]["index"] + 1)
    third = build_round3(start=second[-1]["index"] + 1)
    return [*first, *second, *third]


def write_csv(rows: list[dict]) -> None:
    try:
        with open(OUT / "labels.csv", "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=FIELDS, delimiter=";", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        print("  ! labels.csv открыт в другой программе — закройте Excel.")


def load_plan(rebuild: bool) -> list[dict]:
    path = OUT / "labels.csv"
    if path.exists() and not rebuild:
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f, delimiter=";"))
        if rows:
            return rows
        print("  ! labels.csv пустой — строю план заново.")
    rows = build_plan()
    write_csv(rows)
    return rows


# ------------------------------------------------------------------ утилиты
def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-9))


def trim_speech(samples: np.ndarray, noise: float) -> np.ndarray | None:
    """Оставляет 0.5 с тишины до и после речи."""
    frame = int(0.02 * RATE)
    count = samples.size // frame
    if count < 5:
        return None

    blocks = samples[: count * frame].astype(np.float32) / 32768.0
    blocks = blocks.reshape(count, frame)
    levels = np.sqrt((blocks ** 2).mean(axis=1))

    loud = float(np.percentile(levels, 95))
    threshold = max(noise * 1.5, loud * 0.04, 0.0015)
    mask = levels > threshold
    if not mask.any() or mask.sum() * 0.02 < 0.15:
        return None

    first = int(np.argmax(mask))
    last = count - 1 - int(np.argmax(mask[::-1]))

    margin = int(0.5 * RATE)
    start = first * frame - margin
    end = (last + 1) * frame + margin

    if start < 0:
        out = np.concatenate([np.zeros(-start, dtype=np.int16), samples[:end]])
    else:
        out = samples[start:end]
    if end > samples.size:
        out = np.concatenate([out, np.zeros(end - samples.size, dtype=np.int16)])
    return out


def save_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(samples.astype(np.int16).tobytes())


def clear_keys() -> None:
    if msvcrt is None:
        return
    while msvcrt.kbhit():
        msvcrt.getwch()


def read_char() -> str:
    """Читает символ с учётом раскладки (getwch, а не getch)."""
    if msvcrt is None:
        return ""
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):  # стрелки и прочие расширенные клавиши
        msvcrt.getwch()
        return ""
    return ch


def action_for(ch: str) -> str | None:
    """Пробел, S/R/P/Q — в латинской и русской раскладке."""
    if ch in (" ", "\r", "\n"):
        return "space"
    low = ch.lower()
    if low in ("s", "ы"):
        return "skip"
    if low in ("r", "к"):
        return "redo"
    if low in ("p", "з"):
        return "play"
    if low in ("q", "й") or ch == "\x1b":
        return "quit"
    return None


def wait_key(
    prompt: str,
    allowed: tuple[str, ...] = ("space", "skip", "redo", "play", "quit"),
    reader=None,
) -> str:
    """Ждёт клавишу и возвращает только разрешённые действия.

    Буфер очищается перед чтением: иначе нажатие, сделанное во время
    прослушивания или записи, срабатывает на следующем вопросе.
    """
    print(prompt, end=" ", flush=True)
    if reader is None and msvcrt is None:
        return input().strip().lower() or "space"

    if reader is None:
        clear_keys()
        reader = read_char

    while True:
        key = action_for(reader())
        if key and key in allowed:
            return key


def review(read, play_fn) -> str:
    """Разбор дубля: P можно нажимать сколько угодно, приём — только пробел.

    read() возвращает 'space' | 'redo' | 'skip' | 'play' | 'quit'.
    Возвращает 'space' (принять), 'redo', 'skip' или 'quit'.
    """
    while True:
        key = read()
        if key == "play":
            play_fn()
            continue
        if key in ("space", "redo", "skip", "quit"):
            return key


def capture(mic, max_sec: float) -> np.ndarray | None:
    clear_keys()
    mic.flush()
    chunks: list[bytes] = []
    started = time.monotonic()
    while time.monotonic() - started < max_sec:
        if msvcrt is not None and msvcrt.kbhit():
            key = action_for(read_char())
            if key == "space":
                break
            if key == "quit":
                return None
        block = mic.read(timeout=0.25)
        if block:
            chunks.append(block)
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.frombuffer(b"".join(chunks), dtype=np.int16)


def verdict(level: str, peak_db: float) -> str:
    low, high = LEVEL_RANGES.get(level, (-60.0, 0.0))
    if peak_db < low:
        return f"тише цели ({low:.0f}…{high:.0f})"
    if peak_db > high:
        return f"громче цели ({low:.0f}…{high:.0f})"
    return "в диапазоне"


def play(samples: np.ndarray) -> None:
    try:
        import sounddevice as sd

        sd.play(samples.astype(np.int16), RATE)
    except Exception as exc:  # noqa: BLE001
        print(f"\r  Прослушивание недоступно: {exc}")
        clear_keys()
        return

    duration = samples.size / RATE
    started = time.monotonic()
    while time.monotonic() - started < duration + 0.3:
        if msvcrt is not None and msvcrt.kbhit():
            read_char()
            break
        time.sleep(0.02)
    sd.stop()
    clear_keys()


# -------------------------------------------------------------------- режимы
def cmd_plan(rebuild: bool) -> None:
    rows = load_plan(rebuild)
    total = len(rows)
    by_session: dict[str, int] = {}
    by_block: dict[str, dict[str, int]] = {}
    for row in rows:
        by_session[row["session"]] = by_session.get(row["session"], 0) + 1
        by_block.setdefault(row["session"], {})
        by_block[row["session"]][row["block"]] = (
            by_block[row["session"]].get(row["block"], 0) + 1
        )

    print(f"Файл: {OUT / 'labels.csv'}")
    print(f"Всего дублей: {total}")
    for session in sorted(by_session):
        blocks = ", ".join(
            f"{block} {count}" for block, count in sorted(by_block[session].items())
        )
        print(f"  {session}: {by_session[session]:3d}  ({blocks})")


def cmd_extend() -> None:
    """Добавляет второй круг к уже записанному labels.csv."""
    rows = load_plan(rebuild=False)
    if any(str(row.get("block", "")) in {"I", "J", "K", "L"} for row in rows):
        print("Второй круг уже есть в плане — ничего не меняю.")
        print(f"Файл: {OUT / 'labels.csv'}")
        return
    start = max((int(row["index"]) for row in rows), default=0) + 1
    extra = build_round2(start)
    write_csv([*rows, *extra])
    print(f"Добавлено {len(extra)} дублей: {extra[0]['file']} … {extra[-1]['file']}")
    print(f"Файл: {OUT / 'labels.csv'}")
    print("Запись: start.cmd --session s4")


def cmd_extend3() -> None:
    """Добавляет третий круг: wake в условиях, негативы и команды в фоне."""
    rows = load_plan(rebuild=False)
    if any(str(row.get("block", "")) in {"M", "N", "O"} for row in rows):
        print("Блоки M/N/O уже есть в плане — ничего не меняю.")
        print(f"Файл: {OUT / 'labels.csv'}")
        return
    if not any(str(row.get("block", "")) in {"I", "J", "K", "L"} for row in rows):
        start = max((int(row["index"]) for row in rows), default=0) + 1
        second = build_round2(start)
        rows = [*rows, *second]
        print(f"Сначала добавлен второй круг: {len(second)} дублей.")
    start = max((int(row["index"]) for row in rows), default=0) + 1
    extra = build_round3(start)
    write_csv([*rows, *extra])
    print(f"Добавлено {len(extra)} дублей: {extra[0]['file']} … {extra[-1]['file']}")
    print(f"Файл: {OUT / 'labels.csv'}")
    print("Запись: start.cmd --session s5 (wake) → s6 (негативы) → s7 (команды)")


def ensure_env() -> dict[str, str]:
    path = OUT / "environment.txt"
    if path.exists():
        return {}
    print()
    print("Первый запуск: пара вопросов про микрофон (сохранится в environment.txt).")
    distance = input("  Расстояние до микрофона, см [60]: ").strip() or "60"
    room = input("  Комната [комната]: ").strip() or "комната"
    enhancements = input(
        "  Улучшения микрофона в Windows (AGC/шумодав) включены? [не знаю]: "
    ).strip() or "не знаю"

    try:
        import sounddevice as sd

        info = sd.query_devices(kind="input")
        mic_name = f"{info['name']} (index {info['index']})"
        native = int(info["default_samplerate"])
    except Exception as exc:  # noqa: BLE001
        mic_name = f"не определён: {exc}"
        native = 0

    path.write_text(
        "\n".join([
            f"Дата: {time.strftime('%d.%m.%Y %H:%M')}",
            f"Микрофон: {mic_name}",
            f"Частота устройства: {native} Гц",
            "Захват: mono, int16, 16000 Гц (как у ассистента)",
            f"Расстояние: {distance} см",
            f"Комната: {room}",
            f"Улучшения Windows: {enhancements}",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"  Записано: {path}")
    return {"distance": distance}


def assistant_running() -> bool:
    """Ассистент отвечает на контрольном порту — значит, микрофон занят им."""
    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:8765/ping", timeout=0.7):
            return True
    except Exception:  # noqa: BLE001
        return False


def preflight(mic) -> float:
    if assistant_running():
        print()
        print("Ассистент запущен (отвечает на 127.0.0.1:8765).")
        print("Закройте его: трей -> выход. Иначе он будет слышать ваши фразы")
        print("и выполнять команды во время записи.")
        sys.exit(2)

    data = b""
    for _ in range(4):
        block = mic.read(timeout=0.4)
        if block:
            data += block
    if not data:
        print("Микрофон не отдаёт данные. Проверьте устройство ввода.")
        sys.exit(2)

    samples = np.frombuffer(data, dtype=np.int16)
    if samples.size == 0 or int(np.abs(samples).max()) == 0:
        print()
        print("Микрофон отдаёт цифровую тишину.")
        print("Скорее всего, ассистент уже держит микрофон.")
        print("Закройте его (трей -> выход), проверьте в диспетчере pythonw.exe и запустите снова.")
        sys.exit(2)

    noise = mic.measure_noise(1.2)
    print(f"  Захват: {mic.capture_rate} Гц -> {RATE} Гц")
    print(f"  Фон: {dbfs(noise):.1f} dBFS RMS", end="")
    if noise > 0.01:
        print("  — высоковато: вентилятор, кондиционер или улица?")
    else:
        print("  — нормально")

    import sounddevice as sd

    info = sd.query_devices(kind="input")
    print(f"  Микрофон: {info['name']} (index {info['index']}, {info['default_samplerate']:.0f} Гц)")
    return noise


def make_mic():
    """Открывает микрофон тем же путём, что и ассистент."""
    try:
        from app.audio import Microphone
    except Exception as exc:  # noqa: BLE001
        print(f"Не удалось загрузить app.audio: {exc}")
        print("Запускайте через start.cmd — он берёт python из .venv проекта.")
        return None

    mic = Microphone(samplerate=RATE, block_size=4000)
    try:
        mic.start()
    except Exception as exc:  # noqa: BLE001
        print(f"Не удалось открыть микрофон: {exc}")
        print("Закройте ассистента (трей -> выход) и проверьте устройство ввода.")
        return None
    return mic


def cmd_check() -> None:
    mic = make_mic()
    if mic is None:
        return
    try:
        preflight(mic)
    finally:
        mic.stop()


def cmd_calibrate() -> None:
    mic = make_mic()
    if mic is None:
        return
    try:
        noise = preflight(mic)
        print()
        print("Калибровка: 3 дубля «громкость пять». Пробел — стоп.")
        for level, hint in (
            ("normal", "обычный голос"),
            ("quiet", "тихо, но голосом"),
            ("loud", "громко, без крика"),
        ):
            print()
            print(f"  {level}: {hint}")
            if wait_key("  Пробел — запись · Q — выход",
                        allowed=("space", "quit")) == "quit":
                return
            samples = capture(mic, 5.0)
            if samples is None:
                return
            out = trim_speech(samples, noise)
            if out is None:
                print("  Речь не найдена.")
                continue
            peak = dbfs(float(np.abs(out).max()) / 32768.0)
            rms = dbfs(float(np.sqrt(np.mean(
                (out.astype(np.float32) / 32768.0) ** 2
            ))))
            save_wav(OUT / "calibration" / f"{level}.wav", out)
            print(f"  пик {peak:.1f} dBFS · RMS {rms:.1f} dBFS · "
                  f"{out.size / RATE:.2f} с · {verdict(level, peak)}")
        print()
        print(f"Файлы: {OUT / 'calibration'}")
        print("Пришлите эти три цифры — скажу, попадаете ли вы в диапазон.")
    finally:
        mic.stop()


def parse_only(tokens: list[str]) -> tuple[set[int], set[str]]:
    """Разбирает --only: номера дублей и/или имена файлов без .wav."""
    indices: set[int] = set()
    names: set[str] = set()
    for token in tokens:
        token = token.strip()
        if not token or not any(ch.isalnum() for ch in token):
            continue
        if token.isdigit():
            indices.add(int(token))
            continue
        name = Path(token).name
        if name.lower().endswith(".wav"):
            name = name[:-4]
        names.add(name)
    return indices, names


def select_rows(rows: list[dict], session: str = "all",
                only: list[str] | None = None) -> list[dict]:
    """Строки плана для сессии; при --only — только указанные дубли."""
    selected = [
        row for row in rows
        if session == "all" or row["session"] == session
    ]
    if not only:
        return selected
    indices, names = parse_only(only)
    return [
        row for row in selected
        if int(row["index"]) in indices
        or str(row["file"]) in names
        or Path(str(row["file"])).stem in names
    ]


def cmd_record(session: str, only: list[str] | None = None) -> None:
    rows = load_plan(rebuild=False)
    selected = select_rows(rows, session, only)
    if only:
        pending = selected
        print(f"Перезапись: {len(pending)} из {len(rows)}")
    else:
        pending = [row for row in selected if row["status"] != "done"]
    if not pending:
        if only:
            print(f"Не найдено дублей: {', '.join(only)}")
        else:
            print(f"Сессия {session}: всё уже записано.")
        return

    if only:
        print("Старые файлы сохраняются в _redo.")
    else:
        print(f"Сессия {session}: осталось {len(pending)} из {len(selected)}")
    print("Пробел — запись/стоп. После старта подождите полсекунды и говорите.")

    mic = make_mic()
    if mic is None:
        return
    try:
        noise = preflight(mic)
        print()

        for row in pending:
            index = int(row["index"])
            print()
            print("-" * 66)
            label = f"[{index}/{len(rows)}]  {row['block']} · {row['session']} · {row['level']}"
            if row["rate"] != "normal":
                label += f" · {row['rate']}"
            condition = str(row.get("condition") or "")
            if condition and condition != DEFAULT_CONDITION:
                label += f" · {condition}"
            print(label)
            print(f"  Фраза: «{row['text']}»")
            hint = HINTS.get(row["block"], "")
            if row["block"] == "F":
                hint = "Быстро, слитно." if row["rate"] == "fast" else \
                    "Паузы 0.3-0.5 с между словами."
            print(f"  {hint}")
            if str(row["distance_cm"]) not in ("60", "60.0"):
                print(f"  Дистанция: {row['distance_cm']} см")
            if row["notes"]:
                print(f"  Пометка: {row['notes']}")

            pre = wait_key(
                "  Пробел — запись · S — пропустить · Q — выход",
                allowed=("space", "skip", "quit"),
            )
            if pre == "quit":
                return
            if pre == "skip":
                row["status"] = "skipped"
                write_csv(rows)
                continue

            keep_raw = str(row.get("trim", "1")) == "0"
            max_sec = float(row["max_sec"])

            while True:
                print("  ● Говорите…", end="", flush=True)
                samples = capture(mic, max_sec)
                if samples is None:
                    return
                if not keep_raw:
                    out = trim_speech(samples, noise)
                    if out is None:
                        print("\r  Речь не найдена. Проверьте уровень и повторите.")
                        choice = wait_key(
                            "  Пробел — ещё раз · S — пропустить · Q — выход",
                            allowed=("space", "skip", "quit"),
                        )
                        if choice == "quit":
                            return
                        if choice == "skip":
                            row["status"] = "skipped"
                            write_csv(rows)
                            break
                        continue
                else:
                    out = samples

                peak = dbfs(float(np.abs(out).max()) / 32768.0)
                rms = dbfs(float(np.sqrt(np.mean(
                    (out.astype(np.float32) / 32768.0) ** 2
                ))))
                note = "" if keep_raw else (" · " + verdict(row["level"], peak))
                print(f"\r  пик {peak:6.1f} dBFS · RMS {rms:6.1f} dBFS · "
                      f"{out.size / RATE:4.2f} с{note}")
                if peak > -0.5:
                    print("  ! Клиппинг: отойдите от микрофона или говорите тише.")

                result = review(
                    lambda: wait_key(
                        "  Пробел — принять · R — перезаписать · P — прослушать · "
                        "S — пропустить · Q — выход",
                        allowed=("space", "redo", "play", "skip", "quit"),
                    ),
                    lambda out=out: play(out),
                )
                if result == "quit":
                    return
                if result == "redo":
                    continue
                if result == "skip":
                    row["status"] = "skipped"
                    write_csv(rows)
                    break

                if only:
                    # Перезапись по --only: старый дубль не теряем.
                    target = OUT / "audio" / row["file"]
                    backup = OUT / "_redo" / row["file"]
                    if target.exists() and not backup.exists():
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(target, backup)
                    save_wav(target, out)
                else:
                    save_wav(OUT / "audio" / row["file"], out)
                row["status"] = "done"
                row["peak_db"] = f"{peak:.1f}"
                row["rms_db"] = f"{rms:.1f}"
                row["dur_s"] = f"{out.size / RATE:.2f}"
                write_csv(rows)
                break
    finally:
        mic.stop()

    done = sum(1 for row in rows if row["status"] == "done")
    print()
    print(f"Готово: {done} из {len(rows)}. Разметка: {OUT / 'labels.csv'}")


# ---------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Запись датасета для ассистента")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"папка датасета (по умолчанию {DEFAULT_OUT})")
    parser.add_argument("--session", default="all",
                        help="s1, s2, s3 или all")
    parser.add_argument("--plan", action="store_true", help="показать план")
    parser.add_argument("--check", action="store_true", help="проверить микрофон")
    parser.add_argument("--calibrate", action="store_true", help="3 пробных дубля")
    parser.add_argument("--rebuild", action="store_true", help="пересобрать labels.csv")
    parser.add_argument("--extend", action="store_true",
                        help="добавить второй круг (2 м, цифры, новые команды)")
    parser.add_argument("--extend3", action="store_true",
                        help="добавить третий круг (wake в условиях, негативы, команды в фоне)")
    parser.add_argument("--only", default="",
                        help="перезаписать конкретные дубли: номера или имена через запятую")
    args = parser.parse_args()

    global OUT
    OUT = args.out.expanduser().resolve()
    (OUT / "audio").mkdir(parents=True, exist_ok=True)

    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))

    if args.plan:
        cmd_plan(args.rebuild)
        return 0
    if args.check:
        cmd_check()
        return 0
    if args.calibrate:
        ensure_env()
        cmd_calibrate()
        return 0
    if args.extend:
        cmd_extend()
        return 0
    if args.extend3:
        cmd_extend3()
        return 0

    ensure_env()
    session = args.session.strip().lower()
    if session not in ("all", "s1", "s2", "s3", "s4", "s5", "s6", "s7"):
        print("Сессия: s1–s7 или all")
        return 1
    only = [part for part in args.only.replace(";", ",").split(",") if part.strip()]
    cmd_record(session, only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
