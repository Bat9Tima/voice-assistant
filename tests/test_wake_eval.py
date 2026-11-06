from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "wake_eval", ROOT / "tools" / "wake_eval.py")
we = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(we)

# Метка в аудио: вторые байты дубля ровно 77 запускают срабатывание.
MARKER = 77
STEP = we.STEP


def samples_with_marker(seconds: float = 0.5) -> list[int]:
    """Аудио с меткой: детектор-заглушка «слышит» ключевое слово."""
    count = int(seconds * 16000)
    data = np.zeros(count, dtype=np.int16)
    data[:10] = MARKER
    return data.tolist()


def silent_samples(seconds: float = 0.5) -> list[int]:
    return [0] * int(seconds * 16000)


def tone_samples(seconds: float = 0.5) -> list[int]:
    """Обычная речь без метки: чередование ±3000, метки ни в каком чанке нет."""
    count = int(seconds * 16000)
    data = np.where(np.arange(count) % 2 == 0, 3000, -3000).astype(np.int16)
    assert MARKER not in data.tolist()
    return data.tolist()


def write_wav(path: Path, samples: list[int], rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(b''.join(
            (s & 0xFFFF).to_bytes(2, "little") for s in samples))


def write_labels(path: Path, rows: list[dict]) -> None:
    fields = ["index", "file", "block", "kind", "level", "distance_cm",
              "condition"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            writer.writerow({"index": index, **row})


class StubDetector:
    """Прокси вместо WakeWordDetector: байт-метка == ключевое слово."""

    def __init__(self) -> None:
        self.resets = 0
        self.sizes: list[int] = []

    def reset(self) -> None:
        self.resets += 1

    def feed(self, pcm: bytes) -> bool:
        self.sizes.append(len(pcm))
        return (MARKER).to_bytes(2, "little") in pcm


class DatasetTestCase(unittest.TestCase):
    """Общая «временная папка-датасет» для интеграционных тестов."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name) / "VoiceDataset"
        self.audio = self.base / "audio"
        self.audio.mkdir(parents=True)
        self.addCleanup(self._tmp.cleanup)

        write_wav(self.audio / "001_M_wake_01.wav", samples_with_marker())
        write_wav(self.audio / "002_M_wake_02.wav", tone_samples())
        write_wav(self.audio / "003_N_neg_01.wav", silent_samples())
        write_wav(self.audio / "004_N_neg_02.wav", samples_with_marker())
        write_wav(self.audio / "005_O_cmd01_tv.wav", samples_with_marker())
        write_wav(self.audio / "006_M_wake_03.wav", samples_with_marker(),
                  rate=8000)

        write_labels(self.base / "labels.csv", [
            {"file": "001_M_wake_01.wav", "block": "M", "kind": "wake",
             "level": "normal", "distance_cm": "60", "condition": "silence"},
            {"file": "002_M_wake_02.wav", "block": "M", "kind": "wake",
             "level": "normal", "distance_cm": "150", "condition": "music"},
            {"file": "003_N_neg_01.wav", "block": "N", "kind": "negative",
             "level": "normal", "distance_cm": "60", "condition": "silence"},
            {"file": "004_N_neg_02.wav", "block": "N", "kind": "negative",
             "level": "normal", "distance_cm": "150", "condition": "tv"},
            {"file": "005_O_cmd01_tv.wav", "block": "O", "kind": "command",
             "level": "normal", "distance_cm": "150", "condition": "tv"},
            {"file": "006_M_wake_03.wav", "block": "M", "kind": "wake",
             "level": "normal", "distance_cm": "200", "condition": "fan"},
        ])

    def read_rows(self) -> list[dict]:
        with open(self.base / "labels.csv", encoding="utf-8-sig",
                  newline="") as f:
            return list(csv.DictReader(f, delimiter=";"))


class HelpersTest(unittest.TestCase):
    def test_rate(self) -> None:
        self.assertEqual(we.rate(2, 4), "2/4 (50%)")
        self.assertEqual(we.rate(1, 3), "1/3 (33%)")
        self.assertEqual(we.rate(0, 0), "0/0")

    def test_distance_label(self) -> None:
        for value, want in (("60", "60 см"), ("150", "150 см"),
                            ("1,5", "1.5 см"), ("200.0", "200 см"),
                            ("", "? см"), (None, "? см"), ("?", "? см")):
            with self.subTest(value=value):
                self.assertEqual(we.distance_label(value), want)

    def test_distance_key(self) -> None:
        self.assertEqual(we.distance_key("60 см"), 60.0)
        self.assertEqual(we.distance_key("150 см"), 150.0)
        self.assertEqual(we.distance_key("? см"), 9999.0)
        self.assertEqual(we.distance_key(""), 9999.0)

    def test_feed_slices_step_chunks(self) -> None:
        det = StubDetector()
        pcm = b"\x01\x02" * 10000  # 20000 байт = 2 полных шага + 4000 остатка
        self.assertFalse(we.feed(det, pcm))
        self.assertEqual(det.sizes, [STEP, STEP])

    def test_feed_propagates_hit_from_any_chunk(self) -> None:
        det = StubDetector()
        data = np.zeros(12000, dtype=np.int16)
        data[5999] = MARKER          # внутри второго чанка (8000:16000)
        pcm = b''.join(s.to_bytes(2, "little") for s in data.tolist())
        self.assertTrue(we.feed(det, pcm))

    def test_detect_feeds_lead_pcm_tail_and_resets_twice(self) -> None:
        det = StubDetector()
        det.sizes.clear()
        pcm = b"\xee\xee" * 1000
        hit = we.detect(det, pcm)
        self.assertFalse(hit)
        self.assertEqual(det.resets, 2)
        # lead 32 000 -> 4 шага, pcm короче шага не подаётся вовсе
        # (в датасете файлы ≥ 1.5 с, хвостовой <0.5 с остаток не влияет),
        # tail 32 000 -> ещё 4 шага.
        self.assertEqual(det.sizes, [STEP] * 8)


class EvaluateTest(DatasetTestCase):
    def run_evaluate(self, det: StubDetector | None = None):
        if det is None:
            det = StubDetector()
        out = io.StringIO()  # read_pcm печатает «! пропуск» — глушим шум.
        with contextlib.redirect_stdout(out):
            positives, negatives, payload = we.evaluate(
                self.read_rows(), self.audio, det,
                labels_path=self.base / "labels.csv")
        return det, positives, negatives, payload

    def test_counts_labels_by_kind(self) -> None:
        _, positives, negatives, _ = self.run_evaluate()
        self.assertEqual(len(positives), 2)
        self.assertEqual(len(negatives), 2)

    def test_command_row_is_ignored(self) -> None:
        _, positives, negatives, payload = self.run_evaluate()
        self.assertEqual(payload["positives"], 2)
        self.assertEqual(payload["negatives"], 2)
        for item in [*positives, *negatives]:
            self.assertNotIn(item["file"], ("005_O_cmd01_tv.wav",))

    def test_wrong_sample_rate_is_skipped(self) -> None:
        # 006 записан с 8 кГц — read_pcm пропускает его: позитивов всё еще 2.
        _, _, _, payload = self.run_evaluate()
        self.assertEqual(payload["positives"], 2)
        self.assertEqual(payload["misses"], ["002_M_wake_02.wav"])

    def test_frr_far_math_and_lists(self) -> None:
        _, _, _, payload = self.run_evaluate()
        self.assertEqual(payload["frr"], 0.5)
        self.assertEqual(payload["far"], 0.5)
        self.assertEqual(payload["misses"], ["002_M_wake_02.wav"])
        self.assertEqual(payload["false_accepts"], ["004_N_neg_02.wav"])

    def test_payload_keeps_custom_labels_path(self) -> None:
        _, _, _, payload = self.run_evaluate()
        self.assertIn(str(self.base / "labels.csv"), payload["labels"])

    def test_empty_labels_gives_none_rates(self) -> None:
        write_labels(self.base / "empty.csv", [])
        positives, negatives, payload = we.evaluate([], self.audio,
                                                    StubDetector())
        self.assertEqual(positives, [])
        self.assertEqual(negatives, [])
        self.assertIsNone(payload["frr"])
        self.assertIsNone(payload["far"])

    def test_metadata_reaches_items(self) -> None:
        det, positives, _, _ = self.run_evaluate()
        by_file = {i["file"]: i for i in positives}
        self.assertEqual(by_file["002_M_wake_02.wav"]["condition"], "music")
        self.assertEqual(by_file["002_M_wake_02.wav"]["distance"], "150 см")

    def test_detector_resets_between_and_around_files(self) -> None:
        det, positives, negatives, _ = self.run_evaluate()
        # На каждый файл: reset до и после → 4 файла = 8 сбросов.
        self.assertEqual(det.resets, 8)


class ReportTest(unittest.TestCase):
    def test_group_report_groups_and_sorts(self) -> None:
        items = [
            {"block": "N", "hit": True},
            {"block": "M", "hit": False},
            {"block": "N", "hit": True},
            {"block": "M", "hit": True},
        ]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            we.group_report("по блокам", items, lambda i: i["block"])
        text = out.getvalue()
        m_lines = [line for line in text.splitlines()
                   if line.startswith("    M") and "1/2 (50%)" in line]
        n_lines = [line for line in text.splitlines()
                   if line.startswith("    N") and "2/2 (100%)" in line]
        self.assertTrue(m_lines, f"нет строки M: {text!r}")
        self.assertTrue(n_lines, f"нет строки N: {text!r}")
        self.assertLess(text.index("    M"), text.index("    N"))

    def test_condition_order_silence_first(self) -> None:
        keys = ["tv", "music", "silence", "unknown", "fan"]
        ordered = sorted(keys, key=lambda k: (we.CONDITION_ORDER.get(k, 50), k))
        self.assertEqual(ordered[:2], ["silence", "fan"])
        self.assertEqual(ordered[-1], "unknown")

    def test_print_report_shows_frr_and_far(self) -> None:
        positives = [
            {"file": "a.wav", "block": "M", "condition": "silence",
             "distance": "60 см", "level": "normal", "hit": True},
            {"file": "b.wav", "block": "M", "condition": "silence",
             "distance": "60 см", "level": "normal", "hit": False},
        ]
        negatives = [
            {"file": "c.wav", "block": "N", "condition": "tv",
             "distance": "150 см", "level": "normal", "hit": True},
        ]
        payload = {"labels": "labels.csv", "positives": 2, "negatives": 1,
                   "misses": ["b.wav"], "false_accepts": ["c.wav"],
                   "frr": 0.5, "far": 1.0}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            we.print_report(Path("x/labels.csv"), positives, negatives,
                            payload)
        text = out.getvalue()
        self.assertIn("FRR: 1/2 (50%)", text)
        self.assertIn("FAR: 1/1 (100%)", text)
        self.assertIn("  пропуски:", text)
        self.assertIn("    b.wav", text)
        self.assertIn("  ложные:", text)
        self.assertIn("    c.wav", text)


if __name__ == "__main__":
    unittest.main()
