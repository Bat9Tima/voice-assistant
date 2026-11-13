from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools" / "record_dataset"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


record = _load("record_dataset_record", TOOLS / "record.py")
qc = _load("record_dataset_qc", TOOLS / "qc.py")


class ParseOnlyTest(unittest.TestCase):
    def test_indices_and_names(self) -> None:
        indices, names = record.parse_only(["155", "175_M_wake_22", "178_M_wake_25.wav"])
        self.assertEqual(indices, {155})
        self.assertEqual(names, {"175_M_wake_22", "178_M_wake_25"})

    def test_empty_tokens_ignored(self) -> None:
        self.assertEqual(record.parse_only(["", " ", ","]), (set(), set()))

    def test_full_path_reduced_to_name(self) -> None:
        _indices, names = record.parse_only([r"C:\data\audio\209_O_cmd02_music.wav"])
        self.assertEqual(names, {"209_O_cmd02_music"})


class SelectRowsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"index": 1, "file": "001_A_wake_01.wav", "session": "s1"},
            {"index": 155, "file": "155_M_wake_02.wav", "session": "s5"},
        ]

    def test_session_filter(self) -> None:
        selected = record.select_rows(self.rows, "s5")
        self.assertEqual([row["index"] for row in selected], [155])

    def test_only_by_index(self) -> None:
        selected = record.select_rows(self.rows, "all", ["1"])
        self.assertEqual([row["index"] for row in selected], [1])

    def test_only_by_file_stem(self) -> None:
        selected = record.select_rows(self.rows, "all", ["155_M_wake_02.wav"])
        self.assertEqual([row["index"] for row in selected], [155])

    def test_only_keeps_session(self) -> None:
        selected = record.select_rows(self.rows, "s1", ["155"])
        self.assertEqual(selected, [])

    def test_only_unknown_returns_nothing(self) -> None:
        self.assertEqual(record.select_rows(self.rows, "all", ["999"]), [])


class LabelDefaultsTest(unittest.TestCase):
    def test_empty_old_condition_matches_silence(self) -> None:
        self.assertTrue(qc.same_label_value("condition", "", "silence"))
        self.assertTrue(qc.same_label_value("speaker", "", "self"))
        self.assertTrue(qc.same_label_value("room", "", "комната"))

    def test_empty_non_default_still_mismatch(self) -> None:
        self.assertFalse(qc.same_label_value("condition", "", "music"))
        self.assertFalse(qc.same_label_value("speaker", "", "guest"))

    def test_real_value_compared_exactly(self) -> None:
        self.assertFalse(qc.same_label_value("condition", "tv", "silence"))
        self.assertTrue(qc.same_label_value("condition", "tv", "tv"))

    def test_other_fields_have_no_default(self) -> None:
        self.assertFalse(qc.same_label_value("block", "", "M"))


if __name__ == "__main__":
    unittest.main()
