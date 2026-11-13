from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "wake_model" / "scripts" / "record_positives.py"


def _load():
    spec = importlib.util.spec_from_file_location("record_positives", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["record_positives"] = module
    spec.loader.exec_module(module)
    return module


rec = _load()

CELL = {
    "id": "quiet_d05_norm",
    "phrase": "хей луна",
    "distance_m": 0.5,
    "condition": "quiet",
    "level": "norm",
    "target": 3,
    "accepted": 0,
    "attempts": 0,
    "status": "pending",
}

METRICS = {
    "duration_s": 3.0,
    "speech_s": 1.2,
    "peak_db": -10.0,
    "rms_db": -25.0,
    "clipped": 0,
    "pcm_md5": "hash",
}


def _session(speaker: str = "anna", mic: str = "laptop", target: int = 1) -> dict:
    return {
        "speaker": speaker,
        "mic": mic,
        "settings": dict(rec.DEFAULTS),
        "status": "active",
        "limit": 0,
        "takes": [],
        "plan": [dict(CELL, target=target)],
    }


def _pending() -> dict:
    return {
        "cell": CELL["id"],
        "take": 1,
        "file": "raw/x.wav",
        "phrase": "хей луна",
        "distance_m": 0.5,
        "condition": "quiet",
        "level": "norm",
        "metrics": dict(METRICS),
        "text": None,
        "duplicate": False,
        "warnings": [],
    }


class TokenTest(unittest.TestCase):
    def test_normalizes_latin(self) -> None:
        self.assertEqual(rec.token(" Anna-01 ", "--speaker"), "anna-01")

    def test_rejects_non_latin(self) -> None:
        with self.assertRaises(ValueError):
            rec.token("Анна", "--speaker")


class PlanTest(unittest.TestCase):
    def test_matrix_has_24_cells(self) -> None:
        plan = rec.build_plan("anna", "laptop", rec.DEFAULTS, 3, False, 0)
        self.assertEqual(len(plan), 24)
        self.assertEqual({cell["condition"] for cell in plan}, {"quiet", "music", "tv"})
        self.assertEqual({cell["distance_m"] for cell in plan}, {0.5, 1.0, 1.5, 2.0})
        self.assertTrue(all(cell["target"] == 3 for cell in plan))

    def test_extra_block(self) -> None:
        plan = rec.build_plan("anna", "laptop", rec.DEFAULTS, 2, True, 0)
        self.assertEqual(len(plan), 28)
        extra = [cell for cell in plan if cell["phrase"] == "эй луна"]
        self.assertEqual(len(extra), 4)
        self.assertTrue(all(cell["target"] == 2 for cell in extra))

    def test_order_is_stable_by_seed(self) -> None:
        first = rec.build_plan("anna", "laptop", rec.DEFAULTS, 3, False, 7)
        second = rec.build_plan("anna", "laptop", rec.DEFAULTS, 3, False, 7)
        other = rec.build_plan("tim", "laptop", rec.DEFAULTS, 3, False, 7)
        self.assertEqual([cell["id"] for cell in first], [cell["id"] for cell in second])
        self.assertNotEqual([cell["id"] for cell in first], [cell["id"] for cell in other])

    def test_names(self) -> None:
        self.assertEqual(
            rec.make_take_name("anna", "laptop", CELL, 3, "хей луна"),
            "anna_laptop_d05_quiet_norm_003.wav",
        )
        extra = dict(CELL, phrase="эй луна")
        self.assertEqual(
            rec.make_take_name("anna", "laptop", extra, 1, "хей луна"),
            "anna_laptop_ey_d05_quiet_norm_001.wav",
        )


class SessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "positives_real"
        (self.root / "raw").mkdir(parents=True)
        self.session = {
            "speaker": "anna",
            "mic": "laptop",
            "settings": {"phrase": "хей луна"},
            "takes": [],
            "plan": [dict(CELL)],
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_next_take_skips_existing_files(self) -> None:
        take, name = rec.next_take(self.root, self.session, CELL)
        self.assertEqual(take, 1)
        rec.write_wav(self.root / "raw" / name, np.zeros(160, dtype=np.int16), 16000)
        take, name = rec.next_take(self.root, self.session, CELL)
        self.assertEqual(take, 2)
        self.assertEqual(name, "anna_laptop_d05_quiet_norm_002.wav")

    def test_refresh_cells_and_next_cell(self) -> None:
        self.session["takes"] = [
            {"cell": CELL["id"], "status": "accepted"},
            {"cell": CELL["id"], "status": "rejected"},
        ]
        rec.refresh_cells(self.session)
        cell = self.session["plan"][0]
        self.assertEqual(cell["accepted"], 1)
        self.assertEqual(cell["attempts"], 2)
        self.assertEqual(cell["status"], "pending")
        self.assertEqual(rec.next_cell(self.session)["id"], CELL["id"])

    def test_find_latest_ignores_backups(self) -> None:
        sessions = self.root / "sessions"
        sessions.mkdir()
        active = sessions / "anna_laptop.json"
        backup = sessions / "anna_laptop.20260101-000000.bak.json"
        active.write_text(json.dumps({"takes": []}), encoding="utf-8")
        backup.write_text(json.dumps({"takes": []}), encoding="utf-8")
        self.assertEqual(rec.find_latest_session(self.root), active)

    def test_load_sessions_keeps_backups(self) -> None:
        sessions = self.root / "sessions"
        sessions.mkdir()
        active = sessions / "anna_laptop.json"
        backup = sessions / "anna_laptop.20260101-000000.bak.json"
        active.write_text(json.dumps({"takes": []}), encoding="utf-8")
        backup.write_text(json.dumps({"takes": []}), encoding="utf-8")
        loaded = rec.load_sessions(self.root)
        self.assertEqual({path.name for path, _ in loaded}, {active.name, backup.name})

    def test_resume_can_clear_limit(self) -> None:
        sessions = self.root / "sessions"
        sessions.mkdir()
        path = sessions / "anna_laptop.json"
        session = {
            "speaker": "anna",
            "mic": "laptop",
            "status": "paused",
            "limit": 20,
            "settings": {},
            "plan": [dict(CELL)],
            "takes": [],
        }
        path.write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")
        with patch.object(rec, "run_session", return_value=0) as run:
            self.assertEqual(rec.resume_latest(self.root, None, None, 0), 0)
        self.assertEqual(run.call_args.args[0]["limit"], 0)

    def test_accepted_hashes_include_previous_sessions(self) -> None:
        sessions = self.root / "sessions"
        sessions.mkdir()
        old = sessions / "anna_laptop.20260101-000000.bak.json"
        old.write_text(
            json.dumps(
                {
                    "takes": [
                        {"status": "accepted", "pcm_md5": "aaa"},
                        {"status": "rejected", "pcm_md5": "bbb"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(rec.accepted_hashes(self.root), {"aaa"})


class AudioTest(unittest.TestCase):
    def test_trim_and_pad(self) -> None:
        rate = 16000
        pcm = np.zeros(rate, dtype=np.int16)
        pcm[4000:8000] = 12000
        trimmed = rec.trim_speech(pcm, rate, -40.0, 0.0)
        self.assertEqual(len(trimmed), 4000)
        padded = rec.pad_speech(trimmed, rate, 0.1, 0.2)
        self.assertEqual(len(padded), 4000 + 1600 + 3200)
        self.assertTrue(np.all(padded[:1600] == 0))

    def test_resample_changes_rate(self) -> None:
        source = np.arange(48000, dtype=np.int16)
        result = rec.resample_audio(source, 48000, 16000)
        self.assertEqual(len(result), 16000)
        self.assertEqual(result.dtype, np.int16)

    def test_metrics_and_clipping(self) -> None:
        pcm = np.zeros(16000, dtype=np.int16)
        pcm[1000:5000] = 32767
        metrics = rec.pcm_metrics(pcm, 16000, -40.0, 0.0)
        self.assertGreaterEqual(metrics["clipped"], 4000)
        self.assertEqual(metrics["duration_s"], 1.0)
        self.assertGreater(metrics["peak_db"], -1.0)

    def test_asr_hints_resample_to_16k(self) -> None:
        class Fake:
            def __init__(self) -> None:
                self.sizes: list[int] = []

            def transcribe(self, pcm: bytes) -> str:
                self.sizes.append(len(pcm))
                return "хей луна"

        fake = Fake()
        hints = rec.AsrHints(True)
        hints._transcriber = fake
        text = hints.text(np.zeros(48000, dtype=np.int16), 48000)
        self.assertEqual(text, "хей луна")
        self.assertEqual(fake.sizes, [32000])

    def test_level_hint(self) -> None:
        self.assertIn("тихо", rec.level_hint({"peak_db": -25.0, "clipped": 0}))
        self.assertIn("громко", rec.level_hint({"peak_db": -2.0, "clipped": 0}))
        self.assertIn("клиппинг", rec.level_hint({"peak_db": -1.0, "clipped": 10}))
        self.assertIn("норм", rec.level_hint({"peak_db": -8.0, "clipped": 0}))


class FinalizeTest(unittest.TestCase):
    def test_manifest_and_listen_page(self) -> None:
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "positives_real"
            (root / "raw").mkdir(parents=True)
            (root / "sessions").mkdir()
            pcm = np.zeros(16000, dtype=np.int16)
            pcm[8000:12000] = 9000
            rec.write_wav(root / "raw" / "anna_laptop_d05_quiet_norm_001.wav", pcm, 16000)
            session = {
                "speaker": "anna",
                "mic": "laptop",
                "settings": rec.DEFAULTS,
                "takes": [
                    {
                        "cell": "quiet_d05_norm",
                        "take": 1,
                        "file": "raw/anna_laptop_d05_quiet_norm_001.wav",
                        "status": "accepted",
                        "phrase": "хей луна",
                        "distance_m": 0.5,
                        "condition": "quiet",
                        "level": "norm",
                        "timestamp": "2026-11-07T12:00:00",
                    }
                ],
            }
            (root / "sessions" / "anna_laptop.json").write_text(
                json.dumps(session, ensure_ascii=False), encoding="utf-8"
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(rec.finalize(root, no_stt=True), 0)
            manifest = (root / "manifest.csv").read_text(encoding="utf-8")
            self.assertIn("anna_laptop_d05_quiet_norm_001.wav", manifest)
            self.assertIn("processed/anna_laptop/", manifest)
            self.assertTrue((root / "qc_files.csv").exists())
            self.assertTrue((root / "qc_report.txt").exists())
            self.assertTrue((root / "listen_positives.html").exists())


class ClassificationTest(unittest.TestCase):
    def test_exact(self) -> None:
        self.assertEqual(rec.classify("Хей, луна!"), (True, True, True))

    def test_extra_phrase_is_not_exact(self) -> None:
        self.assertEqual(rec.classify("эй луна"), (True, True, False))

    def test_missing_luna(self) -> None:
        self.assertEqual(rec.classify("хей"), (False, True, False))


class WarningsTest(unittest.TestCase):
    def test_duplicate_and_asr_are_warnings(self) -> None:
        metrics = rec.pcm_metrics(np.full(16000, 12000, dtype=np.int16), 16000, -40.0, 0.0)
        warnings = rec.evaluate_take(metrics, "привет", True, rec.DEFAULTS)
        self.assertTrue(any("повтор" in item for item in warnings))
        self.assertTrue(any("луна" in item for item in warnings))


class ExclusionTest(unittest.TestCase):
    def test_roundtrip_with_comment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exclude.txt"
            rec.append_exclusion(path, "processed/a/one.wav", "шум  с улицы")
            rec.append_exclusion(path, "processed/a/two.wav")
            self.assertEqual(
                rec.read_exclusions(path), {"processed/a/one.wav", "processed/a/two.wav"}
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("# шум с улицы", text)


class UIHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "positives_real"
        (self.root / "raw").mkdir(parents=True)
        (self.root / "sessions").mkdir()
        self.session = _session()
        self.path = self.root / "sessions" / "anna_laptop.json"
        self.pcm = np.zeros(48000, dtype=np.int16)
        self.pcm[8000:40000] = 8000

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_prepare_take_pending_until_decision(self) -> None:
        with patch.object(rec, "capture_take", return_value=(self.pcm, 16000)):
            pending = rec.prepare_take(
                self.root, self.session, self.session["plan"][0], None, None, set()
            )
        self.assertEqual(pending["take"], 1)
        self.assertEqual(pending["file"], "raw/anna_laptop_d05_quiet_norm_001.wav")
        self.assertTrue((self.root / pending["file"]).exists())
        self.assertEqual(self.session["takes"], [])
        self.assertEqual(self.session["plan"][0]["accepted"], 0)

        hashes: set[str] = set()
        rec.save_session(self.path, self.session)
        rec.apply_take_decision(self.session, self.path, pending, "accepted", "", hashes)
        self.assertEqual(self.session["takes"][0]["status"], "accepted")
        self.assertEqual(self.session["plan"][0]["accepted"], 1)
        self.assertIn(pending["metrics"]["pcm_md5"], hashes)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["takes"][0]["status"], "accepted")
        self.assertEqual(saved["status"], "complete")

    def test_run_status_complete_and_limit(self) -> None:
        session = _session()
        rec.register_take(session, session["plan"][0], 1, "raw/a.wav", "accepted", dict(METRICS))
        self.assertEqual(rec.refresh_run_status(session), "complete")

        limited = _session(target=3)
        limited["limit"] = 1
        rec.register_take(
            limited, limited["plan"][0], 1, "raw/b.wav", "accepted", dict(METRICS, pcm_md5="other")
        )
        self.assertEqual(rec.refresh_run_status(limited), "paused")

    def test_snapshot_isolated_per_pair(self) -> None:
        anna = _session("anna", "laptop")
        tim = _session("tim", "studio")
        rec.register_take(anna, anna["plan"][0], 1, "raw/a.wav", "accepted", dict(METRICS))
        snap_anna = rec.session_snapshot(anna)
        snap_tim = rec.session_snapshot(tim)
        self.assertEqual(snap_anna["speaker"], "anna")
        self.assertEqual(snap_anna["accepted"], 1)
        self.assertEqual(snap_anna["cells_done"], 1)
        self.assertEqual(snap_tim["speaker"], "tim")
        self.assertEqual(snap_tim["accepted"], 0)
        self.assertEqual(snap_tim["cells_done"], 0)


class UIStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "positives_real"
        (self.root / "sessions").mkdir(parents=True)
        (self.root / "raw").mkdir()
        self.path_a = self.root / "sessions" / "anna_laptop.json"
        self.path_b = self.root / "sessions" / "tim_studio.json"
        rec.save_session(self.path_a, _session("anna", "laptop"))
        rec.save_session(self.path_b, _session("tim", "studio"))
        self.state = rec.UIState(
            self.root,
            [(self.path_a, rec.load_json(self.path_a)), (self.path_b, rec.load_json(self.path_b))],
            self.path_a,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_select_switches_pair(self) -> None:
        self.state.select("tim", "studio")
        self.assertEqual(self.state.selected_pair(), {"speaker": "tim", "mic": "studio"})

    def test_pending_blocks_select_until_dropped(self) -> None:
        self.state.pending = _pending()
        with self.assertRaises(RuntimeError):
            self.state.select("tim", "studio")
        self.state.drop()
        self.assertIsNone(self.state.pending)
        self.state.select("tim", "studio")
        self.assertEqual(self.state.selected_pair()["speaker"], "tim")

    def test_decide_writes_only_selected_pair(self) -> None:
        self.state.pending = _pending()
        self.state.decide("accept")
        anna = rec.load_json(self.path_a)
        tim = rec.load_json(self.path_b)
        self.assertEqual(anna["takes"][0]["status"], "accepted")
        self.assertEqual(anna["plan"][0]["accepted"], 1)
        self.assertEqual(tim["takes"], [])


class UIServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "positives_real"
        (self.root / "sessions").mkdir(parents=True)
        (self.root / "raw").mkdir()
        self.path_a = self.root / "sessions" / "anna_laptop.json"
        self.path_b = self.root / "sessions" / "tim_studio.json"
        anna = _session("anna", "laptop")
        rec.register_take(anna, anna["plan"][0], 1, "raw/a.wav", "accepted", dict(METRICS))
        rec.save_session(self.path_a, anna)
        rec.save_session(self.path_b, _session("tim", "studio"))
        self.state = rec.UIState(
            self.root,
            [(self.path_a, rec.load_json(self.path_a)), (self.path_b, rec.load_json(self.path_b))],
            self.path_a,
        )
        self.server = rec._UIHTTPServer(("127.0.0.1", 0), self.state)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base + path, data=data, method="POST", headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_state_shows_only_selected_pair(self) -> None:
        data = self._get("/api/state")
        self.assertEqual(data["selected"], {"speaker": "anna", "mic": "laptop"})
        self.assertEqual(data["session"]["speaker"], "anna")
        self.assertEqual(data["session"]["accepted"], 1)
        self.assertEqual(data["session"]["cells_done"], 1)
        self.assertEqual(len(data["pairs"]), 2)
        self.assertIsNone(data["pending"])

    def test_switch_hides_previous_pair_progress(self) -> None:
        before = self._get("/api/state")
        self.assertEqual(before["session"]["accepted"], 1)
        data = self._post("/api/select", {"speaker": "tim", "mic": "studio"})
        self.assertEqual(data["session"]["speaker"], "tim")
        self.assertEqual(data["session"]["accepted"], 0)
        self.assertEqual(data["session"]["cells_done"], 0)
        again = self._get("/api/state")
        self.assertEqual(again["session"]["accepted"], 0)

    def test_page_contains_pair_selector(self) -> None:
        with urllib.request.urlopen(self.base + "/", timeout=5) as response:
            page = response.read().decode("utf-8")
        self.assertIn('id="pair"', page)
        self.assertIn("Пара диктор × микрофон", page)
        self.assertIn("/api/state", page)

    def test_select_switches_pair(self) -> None:
        data = self._post("/api/select", {"speaker": "tim", "mic": "studio"})
        self.assertEqual(data["selected"]["speaker"], "tim")
        self.assertEqual(data["session"]["mic"], "studio")
        again = self._get("/api/state")
        self.assertEqual(again["session"]["speaker"], "tim")

    def test_pending_blocks_select_until_dropped(self) -> None:
        self.state.pending = _pending()
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._post("/api/select", {"speaker": "tim", "mic": "studio"})
        self.assertEqual(caught.exception.code, 409)
        self._post("/api/stop", {})
        self.assertIsNone(self.state.pending)
        data = self._post("/api/select", {"speaker": "tim", "mic": "studio"})
        self.assertEqual(data["selected"]["speaker"], "tim")

    def test_audio_serves_wav_and_blocks_escape(self) -> None:
        rec.write_wav(self.root / "raw" / "a.wav", np.zeros(160, dtype=np.int16), 16000)
        with urllib.request.urlopen(self.base + "/audio/raw/a.wav", timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read()[:4], b"RIFF")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.base + "/audio/../secret.wav", timeout=5)
        self.assertEqual(caught.exception.code, 400)


class CLITest(unittest.TestCase):
    def test_ui_is_default(self) -> None:
        args = rec.parse_args([])
        self.assertFalse(args.console)
        self.assertEqual(args.ui_port, rec.DEFAULT_UI_PORT)
        self.assertFalse(args.no_browser)
        self.assertIsNone(args.mic)

    def test_console_and_port(self) -> None:
        args = rec.parse_args(["--console", "--ui-port", "9000"])
        self.assertTrue(args.console)
        self.assertEqual(args.ui_port, 9000)


if __name__ == "__main__":
    unittest.main()
