import fcntl
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
CHUNK_TOOL = HERE / "chunk_truth_manifest.py"
BASELINE_TOOL = HERE / "dedup_baseline.py"


def make_inventory(source: Path, state: Path) -> None:
    state.mkdir()
    with sqlite3.connect(state / "inventory.sqlite3") as db:
        db.execute(
            "CREATE TABLE entries(path TEXT,kind TEXT,logical_bytes INTEGER,"
            "mtime_ns INTEGER,device INTEGER,inode INTEGER)"
        )
        for path in sorted(source.iterdir()):
            info = path.stat()
            db.execute(
                "INSERT INTO entries VALUES(?,?,?,?,?,?)",
                (path.name, "file", info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino),
            )
    receipt = {
        "format": "GLYPH_1TB_CORPUS_TRUTH_GATE_V1",
        "source": str(source.resolve()),
        "complete": True,
        "traversal_finished": True,
        "inventory": {"errors": 0},
    }
    (state / "GLYPH_1TB_CORPUS_TRUTH_RECEIPT_V1.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )


class DedupBaselineTests(unittest.TestCase):
    def setup_case(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        source, inventory, state = root / "source", root / "inventory", root / "state"
        source.mkdir()
        return temporary, source, inventory, state

    def run_chunk(self, inventory, state, *extra):
        return subprocess.run(
            [sys.executable, str(CHUNK_TOOL), "--inventory-state", str(inventory),
             "--state", str(state), "--chunk-bytes", "4096", *extra],
            text=True, capture_output=True,
        )

    def run_baseline(self, state):
        return subprocess.run(
            [sys.executable, str(BASELINE_TOOL), "--state", str(state)],
            text=True, capture_output=True,
        )

    def test_exact_duplicates_are_measured_and_receipt_is_deterministic(self):
        temporary, source, inventory, state = self.setup_case()
        with temporary:
            payload = b"A" * 4096 + b"B" * 4096
            (source / "a.bin").write_bytes(payload)
            (source / "b.bin").write_bytes(payload)
            (source / "empty").write_bytes(b"")
            make_inventory(source, inventory)
            self.assertEqual(self.run_chunk(inventory, state).returncode, 0)
            first = self.run_baseline(state)
            self.assertEqual(first.returncode, 0, first.stderr)
            path = state / "GLYPH_DEDUP_BASELINE_V1.json"
            raw = path.read_bytes()
            result = json.loads(raw)
            self.assertEqual(result["logical_bytes_total"], len(payload) * 2)
            self.assertEqual(result["exact_duplicate_file_groups"], 1)
            self.assertEqual(result["exact_duplicate_file_references"], 1)
            self.assertEqual(result["exact_duplicate_file_bytes"], len(payload))
            self.assertEqual(result["chunk_references"], 4)
            self.assertEqual(result["unique_chunk_objects"], 2)
            self.assertEqual(result["proven_aligned_duplicate_bytes"], len(payload))
            self.assertEqual(result["aligned_chunk_storage_ratio"], 0.5)
            self.assertFalse(result["claim_compression"])
            expected = hashlib.sha256(raw).hexdigest()
            self.assertEqual(
                path.with_suffix(".json.sha256").read_text(),
                f"{expected}  {path.name}\n",
            )
            second = self.run_baseline(state)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(path.read_bytes(), raw)

    def test_shifted_content_is_not_misreported_as_fixed_chunk_dedup(self):
        temporary, source, inventory, state = self.setup_case()
        with temporary:
            payload = b"".join(
                hashlib.sha256(index.to_bytes(4, "big")).digest()
                for index in range(256)
            )
            (source / "aligned.bin").write_bytes(payload)
            (source / "shifted.bin").write_bytes(b"X" + payload)
            make_inventory(source, inventory)
            self.assertEqual(self.run_chunk(inventory, state).returncode, 0)
            result = self.run_baseline(state)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(
                (state / "GLYPH_DEDUP_BASELINE_V1.json").read_text()
            )
            self.assertEqual(receipt["proven_aligned_duplicate_bytes"], 0)
            self.assertEqual(receipt["aligned_chunk_storage_ratio"], 1.0)
            self.assertFalse(receipt["claim_shifted_content_dedup"])

    def test_incomplete_chunk_state_is_rejected(self):
        temporary, source, inventory, state = self.setup_case()
        with temporary:
            (source / "a.bin").write_bytes(b"A" * 8192)
            make_inventory(source, inventory)
            self.assertEqual(
                self.run_chunk(inventory, state, "--max-chunks", "1").returncode, 75
            )
            result = self.run_baseline(state)
            self.assertEqual(result.returncode, 2)
            self.assertIn("completed error-free", result.stderr)
            self.assertFalse((state / "GLYPH_DEDUP_BASELINE_V1.json").exists())

    def test_receipt_or_database_tampering_is_rejected(self):
        for mutation in ("receipt", "rechecksummed_receipt", "database"):
            with self.subTest(mutation=mutation):
                temporary, source, inventory, state = self.setup_case()
                with temporary:
                    (source / "a.bin").write_bytes(b"A" * 8192)
                    make_inventory(source, inventory)
                    self.assertEqual(self.run_chunk(inventory, state).returncode, 0)
                    if mutation == "receipt":
                        path = state / "GLYPH_CHUNK_TRUTH_RECEIPT_V1.json"
                        path.write_bytes(path.read_bytes().replace(b'"complete": true', b'"complete":false'))
                    elif mutation == "rechecksummed_receipt":
                        path = state / "GLYPH_CHUNK_TRUTH_RECEIPT_V1.json"
                        receipt = json.loads(path.read_text())
                        receipt["aligned_chunk_storage_ratio"] = 0.0
                        raw = (
                            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
                            + "\n"
                        ).encode("utf-8")
                        path.write_bytes(raw)
                        path.with_suffix(".json.sha256").write_text(
                            f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n"
                        )
                    else:
                        with sqlite3.connect(state / "chunk-truth.sqlite3") as db:
                            db.execute("UPDATE chunks SET sha256=? WHERE ordinal=0", ("0" * 64,))
                    result = self.run_baseline(state)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertFalse((state / "GLYPH_DEDUP_BASELINE_V1.json").exists())

    def test_busy_chunk_state_is_rejected_without_output(self):
        temporary, source, inventory, state = self.setup_case()
        with temporary:
            (source / "a.bin").write_bytes(b"A")
            make_inventory(source, inventory)
            self.assertEqual(self.run_chunk(inventory, state).returncode, 0)
            with (state / ".chunk-truth.lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = self.run_baseline(state)
                self.assertEqual(result.returncode, 75)
                self.assertIn("state is busy", result.stderr)
                self.assertFalse((state / "GLYPH_DEDUP_BASELINE_V1.json").exists())


if __name__ == "__main__":
    unittest.main()
