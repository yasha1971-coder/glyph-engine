import hashlib
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
TOOL = HERE / "chunk_truth_manifest.py"
SPEC = importlib.util.spec_from_file_location("chunk_truth", TOOL)
chunk_truth = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chunk_truth)


def make_inventory(source: Path, state: Path) -> None:
    state.mkdir()
    db = sqlite3.connect(state / "inventory.sqlite3")
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
    db.commit()
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


class ChunkTruthManifestTests(unittest.TestCase):
    def setup_case(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        source, inventory, state = root / "source", root / "inventory", root / "state"
        source.mkdir()
        return temporary, source, inventory, state

    def run_tool(self, inventory, state, *extra):
        return subprocess.run(
            [sys.executable, str(TOOL), "--inventory-state", str(inventory),
             "--state", str(state), "--chunk-bytes", "4096", *extra],
            text=True, capture_output=True,
        )

    def test_duplicate_chunks_are_proven(self):
        temporary, source, inventory, state = self.setup_case()
        with temporary:
            payload = b"A" * 4096 + b"B" * 4096
            (source / "a.bin").write_bytes(payload)
            (source / "b.bin").write_bytes(payload)
            make_inventory(source, inventory)
            result = self.run_tool(inventory, state)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((state / "GLYPH_CHUNK_TRUTH_RECEIPT_V1.json").read_text())
            self.assertTrue(receipt["complete"])
            self.assertEqual(receipt["total_chunks"], 4)
            self.assertEqual(receipt["unique_chunks"], 2)
            self.assertEqual(receipt["proven_aligned_duplicate_bytes"], len(payload))
            self.assertEqual(receipt["aligned_chunk_storage_ratio"], 0.5)

    def test_resume_inside_file_is_idempotent(self):
        temporary, source, inventory, state = self.setup_case()
        with temporary:
            (source / "large.bin").write_bytes(os.urandom(4096 * 3))
            make_inventory(source, inventory)
            first = self.run_tool(inventory, state, "--max-chunks", "1")
            self.assertEqual(first.returncode, chunk_truth.INCOMPLETE, first.stderr)
            db = sqlite3.connect(state / "chunk-truth.sqlite3")
            self.assertEqual(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 1)
            db.close()
            second = self.run_tool(inventory, state)
            self.assertEqual(second.returncode, 0, second.stderr)
            root1 = json.loads((state / "GLYPH_CHUNK_TRUTH_RECEIPT_V1.json").read_text())["content_manifest_root_sha256"]
            third = self.run_tool(inventory, state)
            self.assertEqual(third.returncode, 0, third.stderr)
            root2 = json.loads((state / "GLYPH_CHUNK_TRUTH_RECEIPT_V1.json").read_text())["content_manifest_root_sha256"]
            self.assertEqual(root1, root2)

    def test_source_change_after_inventory_fails_closed(self):
        temporary, source, inventory, state = self.setup_case()
        with temporary:
            path = source / "mutable.bin"
            path.write_bytes(b"x" * 8192)
            make_inventory(source, inventory)
            path.write_bytes(b"changed")
            result = self.run_tool(inventory, state)
            self.assertEqual(result.returncode, chunk_truth.UNTRUSTED)
            receipt = json.loads((state / "GLYPH_CHUNK_TRUTH_RECEIPT_V1.json").read_text())
            self.assertFalse(receipt["complete"])
            self.assertEqual(receipt["errors"], 1)

    def test_corrupt_checkpoint_fails_closed(self):
        temporary, source, inventory, state = self.setup_case()
        with temporary:
            (source / "a.bin").write_bytes(b"a" * 8192)
            make_inventory(source, inventory)
            first = self.run_tool(inventory, state, "--max-chunks", "1")
            self.assertEqual(first.returncode, chunk_truth.INCOMPLETE)
            db = sqlite3.connect(state / "chunk-truth.sqlite3")
            db.execute("UPDATE chunks SET offset=1 WHERE ordinal=0")
            db.commit(); db.close()
            result = self.run_tool(inventory, state)
            self.assertEqual(result.returncode, chunk_truth.UNTRUSTED)
            db = sqlite3.connect(state / "chunk-truth.sqlite3")
            error = db.execute("SELECT error FROM files WHERE path='a.bin'").fetchone()[0]
            db.close()
            self.assertIn("non-contiguous", error)

    def test_valid_hex_digest_corruption_fails_closed(self):
        for completed in (False, True):
            with self.subTest(completed=completed):
                temporary, source, inventory, state = self.setup_case()
                with temporary:
                    (source / "a.bin").write_bytes(b"a" * 8192)
                    make_inventory(source, inventory)
                    extra = () if completed else ("--max-chunks", "1")
                    first = self.run_tool(inventory, state, *extra)
                    self.assertEqual(first.returncode, 0 if completed else 75)
                    with sqlite3.connect(state / "chunk-truth.sqlite3") as db:
                        db.execute("UPDATE chunks SET sha256=? WHERE ordinal=0", ("0" * 64,))
                    result = self.run_tool(inventory, state)
                    self.assertEqual(result.returncode, chunk_truth.UNTRUSTED, result.stdout)
                    receipt = json.loads((state / "GLYPH_CHUNK_TRUTH_RECEIPT_V1.json").read_text())
                    self.assertFalse(receipt["complete"])
                    self.assertEqual(receipt["errors"], 1)

    def test_completed_checkpoint_fields_fail_closed(self):
        for sql in (
            "DELETE FROM chunks WHERE ordinal=1",
            "UPDATE files SET content_root_sha256='wrong'",
            "UPDATE files SET next_offset=0",
            "UPDATE files SET chunk_count=0",
        ):
            with self.subTest(sql=sql):
                temporary, source, inventory, state = self.setup_case()
                with temporary:
                    (source / "a.bin").write_bytes(b"a" * 8192)
                    make_inventory(source, inventory)
                    self.assertEqual(self.run_tool(inventory, state).returncode, 0)
                    with sqlite3.connect(state / "chunk-truth.sqlite3") as db:
                        db.execute(sql)
                    result = self.run_tool(inventory, state)
                    self.assertEqual(result.returncode, chunk_truth.UNTRUSTED, result.stdout)

    def test_changed_bytes_with_restored_mtime_fail_closed(self):
        temporary, source, inventory, state = self.setup_case()
        with temporary:
            path = source / "a.bin"
            path.write_bytes(b"a" * 8192)
            make_inventory(source, inventory)
            self.assertEqual(self.run_tool(inventory, state, "--max-chunks", "1").returncode, 75)
            before = path.stat()
            with path.open("r+b") as stream:
                stream.write(b"b")
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            self.assertEqual(self.run_tool(inventory, state).returncode, chunk_truth.UNTRUSTED)

    def test_empty_file_and_tail_remain_unchanged(self):
        temporary, source, inventory, state = self.setup_case()
        with temporary:
            payloads = {"empty": b"", "tail": b"a" * 4096 + b"tail"}
            for name, data in payloads.items():
                (source / name).write_bytes(data)
            make_inventory(source, inventory)
            for _ in range(2):
                self.assertEqual(self.run_tool(inventory, state).returncode, 0)
            for name, data in payloads.items():
                self.assertEqual((source / name).read_bytes(), data)

    def test_state_inside_source_is_rejected(self):
        temporary, source, inventory, _ = self.setup_case()
        with temporary:
            (source / "a.bin").write_bytes(b"a")
            make_inventory(source, inventory)
            result = self.run_tool(inventory, source / "state")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be disjoint", result.stderr)

    def test_receipt_checksum_matches(self):
        temporary, source, inventory, state = self.setup_case()
        with temporary:
            (source / "a.bin").write_bytes(b"abc")
            make_inventory(source, inventory)
            result = self.run_tool(inventory, state)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = state / "GLYPH_CHUNK_TRUTH_RECEIPT_V1.json"
            expected = hashlib.sha256(receipt.read_bytes()).hexdigest()
            self.assertEqual(receipt.with_suffix(".json.sha256").read_text().split()[0], expected)


if __name__ == "__main__":
    unittest.main()
