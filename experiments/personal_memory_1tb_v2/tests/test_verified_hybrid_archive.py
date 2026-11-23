import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
TOOL = HERE / "verified_hybrid_archive.py"


def make_inventory(source: Path, state: Path) -> None:
    state.mkdir()
    with sqlite3.connect(state / "inventory.sqlite3") as db:
        db.execute(
            "CREATE TABLE entries(path TEXT,kind TEXT,logical_bytes INTEGER,"
            "mtime_ns INTEGER,device INTEGER,inode INTEGER)"
        )
        for path in sorted(source.rglob("*")):
            info = path.lstat()
            kind = "directory" if path.is_dir() else "file"
            db.execute(
                "INSERT INTO entries VALUES(?,?,?,?,?,?)",
                (
                    path.relative_to(source).as_posix(), kind,
                    info.st_size if kind == "file" else 0,
                    info.st_mtime_ns, info.st_dev, info.st_ino,
                ),
            )
    receipt = {
        "format": "GLYPH_1TB_CORPUS_TRUTH_GATE_V1",
        "source": str(source.resolve()),
        "complete": True,
        "inventory": {"errors": 0},
    }
    path = state / "GLYPH_1TB_CORPUS_TRUTH_RECEIPT_V1.json"
    raw = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    path.with_suffix(".json.sha256").write_text(
        f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n"
    )


class VerifiedHybridArchiveTests(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run(
            [sys.executable, str(TOOL), *map(str, args)],
            text=True, capture_output=True,
        )

    def setup_case(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        source = root / "source"
        source.mkdir()
        return temporary, root, source

    def test_build_deduplicates_routes_and_restores_exact_bytes(self):
        temporary, root, source = self.setup_case()
        with temporary:
            (source / "nested").mkdir()
            payload = (b"GLYPH exact memory\n" * 4000)
            (source / "nested" / "a.txt").write_bytes(payload)
            (source / "duplicate.txt").write_bytes(payload)
            (source / "empty").write_bytes(b"")
            inventory, archive, restored = root / "inventory", root / "archive", root / "restored"
            make_inventory(source, inventory)
            built = self.run_tool("build", "--inventory-state", inventory, "--output", archive)
            self.assertEqual(built.returncode, 0, built.stderr)
            receipt = json.loads((archive / "GLYPH_VERIFIED_HYBRID_ARCHIVE_V1.json").read_text())
            self.assertEqual(receipt["files_total"], 3)
            self.assertEqual(receipt["unique_objects"], 2)
            self.assertEqual(receipt["duplicate_file_references"], 1)
            self.assertTrue(receipt["byte_perfect_object_roundtrip_verified"])
            self.assertTrue(receipt["target_met"])
            result = self.run_tool("restore", "--archive", archive, "--destination", restored)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((restored / "nested" / "a.txt").read_bytes(), payload)
            self.assertEqual((restored / "duplicate.txt").read_bytes(), payload)
            self.assertEqual((restored / "empty").read_bytes(), b"")

    def test_incompressible_object_uses_raw(self):
        temporary, root, source = self.setup_case()
        with temporary:
            data = os.urandom(32768)
            (source / "random.bin").write_bytes(data)
            inventory, archive = root / "inventory", root / "archive"
            make_inventory(source, inventory)
            result = self.run_tool("build", "--inventory-state", inventory, "--output", archive)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((archive / "GLYPH_VERIFIED_HYBRID_ARCHIVE_V1.json").read_text())
            self.assertEqual(receipt["codec_object_counts"], {"raw": 1})

    def test_source_identity_change_is_rejected_without_archive(self):
        temporary, root, source = self.setup_case()
        with temporary:
            path = source / "a"
            path.write_bytes(b"before")
            inventory, archive = root / "inventory", root / "archive"
            make_inventory(source, inventory)
            path.write_bytes(b"changed length")
            result = self.run_tool("build", "--inventory-state", inventory, "--output", archive)
            self.assertEqual(result.returncode, 2)
            self.assertIn("source identity changed", result.stderr)
            self.assertFalse(archive.exists())

    def test_corrupted_object_is_rejected_without_restore(self):
        temporary, root, source = self.setup_case()
        with temporary:
            (source / "a").write_bytes(b"A" * 10000)
            inventory, archive, restored = root / "inventory", root / "archive", root / "restored"
            make_inventory(source, inventory)
            self.assertEqual(self.run_tool("build", "--inventory-state", inventory, "--output", archive).returncode, 0)
            receipt = json.loads((archive / "GLYPH_VERIFIED_HYBRID_ARCHIVE_V1.json").read_text())
            obj = archive / receipt["objects"][0]["path"]
            obj.write_bytes(obj.read_bytes() + b"damage")
            result = self.run_tool("restore", "--archive", archive, "--destination", restored)
            self.assertEqual(result.returncode, 2)
            self.assertIn("stored object mismatch", result.stderr)
            self.assertFalse(restored.exists())

    def test_receipt_corruption_and_existing_destinations_are_rejected(self):
        temporary, root, source = self.setup_case()
        with temporary:
            (source / "a").write_bytes(b"A" * 10000)
            inventory, archive = root / "inventory", root / "archive"
            make_inventory(source, inventory)
            self.assertEqual(self.run_tool("build", "--inventory-state", inventory, "--output", archive).returncode, 0)
            duplicate = self.run_tool("build", "--inventory-state", inventory, "--output", archive)
            self.assertEqual(duplicate.returncode, 2)
            receipt = archive / "GLYPH_VERIFIED_HYBRID_ARCHIVE_V1.json"
            receipt.write_bytes(receipt.read_bytes() + b" ")
            restored = root / "restored"
            result = self.run_tool("restore", "--archive", archive, "--destination", restored)
            self.assertEqual(result.returncode, 2)
            self.assertIn("checksum mismatch", result.stderr)
            self.assertFalse(restored.exists())


if __name__ == "__main__":
    unittest.main()
