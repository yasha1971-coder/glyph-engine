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
TOOL = HERE / "verified_reversible_precompression_archive.py"


FAKE_PRECOMP = r'''#!/usr/bin/env python3
import sys
from pathlib import Path

restore = "-r" in sys.argv
output = next(arg[2:] for arg in sys.argv if arg.startswith("-o"))
source = Path(sys.argv[-1]).read_bytes()
if restore:
    if not source.startswith(b"PCF"):
        raise SystemExit(1)
    Path(output).write_bytes(b"FAKEJPEG" + source[3:])
    raise SystemExit(0)
if not source.startswith(b"FAKEJPEG"):
    raise SystemExit(2)
Path(output).write_bytes(b"PCF" + source[8:])
'''


BAD_PRECOMP = r'''#!/usr/bin/env python3
import sys
from pathlib import Path

output = next(arg[2:] for arg in sys.argv if arg.startswith("-o"))
if "-r" in sys.argv:
    Path(output).write_bytes(b"wrong")
else:
    Path(output).write_bytes(b"PCFcandidate")
raise SystemExit(0)
'''


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


def make_precomp(root: Path, body: str = FAKE_PRECOMP) -> tuple[Path, str]:
    path = root / "precomp"
    path.write_text(body)
    path.chmod(0o755)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


class VerifiedReversiblePrecompressionArchiveTests(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run(
            [sys.executable, str(TOOL), *map(str, args)],
            text=True, capture_output=True,
        )

    def setup_case(self, body: str = FAKE_PRECOMP):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        source = root / "source"
        source.mkdir()
        precomp, digest = make_precomp(root, body)
        return temporary, root, source, precomp, digest

    def build(self, root, source, precomp, digest):
        inventory = root / "inventory"
        archive = root / "archive"
        make_inventory(source, inventory)
        result = self.run_tool(
            "build", "--inventory-state", inventory, "--output", archive,
            "--precomp", precomp, "--precomp-sha256", digest,
        )
        return result, archive

    def test_verified_candidate_is_selected_and_restores_exact_bytes(self):
        temporary, root, source, precomp, digest = self.setup_case()
        with temporary:
            payload = b"FAKEJPEG" + os.urandom(4096)
            (source / "image.jpeg").write_bytes(payload)
            result, archive = self.build(root, source, precomp, digest)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((archive / "GLYPH_VERIFIED_REVERSIBLE_PRECOMPRESSION_ARCHIVE_V1.json").read_text())
            self.assertEqual(receipt["codec_object_counts"], {"precomp-cn": 1})
            self.assertEqual(receipt["precompression_attempt_counts"], {"verified": 1})
            restored = root / "restored"
            result = self.run_tool(
                "restore", "--archive", archive, "--destination", restored,
                "--precomp", precomp,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((restored / "image.jpeg").read_bytes(), payload)

    def test_noneligible_object_skips_external_tool(self):
        temporary, root, source, precomp, digest = self.setup_case()
        with temporary:
            (source / "notes.txt").write_bytes(b"GLYPH\n" * 1000)
            result, archive = self.build(root, source, precomp, digest)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((archive / "GLYPH_VERIFIED_REVERSIBLE_PRECOMPRESSION_ARCHIVE_V1.json").read_text())
            self.assertEqual(receipt["precompression_attempt_counts"], {"not-eligible": 1})
            self.assertNotIn("precomp-cn", receipt["codec_object_counts"])

    def test_no_supported_stream_falls_back_safely(self):
        temporary, root, source, precomp, digest = self.setup_case()
        with temporary:
            (source / "opaque.png").write_bytes(os.urandom(4096))
            result, archive = self.build(root, source, precomp, digest)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((archive / "GLYPH_VERIFIED_REVERSIBLE_PRECOMPRESSION_ARCHIVE_V1.json").read_text())
            self.assertEqual(receipt["precompression_attempt_counts"], {"no-supported-stream": 1})
            self.assertNotIn("precomp-cn", receipt["codec_object_counts"])

    def test_roundtrip_mismatch_is_never_selected(self):
        temporary, root, source, precomp, digest = self.setup_case(BAD_PRECOMP)
        with temporary:
            (source / "image.jpeg").write_bytes(b"FAKEJPEG" + os.urandom(4096))
            result, archive = self.build(root, source, precomp, digest)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((archive / "GLYPH_VERIFIED_REVERSIBLE_PRECOMPRESSION_ARCHIVE_V1.json").read_text())
            self.assertEqual(receipt["precompression_attempt_counts"], {"roundtrip-mismatch": 1})
            self.assertNotIn("precomp-cn", receipt["codec_object_counts"])

    def test_wrong_binary_hash_is_rejected_without_output(self):
        temporary, root, source, precomp, digest = self.setup_case()
        with temporary:
            (source / "image.jpeg").write_bytes(b"FAKEJPEGdata")
            inventory, archive = root / "inventory", root / "archive"
            make_inventory(source, inventory)
            result = self.run_tool(
                "build", "--inventory-state", inventory, "--output", archive,
                "--precomp", precomp, "--precomp-sha256", "0" * 64,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("SHA-256 mismatch", result.stderr)
            self.assertFalse(archive.exists())

    def test_corrupted_object_is_rejected_without_restore(self):
        temporary, root, source, precomp, digest = self.setup_case()
        with temporary:
            (source / "image.jpeg").write_bytes(b"FAKEJPEG" + os.urandom(4096))
            result, archive = self.build(root, source, precomp, digest)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((archive / "GLYPH_VERIFIED_REVERSIBLE_PRECOMPRESSION_ARCHIVE_V1.json").read_text())
            object_path = archive / receipt["objects"][0]["path"]
            object_path.write_bytes(object_path.read_bytes() + b"damage")
            restored = root / "restored"
            result = self.run_tool(
                "restore", "--archive", archive, "--destination", restored,
                "--precomp", precomp,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("stored object mismatch", result.stderr)
            self.assertFalse(restored.exists())


if __name__ == "__main__":
    unittest.main()
