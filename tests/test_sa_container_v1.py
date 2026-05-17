import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.write_sa_container_v1 import write_container


class TestSAContainerV1(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_valid_sa32_container(self):
        sa = self.tmp / "sa.bin"
        out = self.tmp / "sa_v1.bin"

        values = [2, 0, 1]
        sa.write_bytes(struct.pack("<III", *values))

        info = write_container(
            sa_path=sa,
            out_path=out,
            corpus_bytes=3,
            entry_width=4,
        )

        self.assertEqual(info["magic"], "GLYPHSA1")
        self.assertEqual(info["version"], 1)
        self.assertEqual(info["entry_width"], 4)
        self.assertEqual(info["corpus_bytes"], 3)
        self.assertEqual(info["sa_entries"], 3)
        self.assertEqual(info["output_bytes"], 40 + 3 * 4)

        data = out.read_bytes()
        self.assertEqual(data[:8], b"GLYPHSA1")
        self.assertEqual(struct.unpack("<I", data[8:12])[0], 1)
        self.assertEqual(struct.unpack("<I", data[12:16])[0], 4)
        self.assertEqual(struct.unpack("<Q", data[16:24])[0], 3)
        self.assertEqual(struct.unpack("<Q", data[24:32])[0], 3)

    def test_reject_empty_sa(self):
        sa = self.tmp / "empty.sa"
        out = self.tmp / "sa_v1.bin"
        sa.write_bytes(b"")

        with self.assertRaises(RuntimeError):
            write_container(sa, out, corpus_bytes=0, entry_width=4)

    def test_reject_bad_entry_width(self):
        sa = self.tmp / "sa.bin"
        out = self.tmp / "sa_v1.bin"
        sa.write_bytes(struct.pack("<I", 0))

        with self.assertRaises(RuntimeError):
            write_container(sa, out, corpus_bytes=1, entry_width=3)

    def test_reject_size_not_divisible(self):
        sa = self.tmp / "bad.sa"
        out = self.tmp / "sa_v1.bin"
        sa.write_bytes(b"abc")

        with self.assertRaises(RuntimeError):
            write_container(sa, out, corpus_bytes=1, entry_width=4)

    def test_reject_corpus_entries_mismatch(self):
        sa = self.tmp / "sa.bin"
        out = self.tmp / "sa_v1.bin"
        sa.write_bytes(struct.pack("<II", 0, 1))

        with self.assertRaises(RuntimeError):
            write_container(sa, out, corpus_bytes=3, entry_width=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
