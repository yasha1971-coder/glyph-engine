import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.write_sa_container_v1 import write_container
from tools.read_sa_container_v1 import read_header


class TestSAContainerReaderV1(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _valid_container(self):
        sa = self.tmp / "sa.bin"
        out = self.tmp / "sa_v1.bin"
        sa.write_bytes(struct.pack("<III", 2, 0, 1))
        write_container(sa, out, corpus_bytes=3, entry_width=4)
        return out

    def test_read_valid_header(self):
        out = self._valid_container()
        info = read_header(out)

        self.assertEqual(info["magic"], "GLYPHSA1")
        self.assertEqual(info["version"], 1)
        self.assertEqual(info["entry_width"], 4)
        self.assertEqual(info["corpus_bytes"], 3)
        self.assertEqual(info["sa_entries"], 3)
        self.assertEqual(info["endian"], 1)
        self.assertEqual(info["file_size"], 52)

    def test_reject_bad_magic(self):
        out = self._valid_container()
        data = bytearray(out.read_bytes())
        data[0:8] = b"BADMAGIC"
        out.write_bytes(data)

        with self.assertRaises(RuntimeError):
            read_header(out)

    def test_reject_bad_version(self):
        out = self._valid_container()
        data = bytearray(out.read_bytes())
        data[8:12] = struct.pack("<I", 999)
        out.write_bytes(data)

        with self.assertRaises(RuntimeError):
            read_header(out)

    def test_reject_bad_entry_width(self):
        out = self._valid_container()
        data = bytearray(out.read_bytes())
        data[12:16] = struct.pack("<I", 3)
        out.write_bytes(data)

        with self.assertRaises(RuntimeError):
            read_header(out)

    def test_reject_file_size_mismatch(self):
        out = self._valid_container()
        out.write_bytes(out.read_bytes()[:-1])

        with self.assertRaises(RuntimeError):
            read_header(out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
