#!/usr/bin/env python3
import re
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BUILD_INDEX = ROOT / "tools" / "build_glyph_index_v1.sh"
BUILD_LOCATE = ROOT / "tools" / "build_locate_fixture_v1.py"

QUERY_FM = ROOT / "build" / "query_fm_v1"
LOCATE_BACKEND = ROOT / "build" / "locate_backend_v2"


def oracle_offsets(corpus: bytes, pattern: bytes):
    out = []
    start = 0
    while True:
        i = corpus.find(pattern, start)
        if i == -1:
            break
        out.append(i)
        start = i + 1
    return out


def parse_interval(stdout: str):
    m = re.search(r"interval:\s*\[(\d+),\s*(\d+)\)", stdout)
    if not m:
        raise AssertionError(f"could not parse interval from:\n{stdout}")
    return int(m.group(1)), int(m.group(2))


def build_fixture(corpus: bytes):
    td = tempfile.TemporaryDirectory()
    tdp = Path(td.name)

    raw = tdp / "corpus.bin"
    raw.write_bytes(corpus)

    subprocess.run(
        [str(BUILD_INDEX), str(raw), str(tdp)],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    subprocess.run(
        [
            "python3",
            str(BUILD_LOCATE),
            "--bwt", str(tdp / "bwt.bin"),
            "--sa", str(tdp / "sa.bin"),
            "--out-dir", str(tdp),
        ],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return td, tdp


def fm_interval(index_dir: Path, pattern: bytes):
    out = subprocess.run(
        [
            str(QUERY_FM),
            str(index_dir / "fm.bin"),
            str(index_dir / "bwt.bin"),
            pattern.hex(),
        ],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return parse_interval(out.stdout)


def locate_interval(index_dir: Path, l: int, r: int):
    proc = subprocess.Popen(
        [
            str(LOCATE_BACKEND),
            str(index_dir / "fm_core.bin"),
            str(index_dir / "locate_core_s16.bin"),
            str(index_dir / "bwt.bin"),
        ],
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    req = bytearray()
    req += b"REQ1"
    req += struct.pack("<I", 1)
    req += struct.pack("<Q", l)
    req += struct.pack("<Q", r)

    proc.stdin.write(req)
    proc.stdin.flush()
    proc.stdin.close()

    def read_exact(n):
        data = proc.stdout.read(n)
        if len(data) != n:
            raise AssertionError(f"short read: wanted {n}, got {len(data)}")
        return data

    magic = read_exact(4)
    if magic != b"RES1":
        raise AssertionError(f"bad locate response magic: {magic!r}")

    num_ranges = struct.unpack("<I", read_exact(4))[0]
    if num_ranges != 1:
        raise AssertionError(f"expected 1 range, got {num_ranges}")

    count = struct.unpack("<Q", read_exact(8))[0]
    _total_steps = struct.unpack("<Q", read_exact(8))[0]
    max_steps = struct.unpack("<Q", read_exact(8))[0]
    offsets = [struct.unpack("<Q", read_exact(8))[0] for _ in range(count)]

    stderr = proc.stderr.read().decode("utf-8", errors="replace")
    ret = proc.wait()

    if proc.stdout:
        proc.stdout.close()
    if proc.stderr:
        proc.stderr.close()
    if ret != 0:
        raise AssertionError(f"locate_backend_v2 failed ret={ret}\nstderr={stderr}")

    if max_steps > 16:
        raise AssertionError(f"max locate steps {max_steps} exceeds sample step 16")

    return sorted(offsets)


class TestLocateVerify(unittest.TestCase):

    def assert_verified_locate(self, corpus: bytes, pattern: bytes):
        td, index_dir = build_fixture(corpus)
        try:
            l, r = fm_interval(index_dir, pattern)
            count = r - l
            offsets = locate_interval(index_dir, l, r) if count else []
            oracle = oracle_offsets(corpus, pattern)

            self.assertEqual(count, len(offsets))
            self.assertEqual(offsets, oracle)

            for o in offsets:
                self.assertNotEqual(o, len(corpus), "sentinel position leaked into offsets")
                self.assertEqual(corpus[o:o + len(pattern)], pattern)
        finally:
            td.cleanup()

    def test_locate_single_occurrence(self):
        self.assert_verified_locate(b"hello world", b"world")

    def test_locate_multiple_occurrences(self):
        self.assert_verified_locate(b"the cat sat on the mat the cat", b"the")
        self.assert_verified_locate(b"the cat sat on the mat the cat", b"cat")

    def test_locate_absent_pattern(self):
        self.assert_verified_locate(b"abcdef", b"zzz")

    def test_locate_overlapping_occurrences(self):
        self.assert_verified_locate(b"aaaa", b"aa")
        self.assert_verified_locate(b"aaaa", b"aaa")

    def test_locate_single_byte(self):
        self.assert_verified_locate(b"aaabbbccc", b"a")
        self.assert_verified_locate(b"aaabbbccc", b"z")

    def test_locate_full_corpus_pattern(self):
        corpus = b"exactmatch"
        self.assert_verified_locate(corpus, corpus)


if __name__ == "__main__":
    unittest.main(verbosity=2)
