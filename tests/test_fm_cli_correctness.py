#!/usr/bin/env python3
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "tools" / "build_glyph_index_v1.sh"
QUERY_BIN = ROOT / "build" / "query_fm_v1"


def oracle_count(corpus: bytes, pattern: bytes) -> int:
    count = 0
    start = 0
    while True:
        pos = corpus.find(pattern, start)
        if pos == -1:
            return count
        count += 1
        start = pos + 1


def run_query_count(corpus: bytes, pattern: bytes) -> int:
    if b"\x00" in corpus:
        raise ValueError("v0.x corpus must not contain 0x00")

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        corpus_path = tdp / "corpus.bin"
        corpus_path.write_bytes(corpus)

        subprocess.run(
            [str(BUILD_SCRIPT), str(corpus_path), str(tdp)],
            cwd=str(ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        pattern_hex = pattern.hex()
        out = subprocess.run(
            [str(QUERY_BIN), str(tdp / "fm.bin"), str(tdp / "bwt.bin"), pattern_hex],
            cwd=str(ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout

        m = re.search(r"count:\s+(\d+)", out)
        if not m:
            raise AssertionError(f"Could not parse count from output:\n{out}")
        return int(m.group(1))


class TestFMCLICorrectness(unittest.TestCase):
    def assert_count_matches_oracle(self, corpus: bytes, pattern: bytes):
        expected = oracle_count(corpus, pattern)
        got = run_query_count(corpus, pattern)
        self.assertEqual(got, expected, f"corpus={corpus!r} pattern={pattern!r}")

    def test_single_occurrence(self):
        self.assert_count_matches_oracle(b"hello world", b"world")

    def test_multiple_occurrences(self):
        self.assert_count_matches_oracle(b"the cat sat on the mat the cat", b"the")
        self.assert_count_matches_oracle(b"the cat sat on the mat the cat", b"cat")

    def test_absent_pattern(self):
        self.assert_count_matches_oracle(b"abcdef", b"zzz")

    def test_full_corpus_as_pattern(self):
        self.assert_count_matches_oracle(b"exactmatch", b"exactmatch")

    def test_single_byte(self):
        self.assert_count_matches_oracle(b"aaabbbccc", b"a")
        self.assert_count_matches_oracle(b"aaabbbccc", b"z")

    def test_overlapping_occurrences(self):
        self.assert_count_matches_oracle(b"aaaa", b"aa")
        self.assert_count_matches_oracle(b"aaaa", b"aaa")

    def test_terminal_sentinel_present_once(self):
        got = run_query_count(b"no null bytes here", b"\x00")
        self.assertEqual(got, 1)

    def test_repeated_determinism(self):
        corpus = b"the cat sat on the mat the cat"
        pattern = b"cat"
        r1 = run_query_count(corpus, pattern)
        r2 = run_query_count(corpus, pattern)
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
