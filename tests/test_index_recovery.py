#!/usr/bin/env python3
"""Oracle roundtrips and hostile RLB1/primary-index gates; no private inputs."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from recover_search_index_v1 import recover
from rlbwt_container_v1 import encode_bwt, write_varint


def bwt(raw):
    text = raw + b"\0"
    sa = sorted(range(len(text)), key=lambda i: text[i:])
    return bytes(text[(i - 1) % len(text)] for i in sa)


def main():
    helper = Path(sys.argv[1]).resolve()
    cases = [b"x", b"a" * 257, b"ab" * 129, b"banana", b"line\r\nline\n ",
             bytes(range(1, 256)), bytes(range(255, 0, -1)), b"mississippi",
             b"aabaaaac", b"ba", b"ab", b"\xff\x01\xfe\x02"]
    passed, rejected, primary = 0, 0, set()
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        for i, data in enumerate(cases):
            flat = bwt(data)
            primary.add(flat.index(0))
            (d / "bwt").write_bytes(flat)
            with contextlib.redirect_stdout(io.StringIO()):
                encode_bwt(d / "bwt", d / "rlb")
            out = d / ("out" + str(i))
            recover(d / "rlb", len(data), hashlib.sha256(data).hexdigest(), out, helper, d)
            assert out.read_bytes() == data
            passed += 1
        assert 1 in primary and 257 in primary  # primary=1 and primary=n exercised

        def reject(flat=None, encoded=None, size=6, expected=None):
            nonlocal rejected
            if encoded is None:
                (d / "bwt").write_bytes(flat)
                with contextlib.redirect_stdout(io.StringIO()):
                    encode_bwt(d / "bwt", d / "rlb")
            else:
                (d / "rlb").write_bytes(encoded)
            try:
                recover(d / "rlb", size, expected or "0" * 64, d / "bad", helper, d)
            except (ValueError, EOFError, subprocess.CalledProcessError):
                assert not (d / "bad").exists()
                rejected += 1
            else:
                raise AssertionError("invalid input accepted")

        reject(flat=b"abcdefg")                 # no sentinel
        reject(flat=b"ab\0d\0fg")              # two sentinels
        reject(flat=b"\0abcdef")               # primary zero
        reject(flat=bwt(b"banana"), size=5)     # wrong dimensions
        reject(flat=bwt(b"banana"))             # wrong expected corpus SHA
        reject(flat=b"\0", size=0)             # unsupported empty corpus
        (d / "bwt").write_bytes(bwt(b"banana"))
        with contextlib.redirect_stdout(io.StringIO()):
            encode_bwt(d / "bwt", d / "rlb")
        good = (d / "rlb").read_bytes()
        reject(encoded=good + b"extra")
        reject(encoded=good[:-1])
        reject(encoded=good[:24] + b"x" * 32 + good[56:])
        reject(encoded=b"NOPE" + good[4:])
        reject(encoded=good[:8] + struct.pack("<Q", 8) + good[16:])
        for runs, tail in [(0, b""), (8, b""), (1, b"a\0"),
                           (1, b"a\x08"), (1, b"a" + b"\x80" * 10),
                           (2, b"a\x06b\x02")]:
            reject(encoded=struct.pack("<4sIQQ", b"RLB1", 1, 7, runs) + b"x" * 32 + tail)
    print(json.dumps(dict(ok=True, roundtrips=passed, rejected=rejected,
                          primary_1_and_n=True, corpus_or_query_read=False)))


if __name__ == "__main__":
    main()
