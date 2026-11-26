"""Exact inverse of GLYPH's sentinel BWT. No new format or third-party library."""
import contextlib
import hashlib
import io
import struct
import subprocess
import tempfile
from pathlib import Path

from rlbwt_container_v1 import decode_rlbwt

MAX_BYTES = 128 * 1024 * 1024


def recover(rlbwt, size, expected_corpus_sha256, output, helper, scratch):
    if type(size) is not int or not 1 <= size <= MAX_BYTES:
        raise ValueError("unsupported corpus length")
    with Path(rlbwt).open("rb") as f:
        header = f.read(56)
    if len(header) != 56:
        raise ValueError("truncated RLB1 header")
    magic, version, n, runs = struct.unpack("<4sIQQ", header[:24])
    if magic != b"RLB1" or version != 1 or n != size + 1 or not 1 <= runs <= n:
        raise ValueError("RLB1 dimensions differ from index")
    with tempfile.TemporaryDirectory(dir=scratch, prefix="recover-") as t:
        raw, body = Path(t) / "bwt", Path(t) / "body"
        with contextlib.redirect_stdout(io.StringIO()):
            decode_rlbwt(Path(rlbwt), raw)
        subprocess.run([str(helper), str(raw), str(body)], check=True)
        # One snapshot supplies both the pin check and the returned bytes.
        data = body.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != expected_corpus_sha256:
            raise ValueError("recovered corpus SHA-256 or length mismatch")
        with Path(output).open("xb") as f:
            f.write(data)
    return {"ok": True, "corpus_bytes": size, "corpus_sha256": expected_corpus_sha256}
