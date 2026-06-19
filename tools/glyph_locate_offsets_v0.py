#!/usr/bin/env python3
import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def locate_backend_path() -> Path:
    candidates = [
        ROOT / "build" / "locate_backend_v2",
        ROOT / "build" / "build.saved.verify-test" / "locate_backend_v2",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError("locate_backend_v2 not found")


def run_locate(index_dir: Path, l: int, r: int):
    backend = locate_backend_path()

    cmd = [
        str(backend),
        str(index_dir / "fm_core.bin"),
        str(index_dir / "locate_core_s16.bin"),
        str(index_dir / "bwt.bin"),
    ]

    proc = subprocess.Popen(
        cmd,
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

    out, err = proc.communicate(bytes(req))

    if proc.returncode != 0:
        raise RuntimeError(err.decode("utf-8", errors="replace"))

    off = 0
    magic = out[off:off + 4]
    off += 4
    if magic != b"RES1":
        raise ValueError(f"bad magic: {magic!r}")

    num_ranges = struct.unpack("<I", out[off:off + 4])[0]
    off += 4
    if num_ranges != 1:
        raise ValueError(f"expected 1 range, got {num_ranges}")

    count = struct.unpack("<Q", out[off:off + 8])[0]
    off += 8
    total_steps = struct.unpack("<Q", out[off:off + 8])[0]
    off += 8
    max_steps = struct.unpack("<Q", out[off:off + 8])[0]
    off += 8

    offsets = []
    for _ in range(count):
        offsets.append(struct.unpack("<Q", out[off:off + 8])[0])
        off += 8

    return {
        "count": count,
        "total_steps": total_steps,
        "max_steps": max_steps,
        "offsets": sorted(offsets),
        "offset_mode": "locate_backend_v2",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Locate GLYPH FM interval offsets.")
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--l", type=int, required=True)
    ap.add_argument("--r", type=int, required=True)
    args = ap.parse_args()

    index_dir = Path(args.index_dir).resolve()

    required = [
        index_dir / "fm_core.bin",
        index_dir / "locate_core_s16.bin",
        index_dir / "bwt.bin",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print(json.dumps({
            "ok": False,
            "error": "missing_locate_files",
            "missing": missing,
        }, indent=2))
        return 2

    try:
        result = run_locate(index_dir, args.l, args.r)
    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": str(e),
        }, indent=2))
        return 1

    result["ok"] = True
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
