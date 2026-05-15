#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.verify_manifest_v1 import verify


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verified GLYPH query: manifest check + direct FM query."
    )
    ap.add_argument("index_dir")
    ap.add_argument("pattern")
    ap.add_argument("--hex", action="store_true")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)

    verify(index_dir)

    pattern_hex = args.pattern if args.hex else args.pattern.encode("utf-8").hex()

    cmd = [
        str(ROOT / "build" / "query_fm_v1"),
        str(index_dir / "fm.bin"),
        str(index_dir / "bwt.bin"),
        pattern_hex,
    ]

    r = subprocess.run(cmd, cwd=str(ROOT))
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
