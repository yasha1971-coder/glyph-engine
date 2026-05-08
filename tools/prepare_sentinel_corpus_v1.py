#!/usr/bin/env python3
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(
        description="Prepare corpus for GLYPH FM-index by appending a real unique 0x00 sentinel."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    data = inp.read_bytes()

    if b"\x00" in data:
        raise SystemExit(
            "ERROR: input contains 0x00. v0.x sentinel mode requires corpus without zero bytes. "
            "Use a 257-symbol alphabet implementation for arbitrary raw bytes."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data + b"\x00")

    print("input:", inp)
    print("output:", out)
    print("input_bytes:", len(data))
    print("output_bytes:", len(data) + 1)
    print("sentinel_appended: 0x00")

if __name__ == "__main__":
    main()
