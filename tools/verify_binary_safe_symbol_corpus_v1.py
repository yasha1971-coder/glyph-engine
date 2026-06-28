#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path


SENTINEL_SYMBOL = 256
FORMAT = "GLYPH_BINARY_SAFE_SYMBOL_CORPUS_V1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_u16le(path: Path):
    data = path.read_bytes()
    if len(data) % 2 != 0:
        raise RuntimeError("symbol stream byte length is not divisible by 2")
    return [x[0] for x in struct.iter_unpack("<H", data)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify binary-safe symbol corpus V1.")
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    j = json.loads(manifest_path.read_text())

    errors = []

    def check(cond, msg):
        if not cond:
            errors.append(msg)

    check(j.get("format") == FORMAT, "bad format")

    source_path = Path(j["source_corpus"]["path"])
    symbols_path = Path(j["symbol_stream"]["path"])

    source = source_path.read_bytes()
    symbols = read_u16le(symbols_path)

    check(j["source_corpus"]["bytes"] == len(source), "source byte length mismatch")
    check(j["source_corpus"]["sha256"] == sha256_file(source_path), "source sha256 mismatch")
    check(j["source_corpus"]["nul_bytes"] == source.count(b"\x00"), "source nul count mismatch")

    check(j["symbol_stream"]["symbols"] == len(symbols), "symbol count mismatch")
    check(j["symbol_stream"]["bytes"] == symbols_path.stat().st_size, "symbol byte length mismatch")
    check(j["symbol_stream"]["sha256"] == sha256_file(symbols_path), "symbol sha256 mismatch")

    check(len(symbols) == len(source) + 1, "symbols != source bytes + 1")
    check(symbols[-1] == SENTINEL_SYMBOL, "last symbol is not sentinel 256")
    check(symbols.count(SENTINEL_SYMBOL) == 1, "sentinel count is not exactly one")

    for i, b in enumerate(source):
        if symbols[i] != b:
            errors.append(f"source byte not preserved at offset {i}: source={b}, symbol={symbols[i]}")
            break

    check(all(0 <= x <= 255 for x in symbols[:-1]), "source symbols outside 0..255")
    check(source.count(b"\x00") == sum(1 for x in symbols[:-1] if x == 0), "0x00 not preserved as data symbol 0")

    inv = j.get("invariants", {})
    for k, v in inv.items():
        check(v is True, f"manifest invariant false: {k}")

    result = {
        "ok": not errors,
        "manifest": str(manifest_path),
        "source_bytes": len(source),
        "nul_bytes": source.count(b"\x00"),
        "symbols": len(symbols),
        "sentinel": symbols[-1] if symbols else None,
        "errors": errors,
    }

    print(json.dumps(result, indent=2, sort_keys=True))

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
