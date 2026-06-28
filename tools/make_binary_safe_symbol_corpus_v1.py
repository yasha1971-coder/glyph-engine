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


def main() -> int:
    ap = argparse.ArgumentParser(description="Create binary-safe symbol corpus V1.")
    ap.add_argument("--input", required=True, help="Raw source corpus, arbitrary bytes allowed.")
    ap.add_argument("--symbols-out", required=True, help="Output little-endian u16 symbol stream.")
    ap.add_argument("--manifest-out", required=True, help="Output manifest JSON.")
    args = ap.parse_args()

    inp = Path(args.input).resolve()
    sym_out = Path(args.symbols_out).resolve()
    manifest_out = Path(args.manifest_out).resolve()

    data = inp.read_bytes()

    symbols = list(data)
    symbols.append(SENTINEL_SYMBOL)

    sym_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    with sym_out.open("wb") as f:
        for x in symbols:
            if not (0 <= x <= SENTINEL_SYMBOL):
                raise RuntimeError(f"symbol out of range: {x}")
            f.write(struct.pack("<H", x))

    nul_count = data.count(b"\x00")

    manifest = {
        "format": FORMAT,
        "source_corpus": {
            "path": str(inp),
            "bytes": len(data),
            "sha256": sha256_file(inp),
            "nul_bytes": nul_count,
        },
        "symbol_stream": {
            "path": str(sym_out),
            "encoding": "u16le",
            "symbols": len(symbols),
            "bytes": sym_out.stat().st_size,
            "sha256": sha256_file(sym_out),
        },
        "alphabet": {
            "data_min": 0,
            "data_max": 255,
            "virtual_sentinel": SENTINEL_SYMBOL,
            "sentinel_is_corpus_byte": False,
        },
        "invariants": {
            "symbol_count_equals_source_bytes_plus_one": len(symbols) == len(data) + 1,
            "exactly_one_virtual_sentinel": symbols.count(SENTINEL_SYMBOL) == 1,
            "nul_byte_preserved_as_data_symbol_0": nul_count == sum(1 for x in symbols[:-1] if x == 0),
            "sentinel_is_last_symbol": symbols[-1] == SENTINEL_SYMBOL,
            "all_source_symbols_are_0_to_255": all(0 <= x <= 255 for x in symbols[:-1]),
        },
    }

    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "ok": all(manifest["invariants"].values()),
        "format": FORMAT,
        "source_bytes": len(data),
        "nul_bytes": nul_count,
        "symbols": len(symbols),
        "sentinel": SENTINEL_SYMBOL,
        "symbols_out": str(sym_out),
        "manifest_out": str(manifest_out),
    }, indent=2, sort_keys=True))

    return 0 if all(manifest["invariants"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
