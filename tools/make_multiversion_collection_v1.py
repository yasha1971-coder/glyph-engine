#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_nul(path: Path) -> int:
    total = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            total += chunk.count(b"\x00")
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--version-mib", type=int, default=64)
    ap.add_argument("--shift-mib", type=int, default=8)
    ap.add_argument("--versions", type=int, default=4)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    src = Path(args.source).resolve()
    out = Path(args.out).resolve()
    manifest = Path(args.manifest).resolve()

    if not src.exists():
        raise SystemExit(f"missing source: {src}")

    version_bytes = args.version_mib * 1024 * 1024
    shift_bytes = args.shift_mib * 1024 * 1024

    need = (args.versions - 1) * shift_bytes + version_bytes
    size = src.stat().st_size
    if size < need:
        raise SystemExit(f"source too small: need {need}, have {size}")

    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    records = []
    out_pos = 0

    with src.open("rb") as fsrc, out.open("wb") as fout:
        for i in range(args.versions):
            start = i * shift_bytes
            fsrc.seek(start)
            data = fsrc.read(version_bytes)
            if len(data) != version_bytes:
                raise SystemExit(f"short read at version {i}")

            sep = f"\n---GLYPH_VERSION_BOUNDARY_{args.label}_{i:04d}---\n".encode("ascii")
            fout.write(sep)
            sep_start = out_pos
            out_pos += len(sep)

            data_start = out_pos
            fout.write(data)
            out_pos += len(data)

            records.append({
                "version": i,
                "source_start": start,
                "source_bytes": len(data),
                "collection_separator_start": sep_start,
                "collection_data_start": data_start,
            })

    nul = count_nul(out)

    meta = {
        "format": "GLYPH_MULTIVERSION_COLLECTION_V1",
        "label": args.label,
        "source": str(src),
        "source_bytes": size,
        "source_sha256": sha256_file(src),
        "out": str(out),
        "out_bytes": out.stat().st_size,
        "out_sha256": sha256_file(out),
        "nul_bytes": nul,
        "version_mib": args.version_mib,
        "shift_mib": args.shift_mib,
        "versions": args.versions,
        "records": records,
        "boundary_note": "This is for BWT run-ratio measurement only. Single-sentinel GLYPH retrieval would require boundary-aware collection semantics to prevent cross-version matches."
    }

    manifest.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "label": args.label,
        "out": str(out),
        "manifest": str(manifest),
        "out_bytes": meta["out_bytes"],
        "nul_bytes": nul,
        "source": str(src),
        "versions": args.versions,
        "version_mib": args.version_mib,
        "shift_mib": args.shift_mib,
    }, indent=2))

    if nul != 0:
        raise SystemExit("collection contains NUL bytes; current sentinel-safe builder may reject it")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
