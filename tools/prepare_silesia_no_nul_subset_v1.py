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
    n = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            n += chunk.count(b"\x00")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    src_dir = Path(args.dir).resolve()
    out = Path(args.out).resolve()
    manifest = Path(args.manifest).resolve()

    if not src_dir.is_dir():
        raise SystemExit(f"not a directory: {src_dir}")

    files = [p for p in sorted(src_dir.rglob("*")) if p.is_file()]
    included = []
    skipped = []

    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    with out.open("wb") as fout:
        for p in files:
            nul = count_nul(p)
            rec = {
                "path": str(p),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "nul_bytes": nul,
            }

            if nul != 0:
                skipped.append(rec)
                continue

            sep = f"\n---SILESIA_FILE:{p.name}---\n".encode("utf-8")
            fout.write(sep)
            fout.write(p.read_bytes())
            included.append(rec)

    if not included:
        raise SystemExit("no no-NUL Silesia files found")

    meta = {
        "format": "GLYPH_SILESIA_NO_NUL_SUBSET_V1",
        "source_dir": str(src_dir),
        "out": str(out),
        "out_bytes": out.stat().st_size,
        "out_sha256": sha256_file(out),
        "included_count": len(included),
        "skipped_count": len(skipped),
        "included": included,
        "skipped": skipped,
        "boundary": "This is a sentinel-safe no-NUL subset, not the full Silesia corpus.",
    }

    manifest.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "out": str(out),
        "out_bytes": meta["out_bytes"],
        "included_count": len(included),
        "skipped_count": len(skipped),
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
