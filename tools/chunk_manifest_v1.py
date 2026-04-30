import argparse
import json
import os
import struct
import zlib
from pathlib import Path


def load_chunks(glyph_path):
    chunks = []
    with open(glyph_path, "rb") as f:
        magic = f.read(6)
        if magic != b"GLYPH1":
            raise ValueError(f"Bad glyph magic: {magic!r}")
        _version, ng = struct.unpack("<BH", f.read(3))
        for _ in range(ng):
            sz = struct.unpack("<I", f.read(4))[0]
            chunks.append(zlib.decompress(f.read(sz)))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--out-root", required=True,
                    help="Chunk root, e.g. /home/glyph/GLYPH_CPP_BACKEND/chunk_core")
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional limit on number of chunks; 0 means all")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    manifests_dir = out_root / "manifests"
    chunks_dir = out_root / "chunks"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifests_dir / "chunk_manifest_v1.jsonl"

    chunks = load_chunks(args.glyph)
    if args.limit > 0:
        chunks = chunks[:args.limit]

    print("=" * 60)
    print("  CHUNK MANIFEST V1")
    print("=" * 60)
    print(f"  glyph={args.glyph}")
    print(f"  out_root={out_root}")
    print(f"  chunks={len(chunks)}")
    print(f"  manifest={manifest_path}")

    pos = 0
    total_bytes = 0

    with open(manifest_path, "w", encoding="utf-8") as out:
        for chunk_id, ch in enumerate(chunks):
            raw_len = len(ch)
            start = pos
            end = pos + raw_len

            chunk_dir = chunks_dir / f"chunk_{chunk_id:06d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)

            rec = {
                "chunk_id": chunk_id,
                "source_glyph": args.glyph,
                "raw_len": raw_len,
                "global_start": start,
                "global_end": end,
                "chunk_dir": str(chunk_dir),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            pos = end
            total_bytes += raw_len

    print(f"  total_bytes={total_bytes}")
    print("  done")


if __name__ == "__main__":
    main()