import argparse
import json
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


def build_clean_corpus(raw: bytes) -> bytes:
    return bytes((b + 1) for b in raw) + b"\x00"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional limit on number of manifest records to process; 0 means all")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    if args.limit > 0:
        records = records[:args.limit]

    # group by source glyph so we only load each glyph once
    glyph_cache = {}

    print("=" * 60)
    print("  CHUNK INDEX PREPARE V1")
    print("=" * 60)
    print(f"  manifest={manifest_path}")
    print(f"  records={len(records)}")

    total_raw = 0
    total_clean = 0

    for rec in records:
        chunk_id = rec["chunk_id"]
        source_glyph = rec["source_glyph"]
        chunk_dir = Path(rec["chunk_dir"])
        chunk_dir.mkdir(parents=True, exist_ok=True)

        if source_glyph not in glyph_cache:
            glyph_cache[source_glyph] = load_chunks(source_glyph)

        glyph_chunks = glyph_cache[source_glyph]
        raw = glyph_chunks[chunk_id]
        clean = build_clean_corpus(raw)

        raw_path = chunk_dir / "chunk.raw.bin"
        clean_path = chunk_dir / "chunk.clean.corpus.bin"
        meta_path = chunk_dir / "chunk.meta.json"

        with open(raw_path, "wb") as f:
            f.write(raw)

        with open(clean_path, "wb") as f:
            f.write(clean)

        meta = {
            "chunk_id": chunk_id,
            "source_glyph": source_glyph,
            "raw_len": len(raw),
            "clean_len": len(clean),
            "global_start": rec["global_start"],
            "global_end": rec["global_end"],
            "chunk_dir": str(chunk_dir),
            "raw_path": str(raw_path),
            "clean_corpus_path": str(clean_path),
            "sentinel": 0,
            "byte_shift": 1,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        total_raw += len(raw)
        total_clean += len(clean)

    print(f"  total_raw_bytes={total_raw}")
    print(f"  total_clean_bytes={total_clean}")
    print("  done")


if __name__ == "__main__":
    main()