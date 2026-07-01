#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

VERSION = "GLYPH_STRUCTURAL_FINGERPRINT_V0"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def entropy_from_counts(counts, n):
    if n == 0:
        return 0.0
    e = 0.0
    for c in counts.values():
        if c:
            p = c / n
            e -= p * math.log2(p)
    return e

def byte_stats(path: Path):
    counts = Counter()
    n = 0
    nul = 0
    newline = 0
    printable = 0

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            n += len(chunk)
            nul += chunk.count(b"\x00")
            newline += chunk.count(b"\n")
            for b in chunk:
                counts[b] += 1
                if b in (9, 10, 13) or 32 <= b <= 126:
                    printable += 1

    top = [
        {"byte": b, "hex": f"{b:02x}", "count": c, "fraction": c / n if n else 0.0}
        for b, c in counts.most_common(16)
    ]

    return {
        "bytes": n,
        "alphabet_size": len(counts),
        "nul_bytes": nul,
        "newline_bytes": newline,
        "printable_fraction": printable / n if n else 0.0,
        "entropy_bits_per_byte": entropy_from_counts(counts, n),
        "top_bytes": top,
    }

def chunk_entropy_profile(path: Path, chunk_size: int):
    values = []
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            counts = Counter(chunk)
            values.append(entropy_from_counts(counts, len(chunk)))

    if not values:
        return {
            "chunk_size": chunk_size,
            "chunks": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0,
        }

    mean = sum(values) / len(values)
    return {
        "chunk_size": chunk_size,
        "chunks": len(values),
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "first_16": values[:16],
    }

def repeat_anchor_profile(path: Path, k: int, stride: int, max_freq: int):
    data = path.read_bytes()
    positions = {}
    for i in range(0, max(0, len(data) - k + 1), stride):
        key = data[i:i+k]
        positions.setdefault(key, []).append(i)

    useful = {key: pos for key, pos in positions.items() if 2 <= len(pos) <= max_freq}

    diffs = []
    for pos in useful.values():
        for a, b in zip(pos, pos[1:]):
            diffs.append(b - a)

    if not diffs:
        return {
            "k": k,
            "stride": stride,
            "useful_anchors": 0,
            "pair_count": 0,
            "median_distance": None,
            "far_gt_900kb_fraction": 0.0,
            "near_le_64kb_fraction": 0.0,
        }

    diffs.sort()
    mid = len(diffs) // 2
    median = diffs[mid] if len(diffs) % 2 else (diffs[mid - 1] + diffs[mid]) / 2

    return {
        "k": k,
        "stride": stride,
        "useful_anchors": len(useful),
        "pair_count": len(diffs),
        "median_distance": median,
        "far_gt_900kb_fraction": sum(1 for d in diffs if d > 900 * 1024) / len(diffs),
        "near_le_64kb_fraction": sum(1 for d in diffs if d <= 64 * 1024) / len(diffs),
    }

def bwt_runs(path: Path):
    if not path.exists():
        return None

    n = 0
    runs = 0
    prev = None
    alphabet = set()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            for b in chunk:
                alphabet.add(b)
                n += 1
                if prev is None or b != prev:
                    runs += 1
                    prev = b

    return {
        "bwt_path": str(path),
        "bwt_bytes": n,
        "bwt_runs": runs,
        "bwt_r_over_n": runs / n if n else None,
        "bwt_avg_run_len": n / runs if runs else None,
        "bwt_alphabet_size": len(alphabet),
        "bwt_sha256": sha256_file(path),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-size", type=int, default=65536)
    ap.add_argument("--anchor-stride", type=int, default=32)
    ap.add_argument("--anchor-max-freq", type=int, default=512)
    ap.add_argument("--bwt-path", default="")
    args = ap.parse_args()

    path = Path(args.path).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        raise SystemExit(f"missing input: {path}")

    artifact = {
        "artifact_version": VERSION,
        "purpose": "deterministic structural fingerprint, not codec prediction",
        "non_claims": [
            "does_not_predict_best_codec",
            "does_not_replace_compressor_trials",
            "does_not_claim_semantic_understanding",
        ],
        "source": {
            "path": str(path),
            "name": path.name,
            "sha256": sha256_file(path),
        },
        "byte_stats": byte_stats(path),
        "entropy_profile": chunk_entropy_profile(path, args.chunk_size),
        "anchor_repeat_profiles": [
            repeat_anchor_profile(path, k, args.anchor_stride, args.anchor_max_freq)
            for k in (8, 12, 16, 24, 32)
        ],
    }

    if args.bwt_path:
        artifact["bwt_profile"] = bwt_runs(Path(args.bwt_path).resolve())
    else:
        artifact["bwt_profile"] = None

    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "ok": True,
        "out": str(out),
        "artifact_version": VERSION,
        "source": str(path),
        "bytes": artifact["byte_stats"]["bytes"],
        "sha256": artifact["source"]["sha256"],
    }, indent=2))

if __name__ == "__main__":
    main()
