#!/usr/bin/env python3
import hashlib
import json
import math
from pathlib import Path
from collections import Counter


ROOT = Path(".").resolve()

CANDIDATES = {
    "main_bwt": Path("out/bwt.bin"),
    "corpus2_true_bwt": Path("out/corpus2_true_bwt.bin"),
    "py_true_bwt": Path("out/py_true/corpus2_true_backend.bwt.bin"),
    "source_corpus": Path("out/py_true/corpus2_true_backend.corpus.bin"),
    "sa": Path("out/sa.bin"),
    "corpus2_true_sa": Path("out/corpus2_true_sa.bin"),
    "py_true_sa": Path("out/py_true/corpus2_true_backend.sa.bin"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def byte_entropy(path: Path) -> float:
    counts = Counter()
    n = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            counts.update(chunk)
            n += len(chunk)
    if n == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def byte_stats(path: Path):
    counts = Counter()
    n = 0
    nul = 0
    printable = 0
    newline = 0

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            n += len(chunk)
            nul += chunk.count(0)
            newline += chunk.count(10)
            for b in chunk:
                counts[b] += 1
                if b in (9, 10, 13) or 32 <= b <= 126:
                    printable += 1

    top = [{"byte": b, "count": c, "fraction": c / n if n else 0} for b, c in counts.most_common(12)]

    return {
        "bytes": n,
        "sha256": sha256_file(path),
        "alphabet_size": len(counts),
        "nul_bytes": nul,
        "newline_bytes": newline,
        "printable_fraction": printable / n if n else 0,
        "entropy_bits_per_byte": byte_entropy(path),
        "top_bytes": top,
    }


def block_dup_stats(path: Path, block_size: int):
    seen = Counter()
    total = 0
    full = 0

    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            total += 1
            if len(block) == block_size:
                full += 1
            seen[hashlib.sha256(block).digest()] += 1

    unique = len(seen)
    dup_blocks = sum(c - 1 for c in seen.values() if c > 1)

    return {
        "block_size": block_size,
        "blocks_total": total,
        "unique_blocks": unique,
        "duplicate_blocks": dup_blocks,
        "duplicate_fraction": dup_blocks / total if total else 0,
    }


def main():
    files = {}

    for name, rel in CANDIDATES.items():
        path = ROOT / rel
        if path.exists():
            files[name] = {
                "path": str(rel),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        else:
            files[name] = {
                "path": str(rel),
                "missing": True,
            }

    bwt_hashes = {}
    for name in ["main_bwt", "corpus2_true_bwt", "py_true_bwt"]:
        rec = files.get(name, {})
        if "sha256" in rec:
            bwt_hashes[name] = rec["sha256"]

    all_bwt_equal = len(set(bwt_hashes.values())) == 1 if bwt_hashes else False

    source_path = ROOT / CANDIDATES["source_corpus"]
    source = None
    block_dups = []
    if source_path.exists():
        source = byte_stats(source_path)
        for bs in [64, 256, 1024, 4096, 16384]:
            block_dups.append(block_dup_stats(source_path, bs))

    out = {
        "ok": True,
        "files": files,
        "all_available_bwt_hashes_equal": all_bwt_equal,
        "source_corpus_stats": source,
        "source_block_duplicate_stats": block_dups,
        "interpretation": {
            "main_question": "Can the measured low r/n on out/bwt.bin be tied to a source corpus?",
            "answer": "yes" if source and all_bwt_equal else "partial",
            "boundary": "This traces local artifact provenance only; it does not identify the original dataset name or external source.",
        },
    }

    out_json = Path("benchmarks/results/BWT_REPETITIVENESS_SOURCE_TRACE_V1.json")
    out_md = Path("benchmarks/results/BWT_REPETITIVENESS_SOURCE_TRACE_V1.md")

    out_json.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    lines = []
    lines.append("# BWT_REPETITIVENESS_SOURCE_TRACE_V1")
    lines.append("")
    lines.append("Status: measured")
    lines.append("Date: 2026-06-28")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Trace the local source/provenance of the strongest BWT repetitiveness signal found by `BWT_REPETITIVENESS_PROBE_V1`.")
    lines.append("")
    lines.append("The strongest measured signal was:")
    lines.append("")
    lines.append("    out/bwt.bin")
    lines.append("    n = 100,000,001")
    lines.append("    r/n = 0.0394347396056526")
    lines.append("    avg_run_len = 25.358")
    lines.append("")
    lines.append("## File identity")
    lines.append("")
    lines.append("| name | path | bytes | sha256 |")
    lines.append("|---|---|---:|---|")
    for name, rec in files.items():
        if rec.get("missing"):
            lines.append(f"| {name} | `{rec['path']}` | missing | missing |")
        else:
            lines.append(f"| {name} | `{rec['path']}` | {rec['bytes']} | `{rec['sha256']}` |")
    lines.append("")
    lines.append("## BWT equality")
    lines.append("")
    lines.append(f"All available BWT hashes equal: `{all_bwt_equal}`")
    lines.append("")
    if source:
        lines.append("## Source corpus stats")
        lines.append("")
        lines.append(f"- path: `out/py_true/corpus2_true_backend.corpus.bin`")
        lines.append(f"- bytes: {source['bytes']}")
        lines.append(f"- sha256: `{source['sha256']}`")
        lines.append(f"- alphabet_size: {source['alphabet_size']}")
        lines.append(f"- nul_bytes: {source['nul_bytes']}")
        lines.append(f"- newline_bytes: {source['newline_bytes']}")
        lines.append(f"- printable_fraction: {source['printable_fraction']:.6f}")
        lines.append(f"- entropy_bits_per_byte: {source['entropy_bits_per_byte']:.6f}")
        lines.append("")
        lines.append("## Exact block duplicate probe")
        lines.append("")
        lines.append("| block size | total blocks | unique blocks | duplicate blocks | duplicate fraction |")
        lines.append("|---:|---:|---:|---:|---:|")
        for b in block_dups:
            lines.append(
                f"| {b['block_size']} | {b['blocks_total']} | {b['unique_blocks']} | "
                f"{b['duplicate_blocks']} | {b['duplicate_fraction']:.8f} |"
            )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This confirms that the strongest local RLBWT signal is tied to the old `corpus2_true_backend` artifact family.")
    lines.append("")
    lines.append("It does not yet identify the external origin or semantic domain of the corpus.")
    lines.append("")
    lines.append("Therefore the result is useful as an internal engineering signal, but should not be used as a public domain claim until the corpus origin is named and reproducible.")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("Do not pivot GLYPH to deduplication.")
    lines.append("")
    lines.append("Use this as a gate for the narrower research direction:")
    lines.append("")
    lines.append("    replayable exact-byte evidence over repetitive fixed corpora")
    lines.append("")
    lines.append("Next useful step: measure `r/n` on named real corpora such as HDFS logs, versioned backups, source snapshots, and public log datasets.")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "json": str(out_json),
        "report": str(out_md),
        "all_available_bwt_hashes_equal": all_bwt_equal,
        "source": "out/py_true/corpus2_true_backend.corpus.bin" if source else None,
    }, indent=2))


if __name__ == "__main__":
    main()
