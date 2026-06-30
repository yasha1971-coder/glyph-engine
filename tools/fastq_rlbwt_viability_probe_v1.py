#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(path: Path) -> str:
    s = path.name
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:120]


def is_gzip(path: Path) -> bool:
    if path.suffix.lower() == ".gz":
        return True
    try:
        with path.open("rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False


def copy_prefix(src: Path, dst: Path, max_bytes: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if is_gzip(src) else open
    written = 0

    with opener(src, "rb") as f, dst.open("wb") as out:
        while written < max_bytes:
            chunk = f.read(min(1024 * 1024, max_bytes - written))
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)

    return written


def byte_stats(path: Path):
    counts = Counter()
    n = 0
    nul = 0
    newline = 0
    printable = 0

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            n += len(chunk)
            nul += chunk.count(0)
            newline += chunk.count(10)
            for b in chunk:
                counts[b] += 1
                if b in (9, 10, 13) or 32 <= b <= 126:
                    printable += 1

    ent = 0.0
    for c in counts.values():
        p = c / n if n else 0
        if p:
            ent -= p * math.log2(p)

    return {
        "bytes": n,
        "sha256": sha256_file(path),
        "alphabet_size": len(counts),
        "nul_bytes": nul,
        "newline_bytes": newline,
        "printable_fraction": printable / n if n else 0,
        "entropy_bits_per_byte": ent,
        "top_bytes": [
            {"byte": b, "count": c, "fraction": c / n if n else 0}
            for b, c in counts.most_common(12)
        ],
    }


def block_dup_stats(path: Path, block_size: int, max_probe_bytes: int = 64 * 1024 * 1024):
    seen = Counter()
    total = 0
    read_total = 0

    with path.open("rb") as f:
        while read_total < max_probe_bytes:
            block = f.read(min(block_size, max_probe_bytes - read_total))
            if not block:
                break
            read_total += len(block)
            total += 1
            seen[hashlib.sha256(block).digest()] += 1

    unique = len(seen)
    dup = sum(c - 1 for c in seen.values() if c > 1)

    return {
        "block_size": block_size,
        "probe_bytes": read_total,
        "blocks_total": total,
        "unique_blocks": unique,
        "duplicate_blocks": dup,
        "duplicate_fraction": dup / total if total else 0,
    }


def measure_bwt_runs(path: Path):
    n = 0
    runs = 0
    prev = None
    alphabet = set()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            for b in chunk:
                alphabet.add(b)
                if prev is None or b != prev:
                    runs += 1
                    prev = b
                n += 1

    return {
        "bwt_bytes": n,
        "runs": runs,
        "r_over_n": runs / n if n else None,
        "avg_run_len": n / runs if runs else None,
        "alphabet_size": len(alphabet),
        "sha256": sha256_file(path),
    }


def verdict(r_over_n):
    if r_over_n is None:
        return "unknown"
    if r_over_n <= 0.001:
        return "extremely strong RLBWT signal"
    if r_over_n <= 0.01:
        return "strong RLBWT signal"
    if r_over_n <= 0.05:
        return "possible RLBWT signal"
    if r_over_n <= 0.15:
        return "weak/moderate RLBWT signal"
    return "poor RLBWT signal"


def run_build(sample: Path, index_dir: Path):
    build = ROOT / "tools" / "build_glyph_index_v1.sh"
    if not build.exists():
        return {
            "ok": False,
            "error": "missing tools/build_glyph_index_v1.sh",
        }

    index_dir.mkdir(parents=True, exist_ok=True)

    cmd = [str(build), str(sample), str(index_dir)]
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return {
        "ok": p.returncode == 0,
        "returncode": p.returncode,
        "cmd": cmd,
        "log_tail": "\n".join(p.stdout.splitlines()[-80:]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="benchmarks/work/fastq_rlbwt_viability_v1")
    ap.add_argument("--sample-mib", nargs="+", type=int, default=[256])
    ap.add_argument("--json-out", default="benchmarks/results/FASTQ_RLBWT_VIABILITY_PROBE_V1.json")
    ap.add_argument("--md-out", default="benchmarks/results/FASTQ_RLBWT_VIABILITY_PROBE_V1.md")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for src_s in args.paths:
        src = Path(src_s).expanduser().resolve()
        if not src.exists():
            results.append({
                "ok": False,
                "source": str(src),
                "error": "missing source",
            })
            continue

        for mib in args.sample_mib:
            sample_bytes_target = mib * 1024 * 1024
            label = f"{safe_name(src)}__{mib}MiB"
            case_dir = out_dir / label
            sample = case_dir / "corpus.sample.bin"
            index_dir = case_dir / "index"

            copied = copy_prefix(src, sample, sample_bytes_target)
            raw = byte_stats(sample)

            block_dups = [
                block_dup_stats(sample, bs)
                for bs in [64, 256, 1024, 4096, 16384]
            ]

            build_result = run_build(sample, index_dir)

            bwt_result = None
            bwt_path = index_dir / "bwt.bin"
            if build_result.get("ok") and bwt_path.exists():
                bwt_result = measure_bwt_runs(bwt_path)
                bwt_result["verdict"] = verdict(bwt_result["r_over_n"])

            results.append({
                "ok": True,
                "source": str(src),
                "source_size_bytes": src.stat().st_size,
                "source_is_gzip": is_gzip(src),
                "sample_mib": mib,
                "sample_bytes": copied,
                "sample_path": str(sample.relative_to(ROOT)),
                "sample_sha256": raw["sha256"],
                "raw_stats": raw,
                "block_duplicate_stats": block_dups,
                "build": build_result,
                "bwt": bwt_result,
            })

    json_out = ROOT / args.json_out
    md_out = ROOT / args.md_out

    json_out.write_text(json.dumps({
        "ok": True,
        "results": results,
    }, indent=2, sort_keys=True), encoding="utf-8")

    lines = []
    lines.append("# FASTQ_RLBWT_VIABILITY_PROBE_V1")
    lines.append("")
    lines.append("Status: measured")
    lines.append("Date: 2026-06-28")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Measure whether real FASTQ corpora have low BWT run ratio `r/n`, which would justify further RLBWT/r-index engineering for GLYPH.")
    lines.append("")
    lines.append("This is a viability gate, not a production RLBWT claim.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| source | sample | sample bytes | raw entropy | nul bytes | dup64 | dup256 | dup1024 | BWT r/n | avg run | verdict |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for r in results:
        if not r.get("ok"):
            lines.append(f"| `{r.get('source')}` | error | 0 | - | - | - | - | - | - | - | {r.get('error')} |")
            continue

        raw = r["raw_stats"]
        d = {x["block_size"]: x["duplicate_fraction"] for x in r["block_duplicate_stats"]}
        bwt = r.get("bwt")

        if bwt:
            rn = f"{bwt['r_over_n']:.8f}"
            avg = f"{bwt['avg_run_len']:.3f}"
            ver = bwt["verdict"]
        else:
            rn = "build_failed"
            avg = "-"
            ver = "no BWT"

        lines.append(
            f"| `{Path(r['source']).name}` | {r['sample_mib']} MiB | {r['sample_bytes']} | "
            f"{raw['entropy_bits_per_byte']:.4f} | {raw['nul_bytes']} | "
            f"{d.get(64, 0):.6f} | {d.get(256, 0):.6f} | {d.get(1024, 0):.6f} | "
            f"{rn} | {avg} | {ver} |"
        )

    lines.append("")
    lines.append("## Decision rule")
    lines.append("")
    lines.append("- `r/n <= 0.01`: strong RLBWT direction.")
    lines.append("- `r/n <= 0.05`: worth continuing.")
    lines.append("- `r/n > 0.15`: do not use this corpus as a compact-index argument.")
    lines.append("")
    lines.append("## Non-claims")
    lines.append("")
    lines.append("- This does not build a production RLBWT index.")
    lines.append("- This does not claim FASTQ support at 50 GB full scale.")
    lines.append("- This does not claim binary-safe production runtime.")
    lines.append("- This only measures real-corpus repetitiveness signals for future GLYPH engineering.")
    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append("If 256 MiB shows `r/n <= 0.05`, rerun the same corpus with 1024 MiB before committing to RLBWT engineering.")
    lines.append("")

    md_out.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "json": str(json_out.relative_to(ROOT)),
        "report": str(md_out.relative_to(ROOT)),
        "cases": len(results),
    }, indent=2))


if __name__ == "__main__":
    main()
