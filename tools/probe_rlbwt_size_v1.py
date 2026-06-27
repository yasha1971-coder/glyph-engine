#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def uleb128_len(x: int) -> int:
    if x < 0:
        raise ValueError("negative integer")
    n = 1
    while x >= 128:
        x >>= 7
        n += 1
    return n


def analyze_bwt(path: Path, label: str):
    data = path.read_bytes()
    n = len(data)

    if n == 0:
        raise ValueError(f"empty BWT: {path}")

    runs = 0
    max_run = 0
    run_len = 0
    last = None

    symbol_hist = [0] * 256
    run_symbol_hist = [0] * 256
    run_len_hist_small = {}

    rle_u32_bytes = 0
    rle_u64_bytes = 0
    rle_varint_bytes = 0

    for b in data:
        symbol_hist[b] += 1

        if last is None:
            last = b
            run_len = 1
            runs = 1
            continue

        if b == last:
            run_len += 1
        else:
            run_symbol_hist[last] += 1
            max_run = max(max_run, run_len)
            run_len_hist_small[min(run_len, 1024)] = run_len_hist_small.get(min(run_len, 1024), 0) + 1

            rle_u32_bytes += 1 + 4
            rle_u64_bytes += 1 + 8
            rle_varint_bytes += 1 + uleb128_len(run_len)

            last = b
            run_len = 1
            runs += 1

    run_symbol_hist[last] += 1
    max_run = max(max_run, run_len)
    run_len_hist_small[min(run_len, 1024)] = run_len_hist_small.get(min(run_len, 1024), 0) + 1

    rle_u32_bytes += 1 + 4
    rle_u64_bytes += 1 + 8
    rle_varint_bytes += 1 + uleb128_len(run_len)

    distinct_symbols = sum(1 for x in symbol_hist if x)
    distinct_run_symbols = sum(1 for x in run_symbol_hist if x)

    avg_run = n / runs

    return {
        "label": label,
        "path": str(path),
        "raw_bwt_bytes": n,
        "runs": runs,
        "avg_run_len": avg_run,
        "max_run_len": max_run,
        "distinct_symbols": distinct_symbols,
        "distinct_run_symbols": distinct_run_symbols,
        "rlbwt_u32_pair_bytes": rle_u32_bytes,
        "rlbwt_u64_pair_bytes": rle_u64_bytes,
        "rlbwt_varint_pair_bytes": rle_varint_bytes,
        "rlbwt_u32_ratio_vs_bwt": rle_u32_bytes / n,
        "rlbwt_u64_ratio_vs_bwt": rle_u64_bytes / n,
        "rlbwt_varint_ratio_vs_bwt": rle_varint_bytes / n,
        "rlbwt_u32_saves_bytes": n - rle_u32_bytes,
        "rlbwt_varint_saves_bytes": n - rle_varint_bytes,
    }


def main():
    ap = argparse.ArgumentParser(description="Probe raw BWT run-length compressibility.")
    ap.add_argument("--item", action="append", required=True,
                    help="label:path, e.g. pizza:/path/to/bwt.bin")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for item in args.item:
        if ":" not in item:
            raise SystemExit(f"bad --item, expected label:path: {item}")
        label, path_s = item.split(":", 1)
        p = Path(path_s)
        if not p.exists():
            raise SystemExit(f"missing BWT file for {label}: {p}")
        rows.append(analyze_bwt(p, label))

    csv_path = out_dir / "rlbwt_size_probe_v1.csv"
    json_path = out_dir / "rlbwt_size_probe_v1.json"
    md_path = out_dir / "rlbwt_size_probe_v1.md"

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# RLBWT Size Probe V1\n\n")
        f.write("This measures raw BWT run-length compressibility only.\n\n")
        f.write("It does not implement rank/select over compressed BWT yet.\n\n")
        f.write("| label | raw_bwt_bytes | runs | avg_run_len | max_run_len | rlbwt_u32_pair_bytes | rlbwt_varint_pair_bytes | u32_ratio_vs_bwt | varint_ratio_vs_bwt |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['label']} "
                f"| {r['raw_bwt_bytes']} "
                f"| {r['runs']} "
                f"| {r['avg_run_len']:.3f} "
                f"| {r['max_run_len']} "
                f"| {r['rlbwt_u32_pair_bytes']} "
                f"| {r['rlbwt_varint_pair_bytes']} "
                f"| {r['rlbwt_u32_ratio_vs_bwt']:.3f}x "
                f"| {r['rlbwt_varint_ratio_vs_bwt']:.3f}x |\n"
            )

    print("[rlbwt-probe] wrote:")
    print(csv_path)
    print(json_path)
    print(md_path)
    print()
    print(md_path.read_text())


if __name__ == "__main__":
    main()
