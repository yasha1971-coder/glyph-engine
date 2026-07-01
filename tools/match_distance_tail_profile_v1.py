#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from collections import Counter

ROOT = Path("/home/glyph/GLYPH_CPP_BACKEND")
OUT_JSON = ROOT / "benchmarks/results/MATCH_DISTANCE_TAIL_PROFILE_V1.json"
OUT_JSONL = ROOT / "benchmarks/results/MATCH_DISTANCE_TAIL_PROFILE_V1.jsonl"
OUT_MD = ROOT / "benchmarks/results/MATCH_DISTANCE_TAIL_PROFILE_V1.md"
OUT_DECISION = ROOT / "docs/review/GLYPH_MATCH_DISTANCE_TAIL_PROFILE_DECISION_V1.md"

LABELS = {
    "dickens": "bzip2",
    "mozilla": "xz",
    "mr": "bzip2",
    "nci": "zstd19",
    "ooffice": "xz",
    "osdb": "bzip2",
    "reymont": "bzip2",
    "samba": "xz",
    "sao": "xz",
    "webster": "xz",
    "x-ray": "bzip2",
    "xml": "bzip2",
}

K_LIST = [8, 12, 16]
TARGET_PER_K = 80
MAX_ATTEMPTS_PER_K = 6000
MAX_OCCURRENCES = 20000
RESERVOIR_LIMIT = 100000

NEAR_64K = 64 * 1024
FAR_900K = 900 * 1024
FAR_4M = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def entropy_and_nul(data: bytes) -> dict:
    n = len(data)
    c = Counter(data)
    ent = 0.0
    for v in c.values():
        p = v / n
        ent -= p * math.log2(p)
    printable = sum(v for b, v in c.items() if b in (9, 10, 13) or 32 <= b <= 126)
    return {
        "bytes": n,
        "alphabet": len(c),
        "entropy": ent,
        "nul": c.get(0, 0),
        "printable_fraction": printable / n if n else 0.0,
        "sha256": sha256_file(Path("__dummy__")) if False else None,
    }


def percentile(vals, q):
    vals = sorted(vals)
    if not vals:
        return None
    idx = int(round((len(vals) - 1) * q))
    return vals[idx]


def median(vals):
    vals = list(vals)
    return statistics.median(vals) if vals else None


def mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else None


def find_positions(data: bytes, pat: bytes):
    pos = []
    start = 0
    while True:
        i = data.find(pat, start)
        if i < 0:
            break
        pos.append(i)
        if len(pos) >= MAX_OCCURRENCES:
            break
        start = i + 1
    return pos


def reservoir_add(rng, reservoir, seen_count, values):
    for v in values:
        seen_count += 1
        if len(reservoir) < RESERVOIR_LIMIT:
            reservoir.append(v)
        else:
            j = rng.randrange(seen_count)
            if j < RESERVOIR_LIMIT:
                reservoir[j] = v
    return seen_count


def profile_file(path: Path, label: str, seed: int):
    rng = random.Random(seed)
    data = path.read_bytes()
    n = len(data)

    counts = Counter(data)
    ent = 0.0
    for v in counts.values():
        p = v / n
        ent -= p * math.log2(p)
    printable = sum(v for b, v in counts.items() if b in (9, 10, 13) or 32 <= b <= 126)

    per_k = {}
    all_pattern_medians = []
    all_pattern_p90s = []
    all_pattern_far900 = []
    all_pattern_far4m = []
    all_pattern_near64 = []

    total_diffs = 0
    far900_diffs = 0
    far4m_diffs = 0
    near64_diffs = 0

    reservoir = []
    reservoir_seen = 0

    for k in K_LIST:
        used = 0
        attempts = 0
        seen_patterns = set()

        k_pattern_medians = []
        k_pattern_p90s = []
        k_pattern_far900 = []
        k_pattern_far4m = []
        k_pattern_near64 = []

        while used < TARGET_PER_K and attempts < MAX_ATTEMPTS_PER_K:
            attempts += 1
            if n <= k:
                break

            start = rng.randrange(0, n - k)
            pat = data[start:start + k]

            if pat in seen_patterns:
                continue
            seen_patterns.add(pat)

            positions = find_positions(data, pat)
            if len(positions) < 2:
                continue

            diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            if not diffs:
                continue

            used += 1

            d_med = median(diffs)
            d_p90 = percentile(diffs, 0.90)

            f900 = sum(1 for d in diffs if d > FAR_900K) / len(diffs)
            f4m = sum(1 for d in diffs if d > FAR_4M) / len(diffs)
            n64 = sum(1 for d in diffs if d <= NEAR_64K) / len(diffs)

            k_pattern_medians.append(d_med)
            k_pattern_p90s.append(d_p90)
            k_pattern_far900.append(f900)
            k_pattern_far4m.append(f4m)
            k_pattern_near64.append(n64)

            all_pattern_medians.append(d_med)
            all_pattern_p90s.append(d_p90)
            all_pattern_far900.append(f900)
            all_pattern_far4m.append(f4m)
            all_pattern_near64.append(n64)

            total_diffs += len(diffs)
            far900_diffs += sum(1 for d in diffs if d > FAR_900K)
            far4m_diffs += sum(1 for d in diffs if d > FAR_4M)
            near64_diffs += sum(1 for d in diffs if d <= NEAR_64K)

            reservoir_seen = reservoir_add(rng, reservoir, reservoir_seen, diffs)

        per_k[str(k)] = {
            "patterns": used,
            "attempts": attempts,
            "median_distance_median": median(k_pattern_medians),
            "p90_distance_median": median(k_pattern_p90s),
            "far_gt_900kb_pattern_median": median(k_pattern_far900),
            "far_gt_4mb_pattern_median": median(k_pattern_far4m),
            "near_le_64kb_pattern_median": median(k_pattern_near64),
        }

    global_median = percentile(reservoir, 0.50)
    global_p90 = percentile(reservoir, 0.90)
    global_p99 = percentile(reservoir, 0.99)

    result = {
        "file": path.name,
        "label": label,
        "bytes": n,
        "sha256": sha256_file(path),
        "nul": counts.get(0, 0),
        "alphabet": len(counts),
        "entropy": ent,
        "printable_fraction": printable / n if n else 0.0,
        "patterns_total": len(all_pattern_medians),
        "diffs_total": total_diffs,
        "global_median_distance": global_median,
        "global_p90_distance": global_p90,
        "global_p99_distance": global_p99,
        "pattern_median_distance_median": median(all_pattern_medians),
        "pattern_p90_distance_median": median(all_pattern_p90s),
        "pattern_far_gt_900kb_median": median(all_pattern_far900),
        "pattern_far_gt_900kb_mean": mean(all_pattern_far900),
        "pattern_far_gt_4mb_median": median(all_pattern_far4m),
        "pattern_far_gt_4mb_mean": mean(all_pattern_far4m),
        "pattern_near_le_64kb_median": median(all_pattern_near64),
        "pattern_near_le_64kb_mean": mean(all_pattern_near64),
        "diff_far_gt_900kb_fraction": far900_diffs / total_diffs if total_diffs else None,
        "diff_far_gt_4mb_fraction": far4m_diffs / total_diffs if total_diffs else None,
        "diff_near_le_64kb_fraction": near64_diffs / total_diffs if total_diffs else None,
        "p90_over_median": (global_p90 / global_median) if global_median and global_p90 else None,
        "p99_over_median": (global_p99 / global_median) if global_median and global_p99 else None,
        "per_k": per_k,
    }

    return result


def best_threshold_rule(rows):
    features = [
        "global_median_distance",
        "global_p90_distance",
        "global_p99_distance",
        "pattern_median_distance_median",
        "pattern_p90_distance_median",
        "pattern_far_gt_900kb_median",
        "pattern_far_gt_900kb_mean",
        "pattern_far_gt_4mb_median",
        "pattern_far_gt_4mb_mean",
        "pattern_near_le_64kb_median",
        "pattern_near_le_64kb_mean",
        "diff_far_gt_900kb_fraction",
        "diff_far_gt_4mb_fraction",
        "diff_near_le_64kb_fraction",
        "p90_over_median",
        "p99_over_median",
    ]

    binary = [r for r in rows if r["label"] in ("bzip2", "xz")]
    best = None

    for feat in features:
        vals = sorted(set(r.get(feat) for r in binary if isinstance(r.get(feat), (int, float))))
        if not vals:
            continue

        thresholds = vals[:]
        for direction in ("xz_if_ge", "xz_if_le"):
            for t in thresholds:
                correct = 0
                preds = {}
                for r in binary:
                    v = r.get(feat)
                    if v is None:
                        pred = "unknown"
                    elif direction == "xz_if_ge":
                        pred = "xz" if v >= t else "bzip2"
                    else:
                        pred = "xz" if v <= t else "bzip2"

                    preds[r["file"]] = pred
                    if pred == r["label"]:
                        correct += 1

                item = {
                    "feature": feat,
                    "threshold": t,
                    "direction": direction,
                    "correct": correct,
                    "total": len(binary),
                    "accuracy": correct / len(binary) if binary else 0.0,
                    "predictions": preds,
                }

                if best is None or (item["correct"], item["accuracy"]) > (best["correct"], best["accuracy"]):
                    best = item

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("silesia_dir", nargs="?", default="")
    ap.add_argument("--seed", type=int, default=20260701)
    args = ap.parse_args()

    candidates = []
    if args.silesia_dir:
        candidates.append(Path(args.silesia_dir))
    candidates += [
        Path("/tmp/silesia_check"),
        Path("/tmp/silesia"),
        Path("/home/glyph/silesia"),
        ROOT / "silesia",
    ]

    silesia = None
    for c in candidates:
        if c.exists() and c.is_dir() and (c / "webster").exists():
            silesia = c.resolve()
            break

    if not silesia:
        raise SystemExit("Silesia directory not found. Pass path as argument.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_DECISION.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, label in LABELS.items():
        p = silesia / name
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        print(f"[tail-profile] {name}")
        rows.append(profile_file(p, label, args.seed + len(rows) * 1009))

    rule = best_threshold_rule(rows)

    for r in rows:
        if rule and r["label"] in ("bzip2", "xz"):
            pred = rule["predictions"].get(r["file"], "unknown")
        elif r["label"] == "zstd19":
            pred = "out_of_binary_gate"
        else:
            pred = "unknown"
        r["prediction"] = pred
        r["correct"] = pred == r["label"]

    binary_rows = [r for r in rows if r["label"] in ("bzip2", "xz")]
    binary_correct = sum(1 for r in binary_rows if r["correct"])
    all_correct = sum(1 for r in rows if r["correct"])

    if binary_correct >= 9:
        decision = "tail_profile_alive_for_bzip2_vs_xz_gate"
    elif binary_correct >= 8:
        decision = "tail_profile_weak_borderline"
    else:
        decision = "tail_profile_not_enough"

    out = {
        "ok": True,
        "date": "2026-07-01",
        "silesia_dir": str(silesia),
        "boundary": "raw binary byte scan; feature gate before GLYPH v0.x binary-safe production support",
        "rows": rows,
        "best_rule": rule,
        "binary_correct": binary_correct,
        "binary_total": len(binary_rows),
        "all_correct": all_correct,
        "all_total": len(rows),
        "decision": decision,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    OUT_JSONL.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")

    lines = []
    lines.append("# MATCH_DISTANCE_TAIL_PROFILE_V1")
    lines.append("")
    lines.append("Status: measured")
    lines.append("Date: 2026-07-01")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Test whether match-distance tail shape separates Silesia `bzip2` vs `xz` winners after median match-distance failed.")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("This is a raw-byte feature gate over full Silesia files. It is not a GLYPH v0.x production binary-safe claim.")
    lines.append("")
    lines.append("Current GLYPH v0.x remains sentinel-safe; this test measures whether the feature is worth integrating into a future binary-safe GLYPH/STRIDE path.")
    lines.append("")
    lines.append("## Best one-threshold rule")
    lines.append("")
    if rule:
        lines.append(f"- feature: `{rule['feature']}`")
        lines.append(f"- direction: `{rule['direction']}`")
        lines.append(f"- threshold: `{rule['threshold']}`")
        lines.append(f"- binary accuracy: `{rule['correct']}/{rule['total']}` = `{rule['accuracy']:.4f}`")
    else:
        lines.append("No rule found.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| file | label | predicted | correct | bytes | entropy | NUL | patterns | median | p90 | p99 | far>900KB diff | far>4MB diff | near<=64KB diff | p90/median |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        lines.append(
            f"| `{r['file']}` | {r['label']} | {r['prediction']} | {r['correct']} | "
            f"{r['bytes']} | {r['entropy']:.4f} | {r['nul']} | {r['patterns_total']} | "
            f"{r['global_median_distance']} | {r['global_p90_distance']} | {r['global_p99_distance']} | "
            f"{r['diff_far_gt_900kb_fraction']:.6f} | {r['diff_far_gt_4mb_fraction']:.6f} | "
            f"{r['diff_near_le_64kb_fraction']:.6f} | "
            f"{r['p90_over_median'] if r['p90_over_median'] is not None else 'NA'} |"
        )

    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"`{decision}`")
    lines.append("")
    lines.append(f"- binary gate accuracy: `{binary_correct}/{len(binary_rows)}`")
    lines.append(f"- all-label accuracy including `zstd19` outlier: `{all_correct}/{len(rows)}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Median match-distance alone already failed.")
    lines.append("- This test checks whether the tail of match-distance distribution is a better structural feature.")
    lines.append("- If accuracy is high, the GLYPH/STRIDE bridge remains alive as tail-distance profiling.")
    lines.append("- If accuracy is weak, do not claim codec prediction from match-distance yet.")
    lines.append("")
    lines.append("## Non-claims")
    lines.append("")
    lines.append("- This is not a production codec router.")
    lines.append("- This is not binary-safe GLYPH production support.")
    lines.append("- This does not replace compressor trials.")
    lines.append("- This is an in-sample falsification gate over Silesia.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    d = []
    d.append("# GLYPH_MATCH_DISTANCE_TAIL_PROFILE_DECISION_V1")
    d.append("")
    d.append("Status: measured gate")
    d.append("Date: 2026-07-01")
    d.append("")
    d.append("## Decision")
    d.append("")
    d.append(f"`{decision}`")
    d.append("")
    if rule:
        d.append("## Best rule")
        d.append("")
        d.append(f"- feature: `{rule['feature']}`")
        d.append(f"- direction: `{rule['direction']}`")
        d.append(f"- threshold: `{rule['threshold']}`")
        d.append(f"- binary accuracy: `{rule['correct']}/{rule['total']}`")
        d.append("")
    d.append("## Accuracy")
    d.append("")
    d.append(f"- bzip2 vs xz: `{binary_correct}/{len(binary_rows)}`")
    d.append(f"- all labels including zstd19: `{all_correct}/{len(rows)}`")
    d.append("")
    d.append("## Source report")
    d.append("")
    d.append("`benchmarks/results/MATCH_DISTANCE_TAIL_PROFILE_V1.md`")
    d.append("")
    OUT_DECISION.write_text("\n".join(d), encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "report": str(OUT_MD),
        "decision_file": str(OUT_DECISION),
        "decision": decision,
        "binary_accuracy": f"{binary_correct}/{len(binary_rows)}",
        "all_accuracy": f"{all_correct}/{len(rows)}",
    }, indent=2))

if __name__ == "__main__":
    main()
