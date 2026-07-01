#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/home/glyph/GLYPH_CPP_BACKEND")

OUT_JSON = ROOT / "benchmarks/results/ANCHOR_REPEAT_DISTANCE_PROFILE_V1.json"
OUT_JSONL = ROOT / "benchmarks/results/ANCHOR_REPEAT_DISTANCE_PROFILE_V1.jsonl"
OUT_MD = ROOT / "benchmarks/results/ANCHOR_REPEAT_DISTANCE_PROFILE_V1.md"
OUT_DECISION = ROOT / "docs/review/GLYPH_ANCHOR_REPEAT_DISTANCE_PROFILE_DECISION_V1.md"

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

K_LIST = [8, 12, 16, 24, 32]
STRIDE = 32

MIN_FREQ = 2
MAX_FREQ = 512

NEAR_64K = 64 * 1024
FAR_900K = 900 * 1024
FAR_4M = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def entropy(data: bytes) -> float:
    n = len(data)
    c = Counter(data)
    e = 0.0
    for v in c.values():
        p = v / n
        e -= p * math.log2(p)
    return e


def percentile(vals, q):
    if not vals:
        return None
    vals = sorted(vals)
    idx = int(round((len(vals) - 1) * q))
    return vals[idx]


def mean(vals):
    return sum(vals) / len(vals) if vals else None


def scan_k(data: bytes, k: int):
    pos = defaultdict(list)
    n = len(data)

    for i in range(0, max(0, n - k + 1), STRIDE):
        a = data[i:i + k]
        pos[a].append(i)

    total_anchors = len(pos)
    repeat_keys = []
    useful_keys = []

    pair_diffs = []
    useful_pair_diffs = []

    useful_anchor_count = 0
    repeated_sample_positions = 0

    for a, xs in pos.items():
        f = len(xs)

        if f >= 2:
            repeat_keys.append(a)
            repeated_sample_positions += f

        if MIN_FREQ <= f <= MAX_FREQ:
            useful_keys.append(a)
            useful_anchor_count += 1

            for j in range(len(xs) - 1):
                d = xs[j + 1] - xs[j]
                useful_pair_diffs.append(d)

        if f >= 2:
            for j in range(len(xs) - 1):
                d = xs[j + 1] - xs[j]
                pair_diffs.append(d)

    useful_far900 = [d for d in useful_pair_diffs if d > FAR_900K]
    useful_far4m = [d for d in useful_pair_diffs if d > FAR_4M]
    useful_near64 = [d for d in useful_pair_diffs if d <= NEAR_64K]

    all_far900 = [d for d in pair_diffs if d > FAR_900K]
    all_far4m = [d for d in pair_diffs if d > FAR_4M]
    all_near64 = [d for d in pair_diffs if d <= NEAR_64K]

    sampled_positions = max(1, (n - k + STRIDE) // STRIDE)

    return {
        "k": k,
        "sampled_positions": sampled_positions,
        "unique_anchors": total_anchors,
        "repeated_anchor_keys": len(repeat_keys),
        "useful_anchor_keys": useful_anchor_count,
        "repeat_key_fraction": len(repeat_keys) / total_anchors if total_anchors else 0.0,
        "useful_key_fraction": useful_anchor_count / total_anchors if total_anchors else 0.0,
        "repeated_position_fraction": repeated_sample_positions / sampled_positions if sampled_positions else 0.0,

        "all_pair_count": len(pair_diffs),
        "all_pair_median": percentile(pair_diffs, 0.50),
        "all_pair_p90": percentile(pair_diffs, 0.90),
        "all_far_gt_900kb_fraction": len(all_far900) / len(pair_diffs) if pair_diffs else 0.0,
        "all_far_gt_4mb_fraction": len(all_far4m) / len(pair_diffs) if pair_diffs else 0.0,
        "all_near_le_64kb_fraction": len(all_near64) / len(pair_diffs) if pair_diffs else 0.0,

        "useful_pair_count": len(useful_pair_diffs),
        "useful_pair_median": percentile(useful_pair_diffs, 0.50),
        "useful_pair_p90": percentile(useful_pair_diffs, 0.90),
        "useful_pair_p99": percentile(useful_pair_diffs, 0.99),
        "useful_far_gt_900kb_fraction": len(useful_far900) / len(useful_pair_diffs) if useful_pair_diffs else 0.0,
        "useful_far_gt_4mb_fraction": len(useful_far4m) / len(useful_pair_diffs) if useful_pair_diffs else 0.0,
        "useful_near_le_64kb_fraction": len(useful_near64) / len(useful_pair_diffs) if useful_pair_diffs else 0.0,
        "useful_far900_to_near64": (
            (len(useful_far900) / len(useful_pair_diffs)) /
            max(1e-12, (len(useful_near64) / len(useful_pair_diffs)))
        ) if useful_pair_diffs else 0.0,
    }


def profile_file(path: Path, label: str):
    data = path.read_bytes()
    counts = Counter(data)

    per_k = {}
    flat = {}

    for k in K_LIST:
        r = scan_k(data, k)
        per_k[str(k)] = r

        for key, val in r.items():
            if key == "k":
                continue
            if isinstance(val, (int, float)) or val is None:
                flat[f"k{k}_{key}"] = val

    return {
        "file": path.name,
        "label": label,
        "bytes": len(data),
        "sha256": sha256_file(path),
        "nul": counts.get(0, 0),
        "entropy": entropy(data),
        "per_k": per_k,
        **flat,
    }


def best_threshold_rule(rows):
    binary = [r for r in rows if r["label"] in ("bzip2", "xz")]

    features = []
    for r in binary:
        for k, v in r.items():
            if k.startswith("k") and isinstance(v, (int, float)):
                features.append(k)

    features = sorted(set(features))

    best = None

    for feat in features:
        vals = sorted(set(r.get(feat) for r in binary if isinstance(r.get(feat), (int, float))))
        if not vals:
            continue

        for direction in ("xz_if_ge", "xz_if_le"):
            for t in vals:
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
                    "direction": direction,
                    "threshold": t,
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
            continue

        print(f"[anchor-profile] {name}")
        rows.append(profile_file(p, label))

    rule = best_threshold_rule(rows)

    for r in rows:
        if r["label"] in ("bzip2", "xz") and rule:
            r["prediction"] = rule["predictions"].get(r["file"], "unknown")
        elif r["label"] == "zstd19":
            r["prediction"] = "out_of_binary_gate"
        else:
            r["prediction"] = "unknown"

        r["correct"] = r["prediction"] == r["label"]

    binary = [r for r in rows if r["label"] in ("bzip2", "xz")]
    binary_correct = sum(1 for r in binary if r["correct"])
    all_correct = sum(1 for r in rows if r["correct"])

    if binary_correct >= 10:
        decision = "anchor_repeat_distance_alive"
    elif binary_correct >= 9:
        decision = "anchor_repeat_distance_borderline_alive"
    elif binary_correct >= 8:
        decision = "anchor_repeat_distance_weak"
    else:
        decision = "anchor_repeat_distance_rejected"

    out = {
        "ok": True,
        "date": "2026-07-01",
        "silesia_dir": str(silesia),
        "stride": STRIDE,
        "min_freq": MIN_FREQ,
        "max_freq": MAX_FREQ,
        "k_list": K_LIST,
        "rows": rows,
        "best_rule": rule,
        "binary_correct": binary_correct,
        "binary_total": len(binary),
        "all_correct": all_correct,
        "all_total": len(rows),
        "decision": decision,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    OUT_JSONL.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")

    lines = []
    lines.append("# ANCHOR_REPEAT_DISTANCE_PROFILE_V1")
    lines.append("")
    lines.append("Status: measured")
    lines.append("Date: 2026-07-01")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Test whether useful repeated anchors separate Silesia `bzip2` vs `xz` winners better than random-pattern match-distance.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(f"- deterministic byte anchors sampled every `{STRIDE}` bytes")
    lines.append(f"- k values: `{K_LIST}`")
    lines.append(f"- useful anchor frequency range: `{MIN_FREQ}..{MAX_FREQ}`")
    lines.append("- measures near/far repeat mass among useful anchors")
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
    lines.append("| file | label | predicted | correct | bytes | entropy | NUL | k16 useful pairs | k16 far>900KB | k24 useful pairs | k24 far>900KB | k32 useful pairs | k32 far>900KB |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        lines.append(
            f"| `{r['file']}` | {r['label']} | {r['prediction']} | {r['correct']} | "
            f"{r['bytes']} | {r['entropy']:.4f} | {r['nul']} | "
            f"{r.get('k16_useful_pair_count')} | {r.get('k16_useful_far_gt_900kb_fraction'):.6f} | "
            f"{r.get('k24_useful_pair_count')} | {r.get('k24_useful_far_gt_900kb_fraction'):.6f} | "
            f"{r.get('k32_useful_pair_count')} | {r.get('k32_useful_far_gt_900kb_fraction'):.6f} |"
        )

    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"`{decision}`")
    lines.append("")
    lines.append(f"- bzip2 vs xz accuracy: `{binary_correct}/{len(binary)}`")
    lines.append(f"- all labels including zstd19 outlier: `{all_correct}/{len(rows)}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If this beats the random-pattern tail profile, the useful-anchor hypothesis survives.")
    lines.append("- If it does not, match-distance is not yet a reliable codec predictor.")
    lines.append("- A strong result still needs out-of-sample validation, because this is Silesia in-sample threshold search.")
    lines.append("")
    lines.append("## Non-claims")
    lines.append("")
    lines.append("- This is not a production codec router.")
    lines.append("- This is not binary-safe GLYPH production support.")
    lines.append("- This does not replace compressor trials.")
    lines.append("- This is a falsification gate.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    d = []
    d.append("# GLYPH_ANCHOR_REPEAT_DISTANCE_PROFILE_DECISION_V1")
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
    d.append(f"- bzip2 vs xz: `{binary_correct}/{len(binary)}`")
    d.append(f"- all labels including zstd19: `{all_correct}/{len(rows)}`")
    d.append("")
    d.append("## Source report")
    d.append("")
    d.append("`benchmarks/results/ANCHOR_REPEAT_DISTANCE_PROFILE_V1.md`")
    d.append("")

    OUT_DECISION.write_text("\n".join(d), encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "report": str(OUT_MD),
        "decision_file": str(OUT_DECISION),
        "decision": decision,
        "binary_accuracy": f"{binary_correct}/{len(binary)}",
        "all_accuracy": f"{all_correct}/{len(rows)}",
    }, indent=2))


if __name__ == "__main__":
    main()
