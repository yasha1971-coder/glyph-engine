#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path("/home/glyph/GLYPH_CPP_BACKEND")

IN_JSON = ROOT / "benchmarks/results/ANCHOR_REPEAT_DISTANCE_PROFILE_V1.json"
OUT_JSON = ROOT / "benchmarks/results/ANCHOR_REPEAT_DISTANCE_PREDICTOR_AUDIT_V1.json"
OUT_MD = ROOT / "benchmarks/results/ANCHOR_REPEAT_DISTANCE_PREDICTOR_AUDIT_V1.md"
OUT_DECISION = ROOT / "docs/review/GLYPH_ANCHOR_REPEAT_DISTANCE_PREDICTOR_AUDIT_DECISION_V1.md"

META = {
    "bytes",
    "nul",
    "entropy",
}

LEAKY_WORDS = [
    "count",
    "positions",
    "unique",
    "keys",
    "bytes",
    "nul",
    "entropy",
]

DISTANCE_WORDS = [
    "median",
    "p90",
    "p99",
    "fraction",
    "far900_to_near64",
]


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def numeric_features(rows):
    fs = set()
    for r in rows:
        for k, v in r.items():
            if is_number(v):
                fs.add(k)
    return sorted(fs)


def classify_feature(f):
    if f in META:
        return "metadata"
    if any(w in f for w in LEAKY_WORDS):
        if any(w in f for w in DISTANCE_WORDS):
            return "normalized_or_distance"
        return "count_or_scale"
    if any(w in f for w in DISTANCE_WORDS):
        return "normalized_or_distance"
    return "other_numeric"


def pred(rule, row):
    v = row.get(rule["feature"])
    if not is_number(v):
        return "unknown"
    if rule["direction"] == "xz_if_ge":
        return "xz" if v >= rule["threshold"] else "bzip2"
    return "xz" if v <= rule["threshold"] else "bzip2"


def best_rule(train, features):
    best = None

    for feat in features:
        vals = sorted(set(r.get(feat) for r in train if is_number(r.get(feat))))
        if not vals:
            continue

        for direction in ("xz_if_ge", "xz_if_le"):
            for t in vals:
                correct = 0
                preds = {}

                for r in train:
                    rule = {
                        "feature": feat,
                        "direction": direction,
                        "threshold": t,
                    }
                    p = pred(rule, r)
                    preds[r["file"]] = p
                    if p == r["label"]:
                        correct += 1

                acc = correct / len(train) if train else 0.0
                item = {
                    "feature": feat,
                    "class": classify_feature(feat),
                    "direction": direction,
                    "threshold": t,
                    "correct": correct,
                    "total": len(train),
                    "accuracy": acc,
                    "predictions": preds,
                }

                if best is None:
                    best = item
                elif (item["correct"], item["accuracy"]) > (best["correct"], best["accuracy"]):
                    best = item

    return best


def loocv(rows, features):
    preds = []
    correct = 0

    for hold in rows:
        train = [r for r in rows if r["file"] != hold["file"]]
        rule = best_rule(train, features)

        if rule is None:
            p = "unknown"
        else:
            p = pred(rule, hold)

        ok = p == hold["label"]
        correct += int(ok)

        preds.append({
            "file": hold["file"],
            "label": hold["label"],
            "predicted": p,
            "correct": ok,
            "rule": None if rule is None else {
                "feature": rule["feature"],
                "class": rule["class"],
                "direction": rule["direction"],
                "threshold": rule["threshold"],
                "train_accuracy": rule["accuracy"],
                "train_correct": rule["correct"],
                "train_total": rule["total"],
            }
        })

    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "predictions": preds,
    }


def main():
    data = json.loads(IN_JSON.read_text())

    rows_all = data["rows"]
    binary = [r for r in rows_all if r["label"] in ("bzip2", "xz")]

    all_features = numeric_features(binary)

    feature_sets = {
        "all_numeric": all_features,
        "no_metadata": [
            f for f in all_features
            if classify_feature(f) != "metadata"
        ],
        "normalized_or_distance_only": [
            f for f in all_features
            if classify_feature(f) == "normalized_or_distance"
        ],
        "no_counts_no_metadata": [
            f for f in all_features
            if classify_feature(f) not in ("metadata", "count_or_scale")
        ],
        "count_or_scale_only": [
            f for f in all_features
            if classify_feature(f) == "count_or_scale"
        ],
    }

    results = {}

    for name, fs in feature_sets.items():
        rule = best_rule(binary, fs)
        cv = loocv(binary, fs)

        results[name] = {
            "feature_count": len(fs),
            "best_in_sample_rule": rule,
            "loocv": cv,
        }

    # Main decision: we care about whether the signal survives without count/size leakage.
    strict = results["normalized_or_distance_only"]["loocv"]
    broad = results["all_numeric"]["loocv"]
    no_counts = results["no_counts_no_metadata"]["loocv"]

    if strict["correct"] >= 10:
        decision = "distance_signal_alive"
    elif strict["correct"] >= 9:
        decision = "distance_signal_borderline_alive"
    elif broad["correct"] >= 9 and strict["correct"] < 9:
        decision = "signal_depends_on_scale_or_overfit"
    else:
        decision = "anchor_distance_predictor_rejected_or_weak"

    out = {
        "ok": True,
        "date": "2026-07-01",
        "source": str(IN_JSON),
        "binary_files": len(binary),
        "decision": decision,
        "results": results,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = []
    lines.append("# ANCHOR_REPEAT_DISTANCE_PREDICTOR_AUDIT_V1")
    lines.append("")
    lines.append("Status: measured")
    lines.append("Date: 2026-07-01")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Audit whether `ANCHOR_REPEAT_DISTANCE_PROFILE_V1` is a real distance signal or an in-sample / scale-count artifact.")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"`{decision}`")
    lines.append("")
    lines.append("## Feature-set results")
    lines.append("")
    lines.append("| feature set | features | in-sample best feature | feature class | in-sample accuracy | LOOCV accuracy |")
    lines.append("|---|---:|---|---|---:|---:|")

    for name, r in results.items():
        rule = r["best_in_sample_rule"]
        cv = r["loocv"]

        if rule:
            best_feat = rule["feature"]
            cls = rule["class"]
            ins = f"{rule['correct']}/{rule['total']}"
        else:
            best_feat = "NA"
            cls = "NA"
            ins = "NA"

        lines.append(
            f"| `{name}` | {r['feature_count']} | `{best_feat}` | `{cls}` | "
            f"{ins} | {cv['correct']}/{cv['total']} |"
        )

    lines.append("")
    lines.append("## LOOCV details: normalized_or_distance_only")
    lines.append("")
    lines.append("| held-out file | label | predicted | correct | selected feature | direction | threshold |")
    lines.append("|---|---|---|---:|---|---|---:|")

    for p in results["normalized_or_distance_only"]["loocv"]["predictions"]:
        rule = p["rule"] or {}
        lines.append(
            f"| `{p['file']}` | {p['label']} | {p['predicted']} | {p['correct']} | "
            f"`{rule.get('feature', 'NA')}` | {rule.get('direction', 'NA')} | {rule.get('threshold', 'NA')} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If LOOCV collapses, the previous 9/11 was likely in-sample threshold fitting.")
    lines.append("- If only count/scale features work, the result is not a clean match-distance law.")
    lines.append("- If normalized distance features survive LOOCV, then the GLYPH/STRIDE bridge remains alive.")
    lines.append("")
    lines.append("## Non-claims")
    lines.append("")
    lines.append("- This is still Silesia-only.")
    lines.append("- This does not prove a production codec router.")
    lines.append("- This does not replace out-of-sample validation.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    d = []
    d.append("# GLYPH_ANCHOR_REPEAT_DISTANCE_PREDICTOR_AUDIT_DECISION_V1")
    d.append("")
    d.append("Status: measured gate")
    d.append("Date: 2026-07-01")
    d.append("")
    d.append("## Decision")
    d.append("")
    d.append(f"`{decision}`")
    d.append("")
    d.append("## Core numbers")
    d.append("")
    d.append(f"- all numeric LOOCV: `{broad['correct']}/{broad['total']}`")
    d.append(f"- no-count/no-metadata LOOCV: `{no_counts['correct']}/{no_counts['total']}`")
    d.append(f"- normalized/distance-only LOOCV: `{strict['correct']}/{strict['total']}`")
    d.append("")
    d.append("## Source report")
    d.append("")
    d.append("`benchmarks/results/ANCHOR_REPEAT_DISTANCE_PREDICTOR_AUDIT_V1.md`")
    d.append("")

    OUT_DECISION.write_text("\n".join(d), encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "decision": decision,
        "all_numeric_loocv": f"{broad['correct']}/{broad['total']}",
        "no_counts_no_metadata_loocv": f"{no_counts['correct']}/{no_counts['total']}",
        "normalized_distance_only_loocv": f"{strict['correct']}/{strict['total']}",
        "report": str(OUT_MD),
        "decision_file": str(OUT_DECISION),
    }, indent=2))


if __name__ == "__main__":
    main()
