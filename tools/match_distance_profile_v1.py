#!/usr/bin/env python3
import argparse, hashlib, json, mmap, random, statistics, struct, subprocess
from pathlib import Path

ROOT = Path("/home/glyph/GLYPH_CPP_BACKEND")
WORK = ROOT / "benchmarks/work/match_distance_profile_v1"
OUT_JSON = ROOT / "benchmarks/results/MATCH_DISTANCE_PROFILE_V1.json"
OUT_MD = ROOT / "benchmarks/results/MATCH_DISTANCE_PROFILE_V1.md"
OUT_DECISION = ROOT / "docs/review/GLYPH_MATCH_DISTANCE_PROFILE_DECISION_V1.md"

EXPECTED = {"reymont": "bzip2", "webster": "xz"}
BOUNDARY = 900 * 1024

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()

def count_nul(p):
    n = 0
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            n += c.count(b"\x00")
    return n

def prepare(src, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    nul = count_nul(src)
    if nul:
        with open(src, "rb") as fi, open(out, "wb") as fo:
            for c in iter(lambda: fi.read(1024 * 1024), b""):
                fo.write(c.replace(b"\x00", b""))
        mode = "strip_nul_for_current_glyph_v0x_boundary"
    else:
        out.write_bytes(src.read_bytes())
        mode = "copy_no_nul"
    return {
        "source": str(src),
        "source_bytes": src.stat().st_size,
        "source_sha256": sha256_file(src),
        "source_nul_bytes": nul,
        "prepared": str(out),
        "prepared_bytes": out.stat().st_size,
        "prepared_sha256": sha256_file(out),
        "nul_strategy": mode,
    }

def build(corpus, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    builder = ROOT / "tools/build_glyph_index_v1.sh"
    subprocess.run([str(builder), str(corpus), str(outdir)], cwd=ROOT, check=True)
    return outdir / "sa.bin"

class SA:
    def __init__(self, corpus, sa):
        self.data = corpus.read_bytes()
        self.n = len(self.data)
        self.f = open(sa, "rb")
        self.m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.sa_n = len(self.m) // 4

    def close(self):
        self.m.close()
        self.f.close()

    def at(self, i):
        return struct.unpack_from("<I", self.m, i * 4)[0]

    def cmp(self, i, pat):
        p = self.at(i)
        frag = self.data[p:p+len(pat)] if p < self.n else b""
        return -1 if frag < pat else (1 if frag > pat else 0)

    def interval(self, pat):
        lo, hi = 0, self.sa_n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cmp(mid, pat) < 0:
                lo = mid + 1
            else:
                hi = mid
        l = lo
        lo, hi = 0, self.sa_n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cmp(mid, pat) <= 0:
                lo = mid + 1
            else:
                hi = mid
        return l, lo

    def locate(self, l, r, max_count):
        if r - l < 2 or r - l > max_count:
            return []
        xs = []
        for i in range(l, r):
            p = self.at(i)
            if p < self.n:
                xs.append(p)
        xs.sort()
        return xs

def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1 - (pos - lo)) + xs[hi] * (pos - lo)

def sample(index, k, wanted=100, probes=5000, max_count=5000, seed=1):
    rng = random.Random(seed)
    seen, rows = set(), []
    n = index.n
    positions = set()

    step = max(1, n // max(1, probes // 2))
    for p in range(0, max(1, n-k), step):
        positions.add(p)
    for _ in range(probes):
        positions.add(rng.randrange(0, max(1, n-k)))

    for p in sorted(positions):
        pat = index.data[p:p+k]
        if len(pat) != k or b"\x00" in pat or pat in seen:
            continue
        seen.add(pat)

        l, r = index.interval(pat)
        pos = index.locate(l, r, max_count)
        if len(pos) < 2:
            continue

        diffs = [b-a for a,b in zip(pos, pos[1:])]
        if not diffs:
            continue

        rows.append({
            "k": k,
            "pattern_hex": pat.hex(),
            "count": len(pos),
            "median_distance": statistics.median(diffs),
            "mean_distance": sum(diffs) / len(diffs),
            "p90_distance": pct(diffs, 0.90),
            "far_gt_900kb_fraction": sum(1 for d in diffs if d > BOUNDARY) / len(diffs),
            "near_le_64kb_fraction": sum(1 for d in diffs if d <= 64*1024) / len(diffs),
        })

        if len(rows) >= wanted:
            break
    return rows

def summary(rows):
    if not rows:
        return {"patterns": 0, "prediction": "unknown"}
    med = [r["median_distance"] for r in rows]
    far = [r["far_gt_900kb_fraction"] for r in rows]
    near = [r["near_le_64kb_fraction"] for r in rows]
    m = statistics.median(med)
    f = statistics.median(far)
    pred = "xz" if (m > BOUNDARY or f >= 0.5) else "bzip2"
    return {
        "patterns": len(rows),
        "median_of_median_distances": m,
        "p90_of_median_distances": pct(med, 0.90),
        "median_far_gt_900kb_fraction": f,
        "median_near_le_64kb_fraction": statistics.median(near),
        "prediction": pred,
    }

def profile(name, src):
    case = WORK / name
    prepared = case / f"{name}.prepared.bin"
    prep = prepare(src, prepared)
    sa_path = build(prepared, case / "index")
    idx = SA(prepared, sa_path)

    all_rows, by_k = [], {}
    try:
        seed = int(prep["prepared_sha256"][:8], 16)
        for k in [8, 12, 16]:
            rows = sample(idx, k, seed=seed+k)
            all_rows += rows
            by_k[str(k)] = {"summary": summary(rows), "sample": rows[:10]}
    finally:
        idx.close()

    overall = summary(all_rows)
    expected = EXPECTED.get(name, "unknown")
    correct = expected == overall["prediction"] if overall["prediction"] != "unknown" else None

    return {
        "name": name,
        "expected": expected,
        "prediction": overall["prediction"],
        "correct": correct,
        "prep": prep,
        "overall": overall,
        "by_k": by_k,
    }

def write_reports(results):
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# MATCH_DISTANCE_PROFILE_V1",
        "",
        "Status: measured",
        "Date: 2026-07-01",
        "",
        "## Purpose",
        "",
        "Test whether GLYPH-derived match-distance separates `reymont` and `webster`, where entropy/r/n/BWT-r/n/profile-std failed.",
        "",
        "## Boundary",
        "",
        "Current GLYPH v0.x is sentinel-safe. Files with `0x00` are stripped in working copies and recorded.",
        "",
        "## Results",
        "",
        "| file | expected | predicted | correct | source bytes | NUL | prepared bytes | patterns | median distance | p90 median | far>900KB median | near<=64KB median |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in results:
        o, p = r["overall"], r["prep"]
        def f(x):
            if x is None: return "NA"
            if isinstance(x, float): return f"{x:.6f}"
            return str(x)
        lines.append(
            f"| `{r['name']}` | {r['expected']} | {r['prediction']} | {r['correct']} | "
            f"{p['source_bytes']} | {p['source_nul_bytes']} | {p['prepared_bytes']} | "
            f"{o.get('patterns',0)} | {f(o.get('median_of_median_distances'))} | "
            f"{f(o.get('p90_of_median_distances'))} | {f(o.get('median_far_gt_900kb_fraction'))} | "
            f"{f(o.get('median_near_le_64kb_fraction'))} |"
        )

    lines += [
        "",
        "## Per-k summary",
        "",
    ]

    for r in results:
        lines.append(f"### {r['name']}")
        lines.append("")
        lines.append("| k | patterns | median distance | far>900KB median | prediction |")
        lines.append("|---:|---:|---:|---:|---|")
        for k, v in r["by_k"].items():
            s = v["summary"]
            lines.append(
                f"| {k} | {s.get('patterns',0)} | {s.get('median_of_median_distances','NA')} | "
                f"{s.get('median_far_gt_900kb_fraction','NA')} | {s.get('prediction','unknown')} |"
            )
        lines.append("")

    known = [r for r in results if r["correct"] is not None]
    ok = sum(1 for r in known if r["correct"])

    lines += [
        "## Interpretation",
        "",
        f"Known-label gate accuracy: `{ok}/{len(known)}`.",
        "",
        "If this separates `reymont` and `webster`, match-distance remains alive as a structural feature.",
        "If it does not, do not build claims around it.",
        "",
        "## Non-claims",
        "",
        "- This is not a production codec router.",
        "- This is not binary-safe GLYPH production support.",
        "- This does not replace compressor trials.",
        "- This is a falsification gate.",
        "",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    dec = [
        "# GLYPH_MATCH_DISTANCE_PROFILE_DECISION_V1",
        "",
        "Status: measured gate",
        "Date: 2026-07-01",
        "",
        f"Gate accuracy: `{ok}/{len(known)}`.",
        "",
    ]
    for r in results:
        dec.append(
            f"- `{r['name']}`: expected `{r['expected']}`, predicted `{r['prediction']}`, "
            f"median distance `{r['overall'].get('median_of_median_distances','NA')}`"
        )
    dec += [
        "",
        "Source report: `benchmarks/results/MATCH_DISTANCE_PROFILE_V1.md`",
        "",
    ]
    OUT_DECISION.write_text("\n".join(dec), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("silesia_dir", nargs="?", default="/tmp/silesia_check")
    args = ap.parse_args()

    silesia = Path(args.silesia_dir)
    results = []
    for name in ["reymont", "webster"]:
        src = silesia / name
        if not src.exists():
            raise SystemExit(f"missing {src}")
        print(f"[profile] {name}")
        results.append(profile(name, src))

    write_reports(results)
    print(json.dumps({
        "ok": True,
        "report": str(OUT_MD),
        "decision": str(OUT_DECISION),
        "files": len(results)
    }, indent=2))

if __name__ == "__main__":
    main()
