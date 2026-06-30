#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path("/home/glyph/GLYPH_CPP_BACKEND")
WORK = ROOT / "benchmarks/work/silesia_per_file_rlbwt_v1"
OUT_JSON = ROOT / "benchmarks/results/SILESIA_PER_FILE_RLBWT_PROFILE_V1.json"
OUT_JSONL = ROOT / "benchmarks/results/SILESIA_PER_FILE_RLBWT_PROFILE_V1.jsonl"
OUT_MD = ROOT / "benchmarks/results/SILESIA_PER_FILE_RLBWT_PROFILE_V1.md"
OUT_DECISION = ROOT / "docs/review/GLYPH_SILESIA_FORMAT_RLBWT_DECISION_V1.md"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def file_stats(path):
    counts = Counter()
    n = 0
    nul = 0
    printable = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            n += len(chunk)
            nul += chunk.count(b"\x00")
            printable += sum(1 for b in chunk if 32 <= b <= 126 or b in (9, 10, 13))
            counts.update(chunk)

    entropy = 0.0
    if n:
        for c in counts.values():
            p = c / n
            entropy -= p * math.log2(p)

    return {
        "bytes": n,
        "sha256": sha256_file(path),
        "nul_bytes": nul,
        "alphabet_size": len(counts),
        "printable_fraction": printable / n if n else 0.0,
        "entropy_bits_per_byte": entropy,
    }

def measure_bwt_runs(path):
    n = 0
    runs = 0
    prev = None
    alphabet = set()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            for b in chunk:
                alphabet.add(b)
                if prev is None or b != prev:
                    runs += 1
                    prev = b
                n += 1

    rn = runs / n if n else None
    avg = n / runs if runs else None

    if rn is None:
        cls = "empty"
    elif rn <= 0.001:
        cls = "extremely_repetitive"
    elif rn <= 0.01:
        cls = "highly_repetitive"
    elif rn <= 0.05:
        cls = "repetitive"
    elif rn <= 0.15:
        cls = "moderately_repetitive"
    else:
        cls = "not_run_compressible"

    return {
        "ok": True,
        "path": str(path),
        "bytes": n,
        "runs": runs,
        "r_over_n": rn,
        "avg_run_len": avg,
        "alphabet_size": len(alphabet),
        "classification": cls,
        "sha256": sha256_file(path),
    }

def run_cmd(cmd):
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return p.returncode, p.stdout

def safe_name(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)

def build_index(corpus, out_dir):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ROOT / "tools/build_glyph_index_v1.sh"),
        str(corpus),
        str(out_dir),
    ]
    return run_cmd(cmd)

def choose_mv_params(size_bytes):
    mib = size_bytes // (1024 * 1024)
    if mib < 5:
        return None

    version_mib = min(16, max(1, mib // 4))
    while version_mib >= 1 and version_mib * 4 > mib:
        version_mib -= 1

    if version_mib < 1:
        return None

    shift_mib = max(1, version_mib // 8)

    if version_mib + shift_mib * 3 > mib:
        return None

    return {
        "versions": 4,
        "version_mib": version_mib,
        "nonoverlap_shift_mib": version_mib,
        "overlap_shift_mib": shift_mib,
    }

def make_collection(source, out, manifest, version_mib, shift_mib, versions):
    maker = ROOT / "tools/make_multiversion_collection_v1.py"
    if not maker.exists():
        return 1, "missing tools/make_multiversion_collection_v1.py"

    cmd = [
        "python3", str(maker),
        "--source", str(source),
        "--out", str(out),
        "--manifest", str(manifest),
        "--version-mib", str(version_mib),
        "--shift-mib", str(shift_mib),
        "--versions", str(versions),
        "--label", "silesia_per_file",
    ]
    return run_cmd(cmd)

def process_file(path):
    name = path.name
    case_dir = WORK / safe_name(name)
    case_dir.mkdir(parents=True, exist_ok=True)

    st = file_stats(path)

    row = {
        "file": name,
        "path": str(path),
        "stats": st,
        "single": None,
        "multiversion": None,
        "skip_reason": None,
    }

    if st["bytes"] == 0:
        row["skip_reason"] = "empty_file"
        return row

    if st["nul_bytes"] > 0:
        row["skip_reason"] = "contains_0x00_current_glyph_v0x_boundary"
        return row

    code, log = build_index(path, case_dir / "single_index")
    (case_dir / "single_build.log").write_text(log)

    if code != 0:
        row["skip_reason"] = "single_index_build_failed"
        return row

    bwt = case_dir / "single_index/bwt.bin"
    if not bwt.exists():
        row["skip_reason"] = "single_bwt_missing"
        return row

    row["single"] = measure_bwt_runs(bwt)

    params = choose_mv_params(st["bytes"])
    if not params:
        row["multiversion"] = {"ok": False, "reason": "too_small_for_multiversion_probe"}
        return row

    mv = {
        "ok": True,
        "params": params,
        "nonoverlap": None,
        "overlap": None,
        "improvement_factor": None,
        "decision": None,
    }

    for label, shift in [
        ("nonoverlap", params["nonoverlap_shift_mib"]),
        ("overlap", params["overlap_shift_mib"]),
    ]:
        mv_dir = case_dir / label
        mv_dir.mkdir(parents=True, exist_ok=True)

        corpus = mv_dir / "collection.bin"
        manifest = mv_dir / "collection_manifest.json"
        index = mv_dir / "index"

        code, log = make_collection(
            path,
            corpus,
            manifest,
            params["version_mib"],
            shift,
            params["versions"],
        )
        (mv_dir / "make_collection.log").write_text(log)

        if code != 0:
            mv[label] = {"ok": False, "reason": "collection_build_failed"}
            continue

        code, log = build_index(corpus, index)
        (mv_dir / "index_build.log").write_text(log)

        if code != 0:
            mv[label] = {"ok": False, "reason": "index_build_failed"}
            continue

        bwt = index / "bwt.bin"
        r = measure_bwt_runs(bwt)
        r["collection_bytes"] = corpus.stat().st_size
        r["shift_mib"] = shift
        mv[label] = r

    if mv["nonoverlap"] and mv["overlap"] and mv["nonoverlap"].get("ok") and mv["overlap"].get("ok"):
        no = mv["nonoverlap"]["r_over_n"]
        ov = mv["overlap"]["r_over_n"]
        mv["improvement_factor"] = no / ov if ov else None

        if ov <= 0.05 and mv["improvement_factor"] and mv["improvement_factor"] >= 2.0:
            mv["decision"] = "multiversion_signal_alive"
        elif ov <= 0.10:
            mv["decision"] = "weak_or_moderate_multiversion_signal"
        else:
            mv["decision"] = "no_strong_multiversion_signal"

    row["multiversion"] = mv
    return row

def fnum(x, digits=8):
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"

def find_silesia_dir(arg):
    if arg:
        p = Path(arg).resolve()
        if p.is_dir():
            return p
        raise SystemExit(f"missing Silesia dir: {p}")

    for d in [
        "/tmp/silesia_check",
        "/tmp/silesia",
        "/home/glyph/silesia",
        "/home/glyph/GLYPH_CPP_BACKEND/silesia",
        "/home/glyph/GLYPH_CPP_BACKEND/silesia_data",
    ]:
        p = Path(d)
        if p.is_dir():
            return p.resolve()

    raise SystemExit("Silesia directory not found")

def write_reports(silesia_dir, results):
    OUT_JSON.write_text(json.dumps({
        "ok": True,
        "silesia_dir": str(silesia_dir),
        "results": results,
    }, indent=2))

    OUT_JSONL.write_text("\n".join(json.dumps(r, sort_keys=True) for r in results) + "\n")

    indexed = [r for r in results if r["single"]]
    skipped = [r for r in results if r["skip_reason"]]
    ranked = sorted(indexed, key=lambda r: r["single"]["r_over_n"])

    mv_ranked = sorted(
        [
            r for r in results
            if r.get("multiversion")
            and r["multiversion"].get("ok")
            and r["multiversion"].get("overlap")
            and r["multiversion"]["overlap"].get("ok")
        ],
        key=lambda r: r["multiversion"]["overlap"]["r_over_n"],
    )

    lines = []
    lines.append("# SILESIA_PER_FILE_RLBWT_PROFILE_V1")
    lines.append("")
    lines.append("Status: measured")
    lines.append("Date: 2026-06-28")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Measure Silesia files separately to identify which formats produce stronger BWT run signals for possible RLBWT/r-index direction.")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("Current GLYPH v0.x is sentinel-safe only. Files containing `0x00` are skipped for indexing.")
    lines.append("")
    lines.append("## Silesia directory")
    lines.append("")
    lines.append(f"`{silesia_dir}`")
    lines.append("")
    lines.append("## Single-file results")
    lines.append("")
    lines.append("| file | bytes | nul | entropy | printable | single r/n | avg run | classification | decision |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")

    for r in sorted(results, key=lambda x: x["file"]):
        st = r["stats"]
        if r["single"]:
            rn = fnum(r["single"]["r_over_n"])
            avg = f'{r["single"]["avg_run_len"]:.3f}'
            cls = r["single"]["classification"]
            if r["single"]["r_over_n"] <= 0.05:
                decision = "single_corpus_signal"
            elif r["single"]["r_over_n"] <= 0.15:
                decision = "weak_or_moderate"
            else:
                decision = "poor"
        else:
            rn = "NA"
            avg = "NA"
            cls = "skipped"
            decision = r["skip_reason"] or "not_measured"

        lines.append(
            f"| `{r['file']}` | {st['bytes']} | {st['nul_bytes']} | "
            f"{st['entropy_bits_per_byte']:.4f} | {st['printable_fraction']:.4f} | "
            f"{rn} | {avg} | {cls} | {decision} |"
        )

    lines.append("")
    lines.append("## Single-file ranking")
    lines.append("")
    lines.append("| rank | file | single r/n | avg run | classification |")
    lines.append("|---:|---|---:|---:|---|")

    for i, r in enumerate(ranked, 1):
        s = r["single"]
        lines.append(f"| {i} | `{r['file']}` | {fnum(s['r_over_n'])} | {s['avg_run_len']:.3f} | {s['classification']} |")

    lines.append("")
    lines.append("## Synthetic multiversion overlap probe")
    lines.append("")
    lines.append("| file | version MiB | overlap shift MiB | nonoverlap r/n | overlap r/n | improvement | decision |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")

    for r in mv_ranked:
        mv = r["multiversion"]
        p = mv["params"]
        no = mv["nonoverlap"]
        ov = mv["overlap"]
        imp = mv["improvement_factor"]
        lines.append(
            f"| `{r['file']}` | {p['version_mib']} | {p['overlap_shift_mib']} | "
            f"{fnum(no['r_over_n'])} | {fnum(ov['r_over_n'])} | {imp:.3f}x | {mv['decision']} |"
        )

    lines.append("")
    lines.append("## Skipped files")
    lines.append("")
    lines.append("| file | bytes | nul bytes | reason |")
    lines.append("|---|---:|---:|---|")

    for r in skipped:
        st = r["stats"]
        lines.append(f"| `{r['file']}` | {st['bytes']} | {st['nul_bytes']} | {r['skip_reason']} |")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `single r/n <= 0.05` means the file itself has strong RLBWT potential.")
    lines.append("- `overlap r/n <= 0.05` means version-like collections for that format may be promising.")
    lines.append("- High `r/n` means do not use that format as a compact-index argument.")
    lines.append("")
    lines.append("## Non-claims")
    lines.append("")
    lines.append("- This does not prove production RLBWT memory usage.")
    lines.append("- This does not prove binary-safe production support.")
    lines.append("- This does not prove real versioned workloads behave the same.")

    OUT_MD.write_text("\n".join(lines) + "\n")

    d = []
    d.append("# GLYPH_SILESIA_FORMAT_RLBWT_DECISION_V1")
    d.append("")
    d.append("Status: measured decision")
    d.append("Date: 2026-06-28")
    d.append("")
    d.append("## Decision")
    d.append("")
    if ranked:
        best = ranked[0]
        d.append(f"Best single-file Silesia signal: `{best['file']}` with r/n = `{fnum(best['single']['r_over_n'])}`.")
    else:
        d.append("No sentinel-safe Silesia file was indexed successfully.")
    d.append("")
    if mv_ranked:
        best = mv_ranked[0]
        mv = best["multiversion"]
        d.append(f"Best multiversion Silesia signal: `{best['file']}` with overlap r/n = `{fnum(mv['overlap']['r_over_n'])}` and improvement `{mv['improvement_factor']:.3f}x`.")
    else:
        d.append("No multiversion Silesia overlap probe was completed.")
    d.append("")
    d.append("Only promote RLBWT for a format if measured `r/n <= 0.05` on a named, reproducible corpus or versioned collection.")
    d.append("")
    d.append("Source report: `benchmarks/results/SILESIA_PER_FILE_RLBWT_PROFILE_V1.md`")
    d.append("")

    OUT_DECISION.write_text("\n".join(d) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("silesia_dir", nargs="?")
    args = ap.parse_args()

    silesia = find_silesia_dir(args.silesia_dir)
    files = sorted([p for p in silesia.iterdir() if p.is_file()])

    if not files:
        raise SystemExit(f"no files in {silesia}")

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)

    results = []
    for p in files:
        print(f"[file] {p.name}")
        results.append(process_file(p))

    write_reports(silesia, results)

    print(json.dumps({
        "ok": True,
        "silesia_dir": str(silesia),
        "files": len(results),
        "report": str(OUT_MD),
        "decision": str(OUT_DECISION),
    }, indent=2))

if __name__ == "__main__":
    main()
