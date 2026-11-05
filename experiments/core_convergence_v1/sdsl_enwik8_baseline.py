#!/usr/bin/env python3
"""Pinned external SDSL-lite CSA WT comparator on canonical enwik8.

Index/restore/locate semantics only. SDSL has no GLYPH P/R/O proof contract,
so its file size is not automatically a proof-carrying full-store cost.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from enwik8_loc1_loc2_full_replay import (
    ENWIK8, read_official_enwik8, canonical, file_sha, raw_oracle, run,
)

SDSL_SHA = "c32874cb2d8524119f25f3b501526fe692df29f4"


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sdsl", type=Path, required=True)
    p.add_argument("--wrapper", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require(subprocess.check_output(
        ["git", "-C", str(args.sdsl), "rev-parse", "HEAD"],
        text=True).strip() == SDSL_SHA, "unexpected SDSL source commit")
    require(args.wrapper.is_file(), "SDSL wrapper binary not built")
    out = args.output.resolve()
    require(not out.exists(), "refuse to overwrite evidence directory")
    out.mkdir(parents=True)
    w = out / "work"
    w.mkdir()
    commands = []
    evidence = {
        "schema": "GLYPH_EXTERNAL_SDSL_CSA_WT_ENWIK8_RESEARCH_V1",
        "sdsl_upstream_sha": SDSL_SHA,
        "source": ENWIK8,
        "source_kind": "canonical_enwik8_single_string",
        "status": "NOT_EXECUTED",
    }
    try:
        source = read_official_enwik8(w, commands)
        index = w / "enwik8.sdsl_csa_wt"
        built = run([args.wrapper, "build", source, index], commands, timeout=1800)
        build_report = json.loads(built.stdout.strip())
        require(build_report["source_bytes"] == ENWIK8["bytes"], "SDSL source size mismatch")
        require(build_report["index_bytes"] == index.stat().st_size, "SDSL size mismatch")
        # Exact workload bytes match the LOC1/LOC2 enwik8 experiment.
        raw = source.read_bytes()
        patterns = [raw[i:i+24] for i in
                    (1200, 1000000, 13579113, 27890000, 55432100, 83567543)]
        patterns += [b"\xff\xff\xffGLYPH-no-match", b"\x00\xff", b"\xff"]
        workload = w / "queries.hex"
        workload.write_text("".join(x.hex()+"\n" for x in patterns))
        queried = run([args.wrapper, "query", index, workload], commands, timeout=900)
        query_lines = [json.loads(line) for line in queried.stdout.splitlines()
                       if line.strip()]
        require(len(query_lines) == len(patterns), "external comparator incomplete output")
        measurements = []
        for pattern, row in zip(patterns, query_lines):
            expected = raw_oracle(raw, pattern)
            require(row["pattern_hex"] == pattern.hex(), "wrong query bytes")
            require(row["count"] == len(expected), "SDSL count vs raw mismatch")
            require(row["offsets"] == expected, "SDSL locate vs raw mismatch")
            measurements.append({
                "pattern_sha256": hashlib.sha256(pattern).hexdigest(),
                "pattern_bytes": len(pattern),
                "count": len(expected),
                "query_seconds": row["query_seconds"],
            })
        # Stronger than count/locate: exact byte extraction using ONLY SDSL index.
        heldout = w / "enwik8.original-held-out"
        source.rename(heldout)
        require(not source.exists(), "original source still at indexed path")
        restore = w / "sdsl_restored_enwik8"
        run([args.wrapper, "restore", index, restore], commands, timeout=1200)
        require(restore.stat().st_size == ENWIK8["bytes"]
                and file_sha(restore) == ENWIK8["sha256"],
                "SDSL CSA did not restore original bytes")
        manifest = {
            "format": "EXTERNAL_SDSL_CSA_WT_RESEARCH_MANIFEST_V0",
            "note": "NOT a GLYPH-authenticated or independently anchored proof",
            "sdsl_commit": SDSL_SHA, "source_sha256": ENWIK8["sha256"],
            "index_sha256": file_sha(index), "index_bytes": index.stat().st_size,
        }
        mp = w / "sdsl-research-manifest.json"
        mp.write_bytes(canonical(manifest))
        evidence.update({
            "status": "EXTERNAL_SDSL_CSA_WT_ENWIK8_SOURCE_COUNT_LOCATE_RESTORE_PASS",
            "source_sha256": ENWIK8["sha256"],
            "index_sha256": file_sha(index),
            "index_bytes": index.stat().st_size,
            "index_only_ratio": index.stat().st_size/ENWIK8["bytes"],
            "research_manifest_bytes": mp.stat().st_size,
            "index_and_research_manifest_bytes": index.stat().st_size+mp.stat().st_size,
            "build_seconds": build_report["build_seconds"],
            "query_cases": len(measurements),
            "measurements": measurements,
            "byte_restore_without_source_at_original_path": True,
            "limits": [
                "Single continuous enwik8; no field mask, document identities or Version currentness.",
                "SDSL index size not a completed GLYPH-level authenticated evidence package.",
                "Workload consists of nine length-identified patterns; no comprehensive latency claim.",
                "No external root witness, no independent signed query proof, no P/R/O parity.",
                "SDSL index and research manifest bytes are counted; build files and executable excluded.",
                "Cross-workflow wall/RSS times must not be compared without hardware/workload alignment.",
            ],
        })
    except Exception as exc:
        evidence.update({"status": "BLOCKED", "error": str(exc)})
    finally:
        (out / "result.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        (out / "commands.json").write_text(json.dumps(commands, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k:evidence.get(k) for k in
                      ("status", "error", "index_bytes", "index_only_ratio",
                       "query_cases", "byte_restore_without_source_at_original_path")},
                     sort_keys=True))
    return 0 if evidence["status"].startswith("EXTERNAL_SDSL_CSA_WT_ENWIK8_SOURCE_COUNT_LOCATE_RESTORE_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
