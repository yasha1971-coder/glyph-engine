#!/usr/bin/env python3
"""Validate self-reported GLYPH core benchmark records, never certify evidence."""
import argparse
import json
import math
import sys
from pathlib import Path

SCHEMA = "GLYPH_CORE_FRONTIER_V1"
STORAGE = ("payload", "search_index", "locate", "manifests_evidence", "recovery_metadata", "other_mandatory")
RESOURCE = ("build_peak_rss_bytes", "open_peak_rss_bytes", "index_build_workspace_bytes", "build_time_ms", "startup_ms", "decoded_bytes_total", "requested_bytes_total")
CORRECT = ("count_cases", "locate_cases", "absent_cases", "boundary_cases", "count_mismatches", "locate_mismatches", "boundary_mismatches")
SHA_KEYS = ("source_sha256", "workload_sha256")

class Invalid(ValueError):
    pass


def need(condition, explanation):
    if not condition:
        raise Invalid(explanation)


def integer(obj, name, *, minimum=0):
    v = obj.get(name)
    need(type(v) is int and v >= minimum, f"{name}: expected integer >= {minimum}")
    return v


def number(obj, name):
    v = obj.get(name)
    need(type(v) in (int, float) and math.isfinite(v) and v >= 0, f"{name}: nonnegative finite number required")
    return v


def hash64(v, name):
    need(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v), f"{name}: lowercase SHA-256 hex required")
    return v


def obj(v, name):
    need(isinstance(v, dict), f"{name}: object required")
    return v


def string(v, name):
    need(isinstance(v, str) and bool(v.strip()), f"{name}: nonempty string required")
    return v


def validate(record):
    r = obj(record, "record")
    need(r.get("schema") == SCHEMA, "unknown schema")
    dataset = obj(r.get("dataset"), "dataset")
    name = string(dataset.get("name"), "dataset.name")
    raw = integer(dataset, "raw_bytes", minimum=1)
    ssha = hash64(dataset.get("source_sha256"), "source_sha256")
    wsha = hash64(dataset.get("workload_sha256"), "workload_sha256")
    string(dataset.get("document_order_id"), "document_order_id")
    candidate = obj(r.get("candidate"), "candidate")
    candidate_name = string(candidate.get("name"), "candidate.name")
    sha = candidate.get("commit")
    need(isinstance(sha, str) and len(sha)==40 and all(c in '0123456789abcdef' for c in sha), "candidate.commit: git SHA-1 expected")
    need(candidate.get("scope") == "complete_searchable_store", "kernel/payload-only measurements are not full-store comparisons")
    profile = string(candidate.get("profile"), "candidate.profile")
    store = obj(r.get("storage_bytes"), "storage_bytes")
    need(set(store) == set(STORAGE), "storage_bytes must provide exactly all disjoint mandatory categories")
    total = sum(integer(store, k) for k in STORAGE)
    need(total > 0, "empty archive")
    correct = obj(r.get("correctness"), "correctness")
    need(all(k in correct for k in CORRECT), "correctness missing mandatory field")
    for k in CORRECT:
        integer(correct, k, minimum=1 if k in ("count_cases", "locate_cases", "absent_cases", "boundary_cases") else 0)
    need(correct.get("full_coverage") is True, "no complete coverage: absence cannot be proved")
    need(correct.get("independent_oracle") is True, "independent oracle required by contract (receipt verification still external)")
    need(hash64(correct.get("restored_sha256"), "restored_sha256") == ssha, "restore SHA-256 mismatches source")
    need(all(correct[k] == 0 for k in ("count_mismatches", "locate_mismatches", "boundary_mismatches")), "exact correctness mismatch")
    resources = obj(r.get("resources"), "resources")
    for k in RESOURCE:
        if k in ("build_peak_rss_bytes", "open_peak_rss_bytes", "decoded_bytes_total", "requested_bytes_total"):
            integer(resources, k, minimum=1)
        elif k == "index_build_workspace_bytes":
            integer(resources, k)
        else:
            number(resources, k)
    sample_stats = {}
    for kind in ("count", "locate", "extract"):
        vals = resources.get(kind + "_samples_ms")
        need(isinstance(vals, list) and len(vals) >= 20, f"{kind}: >=20 latency samples required")
        for v in vals:
            need(type(v) in (int, float) and math.isfinite(v) and v >= 0, f"{kind}: invalid sample")
        ordered = sorted(vals)
        # Nearest-rank empirical quantiles: p50 = sample ceil(0.50*n), p95 = ceil(0.95*n).
        sample_stats[kind + "_p50_ms"] = ordered[math.ceil(0.5 * len(ordered)) - 1]
        sample_stats[kind + "_p95_ms"] = ordered[math.ceil(0.95 * len(ordered)) - 1]
    provenance = obj(r.get("provenance"), "provenance")
    commands = provenance.get("commands")
    need(isinstance(commands, list) and bool(commands) and all(isinstance(x,str) and x.strip() for x in commands), "replay commands required")
    hash64(provenance.get("evidence_sha256"), "evidence_sha256")
    env = string(provenance.get("environment_id"), "environment_id")
    string(provenance.get("oracle_method"), "oracle_method")
    need(provenance.get("clean_source_tree") is True, "dirty source tree must be reported separately")
    return {
        "candidate": candidate_name,
        "commit": sha,
        "profile": profile,
        "dataset": name,
        "dataset_sha256": ssha,
        "workload_sha256": wsha,
        "environment_id": env,
        "raw_bytes": raw,
        "full_store_bytes": total,
        "full_store_ratio": total/raw,
        "count_p95_ms": sample_stats["count_p95_ms"],
        "locate_p95_ms": sample_stats["locate_p95_ms"],
        "extract_p95_ms": sample_stats["extract_p95_ms"],
        "open_peak_rss_bytes": resources["open_peak_rss_bytes"],
        "build_peak_rss_bytes": resources["build_peak_rss_bytes"],
        "index_build_workspace_bytes": resources["index_build_workspace_bytes"],
        "build_time_ms": resources["build_time_ms"],
        **sample_stats,
        "startup_ms": resources["startup_ms"],
        "decode_amplification": resources["decoded_bytes_total"]/resources["requested_bytes_total"],
        "verdict": "REPORTED_FIELDS_CONSISTENT_NOT_REPLAY_VERIFIED",
    }


def frontier(rows):
    need(bool(rows), "no records")
    key = lambda r: (r["dataset_sha256"],r["workload_sha256"],r["environment_id"])
    need(all(key(r)==key(rows[0]) for r in rows), "incomparable corpus, workload or environment")
    axes = ("full_store_bytes", "locate_p95_ms", "open_peak_rss_bytes", "decode_amplification")
    for i, r in enumerate(rows):
        dominating = [j for j, other in enumerate(rows) if j != i and all(other[k] <= r[k] for k in axes) and any(other[k] < r[k] for k in axes)]
        r["pareto_status"] = "DOMINATED" if dominating else "NONDOMINATED"
        r["dominated_by"] = [rows[j]["candidate"] for j in dominating]
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records",nargs="+",type=Path,help="candidate record JSON files")
    args=parser.parse_args(argv)
    try:
        rows=[]
        for path in args.records:
            rows.append(validate(json.loads(path.read_text(encoding="utf-8"))))
        result={"schema":SCHEMA,"interpretation":"STRUCTURAL_CONSISTENCY_ONLY; SOURCE, QUERIES, HASHES, AND REPLAY ARE NOT INDEPENDENTLY CHECKED", "records":frontier(rows)}
        print(json.dumps(result,sort_keys=True,indent=2))
        return 0
    except (Invalid, ValueError, OSError) as e:
        print(f"CORE FRONTIER REJECTED: {e}",file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
