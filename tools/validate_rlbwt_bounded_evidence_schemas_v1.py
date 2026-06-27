#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


HEX64 = re.compile(r"^[0-9a-f]{64}$")


def load_json(path: Path):
    return json.loads(path.read_text())


def require(cond, msg, errors):
    if not cond:
        errors.append(msg)


def validate_artifact(j, errors):
    require(j.get("artifact_version") == "RLBWT_BOUNDED_EVIDENCE_V1", "bad artifact_version", errors)
    require(j.get("profile") == "RLBWT_FULL_RUNTIME_PROFILE_V1", "bad profile", errors)

    q = j.get("query", {})
    require(isinstance(q.get("text"), str), "query.text missing", errors)
    require(isinstance(q.get("hex"), str), "query.hex missing", errors)
    require(isinstance(q.get("bytes"), int), "query.bytes missing", errors)
    require(isinstance(q.get("sha256"), str) and HEX64.match(q["sha256"]), "query.sha256 invalid", errors)

    sc = j.get("source_corpus", {})
    require(isinstance(sc.get("path"), str), "source_corpus.path missing", errors)
    require(isinstance(sc.get("bytes"), int), "source_corpus.bytes missing", errors)
    require(isinstance(sc.get("sha256"), str) and HEX64.match(sc["sha256"]), "source_corpus.sha256 invalid", errors)

    rf = j.get("runtime_files")
    require(isinstance(rf, list) and len(rf) >= 5, "runtime_files invalid", errors)

    names = {x.get("name") for x in rf or []}
    for name in [
        "bwt.rlbwt",
        "bwt.rlbwt.rank",
        "locate_core_s128.bin",
        "manifest.json",
        "rlbwt_full_runtime_manifest_v1.json",
    ]:
        require(name in names, f"runtime file missing: {name}", errors)

    r = j.get("retrieval", {})
    interval = r.get("fm_interval")
    require(isinstance(interval, list) and len(interval) == 2, "retrieval.fm_interval invalid", errors)

    for k in ["match_count", "total_possible_count", "max_offsets", "returned_count"]:
        require(isinstance(r.get(k), int), f"retrieval.{k} missing", errors)

    require(isinstance(r.get("bounded"), bool), "retrieval.bounded missing", errors)
    require(isinstance(r.get("offsets"), list), "retrieval.offsets missing", errors)

    if isinstance(interval, list) and len(interval) == 2 and isinstance(r.get("match_count"), int):
        require(r["match_count"] == interval[1] - interval[0], "match_count != r-l", errors)

    if isinstance(r.get("returned_count"), int) and isinstance(r.get("offsets"), list):
        require(r["returned_count"] == len(r["offsets"]), "returned_count != len(offsets)", errors)

    bc = j.get("byte_check", {})
    require(bc.get("all_returned_offsets_match_query") is True, "byte_check is not true", errors)


def validate_bundle_manifest(j, errors):
    require(j.get("bundle_version") == "RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1", "bad bundle_version", errors)
    require(j.get("artifact_version") == "RLBWT_BOUNDED_EVIDENCE_V1", "bad bundle artifact_version", errors)
    require(j.get("profile") == "RLBWT_FULL_RUNTIME_PROFILE_V1", "bad bundle profile", errors)

    for k in ["evidence", "source_corpus", "runtime_dir", "replay_command"]:
        require(isinstance(j.get(k), str), f"bundle {k} missing", errors)

    files = j.get("files")
    require(isinstance(files, list) and len(files) >= 8, "bundle files invalid", errors)

    paths = {x.get("path") for x in files or []}
    for path in [
        "README_REPLAY.md",
        "corpus.bin",
        "evidence.json",
        "runtime/bwt.rlbwt",
        "runtime/bwt.rlbwt.rank",
        "runtime/locate_core_s128.bin",
        "runtime/manifest.json",
        "runtime/rlbwt_full_runtime_manifest_v1.json",
    ]:
        require(path in paths, f"bundle file missing: {path}", errors)

    for rec in files or []:
        require(isinstance(rec.get("bytes"), int), f"bundle file bytes invalid: {rec}", errors)
        require(isinstance(rec.get("sha256"), str) and HEX64.match(rec["sha256"]), f"bundle file sha invalid: {rec}", errors)


def main() -> int:
    ap = argparse.ArgumentParser(description="Dependency-free schema smoke validator for RLBWT bounded evidence V1.")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--bundle-manifest", required=True)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    schema_paths = [
        root / "docs/schemas/RLBWT_BOUNDED_EVIDENCE_SCHEMA_V1.json",
        root / "docs/schemas/RLBWT_BOUNDED_EVIDENCE_BUNDLE_MANIFEST_SCHEMA_V1.json",
    ]

    errors = []

    for p in schema_paths:
        try:
            load_json(p)
        except Exception as e:
            errors.append(f"schema JSON parse failed {p}: {e}")

    validate_artifact(load_json(Path(args.artifact)), errors)
    validate_bundle_manifest(load_json(Path(args.bundle_manifest)), errors)

    result = {
        "ok": not errors,
        "artifact": str(Path(args.artifact)),
        "bundle_manifest": str(Path(args.bundle_manifest)),
        "schemas": [str(p) for p in schema_paths],
        "errors": errors,
    }

    print(json.dumps(result, indent=2, sort_keys=True))

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
