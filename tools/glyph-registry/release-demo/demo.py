#!/usr/bin/env python3
"""Two offline receiver examples; the second includes an expected forgery rejection."""
from pathlib import Path
import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
import glyph_registry as registry
from verify_handoff import DATA_FILES

ROOT = Path(__file__).resolve().parent
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")


def require(value, reason):
    if not value:
        raise ValueError(reason)


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def execute(work, label, command, expected=0):
    stamp = time.perf_counter()
    record = {"command": command, "state": "running", "start_unix": time.time()}
    save(work / (label + ".command.json"), record)
    p = subprocess.run(command, capture_output=True, env=ENV)
    (work / (label + ".stdout")).write_bytes(p.stdout)
    (work / (label + ".stderr")).write_bytes(p.stderr)
    save(work / (label + ".command.json"), dict(record, state="complete", exit_code=p.returncode, seconds=time.perf_counter() - stamp))
    require(p.returncode == expected, label + " exit=" + str(p.returncode) + "; see recorded stderr")
    return p


def query_bytes(args):
    if args.query_file:
        query = Path(args.query_file).read_bytes()
    elif args.query is not None:
        query = args.query.encode("utf-8")
    elif args.command == "chain":
        query = (ROOT / "acceptance/chain.query.bin").read_bytes()
    else:
        raise ValueError("provide your exact --query or --query-file")
    require(1 <= len(query) <= 1024 and 0 not in query and 1 not in query, "nonempty UTF-8 query up to1024bytes without00/01 required")
    query.decode("utf-8")
    return query


def check_known_example(kind, query, result):
    example = json.loads((ROOT / "acceptance/EXPECTED.json").read_bytes())[kind]
    if hashlib.sha256(query).hexdigest() != example["query_sha256"]:
        return
    allowed = {(row["path"], row["offset"]) for row in example["full_matches"]}
    require(result["count"] == example["count"] and len(result["matches"]) == example["returned"], "predeclared count/cardinality mismatch")
    require(len({(row["path"], row["offset"]) for row in result["matches"]}) == len(result["matches"]) and all((row["path"], row["offset"]) in allowed for row in result["matches"]), "predeclared coordinate mismatch")


def checked_index(index, pins):
    require(registry.digest(index / "registry.json") == pins["registry_sha256"], "original registry identity differs")
    value = json.loads((index / "registry.json").read_bytes())
    for name, spec in value["files"].items():
        p = index / name
        require(p.is_file() and p.stat().st_size == spec["bytes"] and registry.digest(p) == spec["sha256"], "original index part differs: " + name)
    return pins["registry_sha256"]


def pack_data(kit, target, forge_count=False):
    overrides = {}
    if forge_count:
        core = json.loads((kit / "results/index-result.json").read_bytes())
        core["count"] += 1
        overrides["results/index-result.json"] = (json.dumps(core, sort_keys=True, separators=(",", ":")) + "\n").encode()
        mapping = json.loads((kit / "results/registry-result.json").read_bytes())
        mapping["count"] = core["count"]
        mapping["index_result_sha256"] = hashlib.sha256(overrides["results/index-result.json"]).hexdigest()
        overrides["results/registry-result.json"] = (json.dumps(mapping, sort_keys=True, indent=2) + "\n").encode()
    with target.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|", format=tarfile.GNU_FORMAT) as archive:
            for name in sorted(DATA_FILES):
                body = overrides.get(name)
                if body is None:
                    body = (kit / name).read_bytes()
                entry = tarfile.TarInfo("kit/" + name)
                entry.uid = entry.gid = 0
                entry.uname = entry.gname = "root"
                entry.mode = 0o644
                entry.mtime = 0
                entry.size = len(body)
                archive.addfile(entry, io.BytesIO(body))
    return registry.digest(target)


def run(args):
    registry.check_files()
    work = Path(args.work).resolve()
    require(work != ROOT and ROOT not in work.parents, "use a new work folder outside the immutable kit")
    work.mkdir(parents=True, exist_ok=False)
    if args.command == "standalone":
        query = query_bytes(args)
        (work / "query.bin").write_bytes(query)
        p = execute(work, "standalone", [sys.executable, str(ROOT / "glyph_registry.py"), "search", "--dataset", "Git", "--query-file", str(work / "query.bin"), "--limit", "10", "--output", str(work / "search")])
        result = json.loads(p.stdout)
        check_known_example("standalone", query, result)
        answer = {"ok": True, "scenario": "3", "count": result["count"], "returned": len(result["matches"]),
                  "matches": result["matches"], "byte_check": result["byte_check"], "files_verified": result["files_verified"]}
        save(work / "summary.json", answer)
        return answer

    events = []

    def event(stage, **fields):
        events.append(dict(stage=stage, time_unix=time.time(), **fields))
        save(work / "stages.json", events)

    # B's installation is copied from the author's authenticated archive before
    # A chooses a request or delivers an answer. No index/source is copied to B.
    receiver = work / "second-receiver"
    tools = receiver / "trusted-tools"
    tools.mkdir(parents=True)
    code = json.loads((ROOT / "TRUSTED_CODE.json").read_bytes())
    for name in code:
        dest = tools / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    shutil.copy2(ROOT / "TRUSTED_CODE.json", tools / "TRUSTED_CODE.json")
    anchor_sha = registry.digest(tools / "trust/Git-anchor.json")
    require(anchor_sha == json.loads((ROOT / "acceptance/EXPECTED.json").read_bytes())["anchor_sha256"], "predeclared author anchor changed")
    event("B_trusted_programs_and_publisher_anchor_fixed", anchor_sha256=anchor_sha, received_index=False)
    first = work / "first-receiver"
    first.mkdir()
    received = first / "index-kit"
    shutil.copytree(ROOT, received)
    pins = json.loads((received / "datasets/Git/pins.json").read_bytes())
    original = checked_index(received / "datasets/Git/index", pins)
    event("index_delivered_to_A", registry_sha256=original)
    query = query_bytes(args)  # Read/select the new query only after index delivery.
    (first / "request.bin").write_bytes(query)
    qsha = hashlib.sha256(query).hexdigest()
    event("new_query_selected_by_A", query_sha256=qsha, bytes=len(query))
    # This separate agreed request is B's query authority, never the answer claim.
    (receiver / "agreed-query.bin").write_bytes(query)
    save(receiver / "agreed-request.json", {"query_sha256": qsha, "limit": 10, "publisher_anchor_sha256": anchor_sha})
    p = execute(work, "A-create-result", [sys.executable, str(received / "glyph_registry.py"), "handoff", "--dataset", "Git", "--query-file", str(first / "request.bin"), "--limit", "10", "--output", str(first / "sender")])
    created = json.loads(p.stdout)
    require(checked_index(received / "datasets/Git/index", pins) == original, "query rebuilt/changed the received index")
    event("result_created_without_index_change", registry_sha256=original)
    incoming = receiver / "incoming"
    incoming.mkdir()
    good = incoming / "result-data.tar.gz"
    transport_sha = pack_data(first / "sender/kit", good)
    common = ["--expected-anchor-sha", anchor_sha, "--expected-query-sha", qsha, "--limit", "10", "--isolate"]
    p = execute(work, "B-verify-success", [sys.executable, str(tools / "verify_handoff.py"), "--archive", str(good), "--transport-sha", transport_sha, "--output", str(receiver / "positive"), *common])
    verified = json.loads(p.stdout)
    check_known_example("chain", query, verified)
    require(verified["count"] == created["count"] and verified["matches"] == created["matches"] and not verified["sender_code_executed"], "producer/independent verifier mismatch")
    event("B_independent_success", count=verified["count"], files=verified["files_verified"])
    forged = incoming / "forged-count-data.tar.gz"
    forged_sha = pack_data(first / "sender/kit", forged, forge_count=True)
    p = execute(work, "B-expected-rejection", [sys.executable, str(tools / "verify_handoff.py"), "--archive", str(forged), "--transport-sha", forged_sha, "--output", str(receiver / "negative"), *common], expected=1)
    error = json.loads(p.stderr)
    require(error.get("ok") is False and "count or coverage mismatch" in error["error"], "forgery rejected for an unexpected reason")
    event("B_forged_count_rejected", exit_code=p.returncode, transport_sha_recomputed=True)
    answer = {"ok": True, "scenario": "3->2->1", "count": verified["count"], "returned": len(verified["matches"]),
              "matches": verified["matches"], "files_verified": verified["files_verified"], "byte_check": True,
              "native_replay": True, "index_unchanged": True, "sender_code_executed": False,
              "expected_rejection": {"exit_code": 1, "reason_contains": "count or coverage mismatch"}}
    save(work / "summary.json", answer)
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["standalone", "chain"])
    query = parser.add_mutually_exclusive_group()
    query.add_argument("--query")
    query.add_argument("--query-file")
    parser.add_argument("--work", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error), "type": type(error).__name__}), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
