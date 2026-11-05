#!/usr/bin/env python3
"""LOC1-vs-LOC2: same RLB2/RLR2, exact SA oracle and hostile research binding.

Uses original historical V2 tools from an exact pinned checkout. Only a small
synthetic correctness gate, NOT the 100MB enwik8 size/latency benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REFERENCE = "c33cb493bc75f13ee4a904fabb190a29a196c94b"
SA_HEADER = 60
LOC2_HDR = struct.Struct("<4sIQIIQ")


def require(ok, description):
    if not ok:
        raise AssertionError(description)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec and spec.loader, "module loader missing")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def direct_offsets(source, pattern):
    positions = []
    start = 0
    while True:
        at = source.find(pattern, start)
        if at < 0:
            return positions
        positions.append(at)
        start = at + 1


def run(*command, ok=True):
    r = subprocess.run([str(x) for x in command], capture_output=True,
                       text=True, timeout=150)
    if ok:
        require(r.returncode == 0, f"failed {command}: {r.stderr[-1500:]}")
    return r


def compare(ref, output):
    checkout = ref.resolve()
    require(subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True).strip() == REFERENCE, "wrong historical source")
    tools = checkout / "tools"
    vault = checkout / "experiments/personal_vault_v0"
    for path in (str(tools), str(vault)):
        if path not in sys.path:
            sys.path.insert(0, path)
    gate = module(tools / "check_rlbwt_query_v2.py", "pinned_rlb2_gate")
    query = module(tools / "rlbwt_query_v2.py", "rlbwt_query_v2")
    loc2tool = module(vault / "loc2_experimental.py", "pinned_loc2_builder")
    query2 = module(vault / "query_loc2_experimental.py", "pinned_loc2_query")
    temp = output / "fixture"
    temp.mkdir()
    fixture = gate.build_fixture(temp,
                                 tools / "rlbwt_container_v2.py",
                                 tools / "rlbwt_rank_v2.py")
    sa = temp / "fixture.sa.binary_v1"
    vals = fixture["suffix_array"]
    sa.write_bytes(b"\x00" * SA_HEADER
                   + struct.pack("<" + "I" * len(vals), *vals))
    loc2 = temp / "locate.loc2"
    loc2tool.build(sa, len(vals), fixture["sample_step"], loc2)
    loc2tool.verify(sa, loc2)
    expected = LOC2_HDR.size + len(range(0, len(vals), fixture["sample_step"])) * 4
    require(loc2.stat().st_size == expected, "LOC2 compact geometry mismatch")
    bindings = {
        "corpus_sha256": hashlib.sha256(fixture["source"]).hexdigest(),
        "rlb2_sha256": sha(fixture["rlb2"]),
        "rlr2_sha256": sha(fixture["rlr2"]),
        "loc1_sha256": sha(fixture["locate"]),
        "loc2_sha256": sha(loc2),
    }
    data = fixture["source"]
    patterns = [b"\x00", b"\xff", b"banana", b"bandana",
                b"\xff\x00", b"na", b"_", b"\x00banana", b"absent",
                b"banana_bandana", b"na!", b"an"]
    cases = 0
    maximum_lf = 0
    timings = {"LOC1": [], "LOC2": []}
    r1 = query.QueryRuntime(fixture["rlb2"], fixture["rlr2"], fixture["locate"])
    r2 = query2.Runtime2(fixture["rlb2"], fixture["rlr2"], loc2)
    try:
        for pattern in patterns:
            oracle = direct_offsets(data, pattern)
            for limit in (-1, 0, 1, 2):
                for repeat in range(3):
                    a = r1.query(pattern, limit)
                    b = r2.query(pattern, limit)
                    expected_count = len(oracle)
                    # Bounded locate consumes the FIRST FM rows then sorts the
                    # selected offsets. This is not the text-order top-k.
                    require(a["count"] == b["count"] == expected_count,
                            f"count parity failure {pattern!r} limit={limit}: "
                            f"{a['count']} / {b['count']} / {expected_count}")
                    require(a["locate_offsets"] == b["locate_offsets"],
                            f"LOC1 vs LOC2 mismatch {pattern!r} limit={limit}: "
                            f"{a['locate_offsets']} != {b['locate_offsets']}")
                    if limit == -1:
                        require(a["locate_offsets"] == oracle,
                                f"full raw-byte oracle mismatch {pattern!r}")
                    else:
                        require(len(a["locate_offsets"]) == min(limit, expected_count)
                                and len(set(a["locate_offsets"])) == len(a["locate_offsets"])
                                and set(a["locate_offsets"]).issubset(set(oracle)),
                                f"bounded offsets not an exact FM-row subset {pattern!r}")
                    require(a["fm_interval"] == b["fm_interval"], "FM interval mismatch")
                    require(a["maximum_lf_steps"] == b["maximum_lf_steps"],
                            "LF path mismatch")
                    timings["LOC1"].append(a["query_elapsed_ns"])
                    timings["LOC2"].append(b["query_elapsed_ns"])
                    maximum_lf = max(maximum_lf, a["maximum_lf_steps"])
                    cases += 1
    finally:
        r1.close()
        r2.close()
    # Authenticated research boundary: the original LOC2 reader does not check
    # external pinned SHA or source SA. Enforce a separate immutable pin.
    pin = bindings["loc2_sha256"]
    p = bytearray(loc2.read_bytes())
    p[-1] ^= 1
    loc2.write_bytes(p)
    require(sha(loc2) != pin, "mutated LOC2 not rejected by pinned SHA")
    bad = run(sys.executable, vault / "loc2_experimental.py",
              "verify", sa, loc2, ok=False)
    require(bad.returncode != 0, "source-SA verifier accepted corrupt sample")
    p[-1] ^= 1
    loc2.write_bytes(p)
    require(sha(loc2) == pin, "LOC2 restore did not recover")
    header_bad = bytearray(loc2.read_bytes())
    header_bad[0] ^= 1
    loc2.write_bytes(header_bad)
    bad = run(sys.executable, vault / "loc2_experimental.py",
              "verify", sa, loc2, ok=False)
    require(bad.returncode != 0, "LOC2 invalid header was accepted")
    result = {
        "status": "SYNTHETIC_LOC1_LOC2_PARITY_PASS_NOT_ENWIK8",
        "reference_commit": REFERENCE,
        "source_bytes": len(data), "corpus_sha256": bindings["corpus_sha256"],
        "case_count": cases, "pattern_count": len(patterns),
        "loc1_bytes": fixture["locate"].stat().st_size,
        "loc2_bytes": expected,
        "same_rlb2_and_rlr2": True, "maximum_lf_steps": maximum_lf,
        "hostile": {"changed_loc2_sample": "REJECTED_BY_PIN_AND_SA_ORACLE",
                    "wrong_header": "REJECTED"},
        "bindings": bindings,
        "limitations": [
            "Tiny synthetic fixture, not an enwik8 or 1 TB benchmark.",
            "LOC2 SHA pin supplied by the experiment; not trusted remote currentness.",
            "Original explicit LOC2 loader is not manifest bound, unlike LOC1 V2.",
            "Timing samples in same process; no performance superiority inferred.",
            "Bounded locate means first FM rows, not first offsets in source order.",
        ],
    }
    (output / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    out = args.output.resolve()
    require(not out.exists(), "output must be new/disposable")
    out.mkdir(parents=True)
    compare(args.reference, out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("LOC1_LOC2_GATE_REJECTED: " + repr(exc), file=sys.stderr)
        sys.exit(2)
