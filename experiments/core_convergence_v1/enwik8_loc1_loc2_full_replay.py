#!/usr/bin/env python3
"""Rebuild enwik8 *from pinned public source* and compare original LOC1 vs LOC2.

No frozen historical artifact is edited. Non-portable LOC2 is scored only as
research candidate; no production/cryptographic proof contract is asserted.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import math
import mmap
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REFERENCE_SHA = "c33cb493bc75f13ee4a904fabb190a29a196c94b"
URL = "https://mattmahoney.net/dc/enwik8.zip"
ENWIK8 = {"bytes": 100000000,
          "sha256": "2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8",
          "md5": "a1fa5ffddb56f4953e226637dabbb36a"}
PIN = {
    "sa": "a70951cba4f51915f9e4ef7d7e819c11c09dcaf57677cd573a9ab746e9aed727",
    "bwt": "d00b44f165ba78dc11bdc048f8578021003f17fa19587fea944e499fb6514195",
    "rlb2": "4ca1218cc84b38539b6d12b0aacdfa30b21e387b24e4f8abaa828f1a6fffc0d3",
    "rlr2": "e76a49db554860bf907aa307a503240d1e71ef0400282c34548a9726a3f22719",
    "loc1": "1ba04acce3b2ebea10ae3ab703138c1a33e7c9caa27fe7a1056d4ed8c44f7f6f",
    "manifest": "ddcaebdf9b518f63be4d4a47a801b89d155a5372ebeb6ada8c6866dcd268b827",
}
EXPECTED_BYTES = {
    "sa": 400000064,
    "bwt": 200000058,
    "rlb2": 73365105,
    "rlr2": 12844028,
    "loc1": 12500040,
    "manifest": 828,
    "loc2": 3125036,
}
SAMPLES = 128


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def canonical(data):
    return (json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n").encode()


def run(command, log, *, timeout=600):
    started = time.perf_counter()
    command = [str(x) for x in command]
    # GNU time is optional; when present, report per-process peak resident memory.
    command_with_time = (["/usr/bin/time", "-f", "GLYPH_BENCH_RSS_KB=%M"]
                         + command if Path("/usr/bin/time").is_file() else command)
    proc = subprocess.run(command_with_time, text=True, capture_output=True, timeout=timeout,
                          stdin=subprocess.DEVNULL)
    match = re.search(r"GLYPH_BENCH_RSS_KB=(\d+)", proc.stderr)
    record = {"command": command, "elapsed_seconds": round(time.perf_counter()-started, 6),
              "return_code": proc.returncode,
              "peak_rss_kb": int(match.group(1)) if match else None,
              "stdout_tail": proc.stdout[-250:], "stderr_tail": proc.stderr[-350:]}
    log.append(record)
    require(proc.returncode == 0, f"command failed: {record}")
    return proc


def read_official_enwik8(folder, log):
    started = time.perf_counter()
    archive = folder / "enwik8.zip"
    with urllib.request.urlopen(URL, timeout=180) as response, archive.open("wb") as out:
        shutil.copyfileobj(response, out, length=2**20)
    require(archive.stat().st_size == 36445475,
            f"official zip size changed: {archive.stat().st_size}")
    require(zipfile.is_zipfile(archive), "upstream ZIP invalid")
    with zipfile.ZipFile(archive) as z:
        require(z.namelist() == ["enwik8"], f"unexpected ZIP members: {z.namelist()}")
        require(z.getinfo("enwik8").file_size == ENWIK8["bytes"], "source zip geometry")
        source = folder / "enwik8"
        with z.open("enwik8") as stream, source.open("wb") as out:
            shutil.copyfileobj(stream, out, length=2**20)
    require(source.stat().st_size == ENWIK8["bytes"], "source byte length mismatch")
    require(file_sha(source) == ENWIK8["sha256"], "source SHA256 mismatched pinned enwik8")
    log.append({"step": "download_and_hash_source", "url": URL,
                "source_bytes": source.stat().st_size, "zip_bytes": archive.stat().st_size,
                "seconds": round(time.perf_counter()-started, 6)})
    archive.unlink()  # Retain raw canonical source only inside disposable workspace.
    return source


def emit_loc1(sa, output, rows, step):
    count = (rows + step - 1) // step
    with sa.open("rb") as f, output.open("wb") as w:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            require(len(m) == 60 + 4 * rows, "SA file geometry")
            w.write(struct.pack("<4sQIQ", b"LOC1", rows, step, count))
            for row in range(0, rows, step):
                suffix = struct.unpack_from("<I", m, 60 + 4 * row)[0]
                w.write(struct.pack("<QQ", row, suffix))
        finally:
            m.close()


def timed_quantile(values, p):
    return sorted(values)[math.ceil(len(values) * p) - 1] / 1e6


def raw_oracle(source, pattern):
    offsets = []
    start = 0
    while True:
        found = source.find(pattern, start)
        if found < 0:
            break
        offsets.append(found)
        start = found + 1
    return offsets


def strict_compare(reference, files, log, output):
    tools = reference / "tools"
    vault = reference / "experiments/personal_vault_v0"
    sys.path[:0] = [str(tools), str(vault)]
    import rlbwt_query_v2 as query
    import query_loc2_experimental as loc2
    verified_binding = query.load_runtime_manifest(files["manifest"])
    require(verified_binding["manifest_sha256"] == PIN["manifest"],
            "historical LOC1 manifest not honored")
    source = files["source"].read_bytes()
    require(len(source) == ENWIK8["bytes"], "source out of sync")
    patterns = [
        source[i:i + 24] for i in (1200, 1000000, 13579113, 27890000,
                                  55432100, 83567543)
    ] + [b"\xff\xff\xffGLYPH-no-match", b"\x00\xff", b"\xff"]
    require(all(patterns), "empty pattern")
    before = time.perf_counter()
    a = query.QueryRuntime(files["rlb2"], files["rlr2"], files["loc1"])
    loc1_open_seconds = time.perf_counter()-before
    before = time.perf_counter()
    b = loc2.Runtime2(files["rlb2"], files["rlr2"], files["loc2"])
    loc2_open_seconds = time.perf_counter()-before
    times = {"loc1": [], "loc2": []}
    count = 0
    cases = 0
    max_lf = 0
    try:
        for p in patterns:
            oracle = raw_oracle(source, p)
            # Distinct count/locate semantics, including FM-row bounded positions.
            for limit in (-1, 0, 1, 2):
                for _ in range(5):
                    r1 = a.query(p, limit)
                    r2 = b.query(p, limit)
                    require(r1["count"] == r2["count"] == len(oracle),
                            f"count mismatch pattern={p.hex()} limit={limit}")
                    require(r1["fm_interval"] == r2["fm_interval"],
                            "FM interval mismatch")
                    require(r1["locate_offsets"] == r2["locate_offsets"],
                            f"LOC1/LOC2 offsets mismatch {p.hex()} limit={limit}")
                    require(r1["maximum_lf_steps"] == r2["maximum_lf_steps"],
                            "locate LF parity mismatch")
                    if limit == -1:
                        require(r1["locate_offsets"] == oracle,
                                "complete offsets mismatched raw source")
                        times["loc1"].append(r1["query_elapsed_ns"])
                        times["loc2"].append(r2["query_elapsed_ns"])
                    else:
                        vals = r1["locate_offsets"]
                        require(len(vals) == min(len(oracle), limit)
                                and len(set(vals)) == len(vals)
                                and set(vals).issubset(set(oracle)),
                                "bounded FM-row locate not a valid exact subset")
                    max_lf = max(max_lf, r1["maximum_lf_steps"])
                    count += r1["count"]
                    cases += 1
    finally:
        a.close()
        b.close()
    log.append({"step": "runtime_query_comparison", "patterns": len(patterns),
                "cases": cases, "maximum_observed_LF_steps": max_lf,
                "loc1_runtime_open_s": loc1_open_seconds,
                "loc2_runtime_open_s": loc2_open_seconds})
    return {
        "patterns": len(patterns), "cases": cases, "max_lf": max_lf,
        "loc1_open_seconds": loc1_open_seconds,
        "loc2_open_seconds": loc2_open_seconds,
        "loc1_warm_query_p50_ms": timed_quantile(times["loc1"], 0.5),
        "loc1_warm_query_p95_ms": timed_quantile(times["loc1"], 0.95),
        "loc2_warm_query_p50_ms": timed_quantile(times["loc2"], 0.5),
        "loc2_warm_query_p95_ms": timed_quantile(times["loc2"], 0.95),
        "timing_samples_per_candidate": len(times["loc1"]),
        "experimental_timing_caveat": "Same-process warm query timings, fixed order, no statistical superiority claim.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ref = args.reference.resolve()
    out = args.output.resolve()
    require(not out.exists(), "output directory must be newly created")
    require(subprocess.check_output(
        ["git", "-C", str(ref), "rev-parse", "HEAD"],
        text=True).strip() == REFERENCE_SHA, "wrong historical checkout")
    out.mkdir(parents=True)
    work = out / "work"
    work.mkdir()
    log = []
    evidence = {"schema": "GLYPH_ENWIK8_LOC1_VS_LOC2_RESEARCH_V1",
                "status": "NOT_EXECUTED", "reference_commit": REFERENCE_SHA,
                "source": ENWIK8, "source_url": URL}
    try:
        source = read_official_enwik8(work, log)
        sa = work / "enwik8.sa.binary_v1"
        bwt = work / "enwik8.bwt.binary_v1"
        rlb2 = work / "bwt.experimental.rlb2"
        rlr2 = work / "bwt.experimental.rlr2"
        loc1 = work / "locate_core_s128.bin"
        loc2 = work / "locate_core_s128.loc2"
        manifest = work / "runtime_manifest_v2.json"
        files = {"source": source, "sa": sa, "bwt": bwt, "rlb2": rlb2,
                 "rlr2": rlr2, "loc1": loc1, "loc2": loc2, "manifest": manifest}
        run([ref / "build/build_sa_binary_v1", source, sa], log)
        run([ref / "build/build_bwt_binary_v1", source, sa, bwt], log)
        for key in ("sa", "bwt"):
            require(files[key].stat().st_size == EXPECTED_BYTES[key]
                    and file_sha(files[key]) == PIN[key],
                    f"{key}: canonical size/hash mismatch")
        run([sys.executable, ref / "tools/rlbwt_container_v2.py", "encode",
             bwt, rlb2], log)
        run([sys.executable, ref / "tools/rlbwt_rank_v2.py", "build",
             rlb2, rlr2, "--rank-step", "8192"], log)
        for key in ("rlb2", "rlr2"):
            require(files[key].stat().st_size == EXPECTED_BYTES[key]
                    and file_sha(files[key]) == PIN[key],
                    f"{key}: canonical size/hash mismatch")
        emit_loc1(sa, loc1, ENWIK8["bytes"]+1, SAMPLES)
        require(loc1.stat().st_size == EXPECTED_BYTES["loc1"]
                and file_sha(loc1) == PIN["loc1"], "LOC1 canonical hash mismatch")
        # Existing experimental builder: not an approved V2 manifest type.
        run([sys.executable, ref / "experiments/personal_vault_v0/loc2_experimental.py",
             "build", sa, str(ENWIK8["bytes"]+1), str(SAMPLES), loc2], log)
        run([sys.executable, ref / "experiments/personal_vault_v0/loc2_experimental.py",
             "verify", sa, loc2], log)
        require(loc2.stat().st_size == EXPECTED_BYTES["loc2"], "LOC2 size wrong")
        frozen_manifest = ref / "benchmarks/results/RLBWT_BINARY_SAFE_RUNTIME_V2_ENWIK8_MANIFEST_V1.json"
        shutil.copyfile(frozen_manifest, manifest)
        require(manifest.stat().st_size == EXPECTED_BYTES["manifest"]
                and file_sha(manifest) == PIN["manifest"],
                "historical V2 binding manifest hash mismatch")
        # Research-only binding to all compact artifacts and original source.
        research = {
            "format": "GLYPH_ENWIK8_LOC2_RESEARCH_MANIFEST_V0",
            "warning": "NOT A V2 CANONICAL RUNTIME; NOT A PORTABLE PROOF BUNDLE",
            "source_sha256": ENWIK8["sha256"], "source_bytes": ENWIK8["bytes"],
            "historical_loc1_manifest_sha256": PIN["manifest"],
            "rank_step": 8192, "sample_step": SAMPLES,
            "files": {k: {"bytes": files[k].stat().st_size, "sha256": file_sha(files[k])}
                      for k in ("rlb2", "rlr2", "loc2")},
        }
        research_path = work / "loc2_research_manifest_v0.json"
        research_path.write_bytes(canonical(research))
        verification = strict_compare(ref, files, log, out)
        # Freeze an out-of-band checksum within this experiment; rehash then fail
        # before using a corrupted index. This pin is not a remote trust witness.
        pin = file_sha(loc2)
        with loc2.open("r+b") as f:
            f.seek(-1, os.SEEK_END)
            byte = f.read(1)
            f.seek(-1, os.SEEK_END)
            f.write(bytes([byte[0] ^ 1]))
        require(file_sha(loc2) != pin, "mutated LOC2 not rejected by SHA")
        with loc2.open("r+b") as f:
            f.seek(-1, os.SEEK_END)
            f.write(byte)
        require(file_sha(loc2) == pin, "tampered LOC2 could not be restored")
        storage1 = sum(files[k].stat().st_size for k in ("rlb2", "rlr2", "loc1", "manifest"))
        storage2 = sum(files[k].stat().st_size for k in ("rlb2", "rlr2", "loc2")) + research_path.stat().st_size
        evidence.update({
            "status": "ENWIK8_LOC1_LOC2_REBUILT_PARITY_PASS_RESEARCH_SCOPE",
            "canonical_historical_hashes": {k: PIN[k] for k in PIN},
            "loc2_sha256": pin,
            "verification": verification,
            "loc1_storage": {"bytes": storage1, "ratio": storage1/ENWIK8["bytes"]},
            "loc2_research_storage": {
                "bytes": storage2, "ratio": storage2/ENWIK8["bytes"],
                "research_manifest_bytes": research_path.stat().st_size,
                "loc2_index_bytes": loc2.stat().st_size},
            "saving_index_bytes_only": loc1.stat().st_size-loc2.stat().st_size,
            "source_and_sa_bwt_are_build_only": True,
            "stale_index_sha_mutation_rejected": True,
            "limitations": [
                "LOC2 variant runs via explicit research path, not the LOC1 canonical manifest contract.",
                "Research source & manifest hashes are not externally signed or rollback protected.",
                "Verifying all LOC2 samples against SA uses construction-only 400MB SA.",
                "No authenticated portable source-payload recovery is added by LOC2.",
                "One continuous enwik8 corpus; not document-aware proof or Personal Vault.",
                "Python query timings are same-process warm measurements, not native production SLA.",
                "Comparative ratio excludes verification program code, as did historical Runtime V2.",
            ],
        })
    except Exception as e:
        evidence.update({"status": "BLOCKED", "error": str(e)})
    finally:
        (out / "result.json").write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
        (out / "commands.json").write_text(json.dumps(log, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": evidence["status"],
                      "loc1": evidence.get("loc1_storage"),
                      "loc2": evidence.get("loc2_research_storage"),
                      "verified_case_count": evidence.get("verification", {}).get("cases"),
                      "error": evidence.get("error")}, sort_keys=True))
    return 0 if evidence["status"].startswith("ENWIK8_LOC1_LOC2_REBUILT_PARITY_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
