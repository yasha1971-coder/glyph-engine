#!/usr/bin/env python3
"""CORE-G1B: real 30-object public corpus through EXISTING proof-carrying V0.

Runs only in a newly created disposable location. Does not modify existing
V0 implementation or any real user data. Does not claim full O1-O6 portability.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "experiments" / "personal_vault_v0"
PY = sys.executable
SOURCE_SHA = "4f3fa33cc6276dd8ec1f8099056646b5a63f1e18d2230753618db8813eff6a67"
SOURCE_BYTES = 7230581
PATTERN = "Wonderland"


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def invoke(args, cwd=ROOT, allow_failure=False):
    t = time.perf_counter()
    result = subprocess.run([str(x) for x in args], cwd=cwd, capture_output=True,
                            text=True, timeout=900)
    output = {"args": [str(x) for x in args], "exit": result.returncode,
              "seconds": round(time.perf_counter()-t, 6),
              "stdout_tail": result.stdout[-900:], "stderr_tail": result.stderr[-900:]}
    if result.returncode and not allow_failure:
        raise RuntimeError(f"command failed: {output}")
    return output


def report_size(vault, receipt):
    seg = vault / "segments" / "00000001"
    roots = sorted((vault / "manifests" / "roots").glob("*.json"))
    if len(roots) != 1:
        raise AssertionError(f"expected one root, saw {len(roots)}")
    roles = {
        "rlb3x_shared_compressed_self_index": seg / "bwt.rlb3x",
        "locate_index": seg / "locate.loc2",
        "object_map": seg / "objects.json",
        "source_metadata": seg / "source-state.json",
        "segment_manifest": seg / "segment-manifest.json",
        "root_manifest": roots[0],
        "repository_metadata": vault / "repo.meta",
        "receipt_per_one_query": receipt,
    }
    counted_paths = {p.resolve() for p in roles.values()}
    # Auxiliary last-verify report is retained in this run; account it as well.
    verify_sidecar = vault / "manifests" / "last-verify.json"
    if verify_sidecar.exists():
        roles["verify_diagnostic_sidecar"] = verify_sidecar
        counted_paths.add(verify_sidecar.resolve())
    # Fail closed when a non-build artifact is forgotten in accounting.
    forgotten = []
    build_ws = 0
    for p in vault.rglob("*"):
        if not p.is_file():
            continue
        if "cache" in p.relative_to(vault).parts or "journal" in p.relative_to(vault).parts:
            build_ws += p.stat().st_size
            continue
        if p.resolve() not in counted_paths:
            forgotten.append(str(p.relative_to(vault)))
    if forgotten:
        raise AssertionError("unaccounted persistent Vault files: " + repr(forgotten))
    sizes = {name: p.stat().st_size for name, p in roles.items()}
    return {
        "category_bytes": sizes,
        "counted_persistent_bytes_in_v0_vault_plus_one_receipt": sum(sizes.values()),
        "construction_cache_and_journal_bytes_excluded": build_ws,
        "all_nonbuild_vault_files_accounted": not forgotten,
        "required_runtime_and_verifier_code_not_packaged": True,
        "not_a_portable_O1_O6_bundle": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="new output folder")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit("refuse pre-existing --output")
    out.mkdir(parents=True)
    workspace = out / "work"
    workspace.mkdir()
    log = []
    evidence = {
        "schema": "GLYPH_CORE_G1B_V0_PUBLIC_PROOF_PROBE",
        "target": "legacy proof-carrying RLB3X, not full P/R/O integrated archive",
        "corpus_commit": "5583ca94d1643b6dcd6b6dd2ad0c5704a4afa094",
        "status": "NOT_EXECUTED",
    }
    try:
        original = workspace / "source"
        vault = workspace / "vault"
        receipt = workspace / "receipt.json"
        log.append(invoke(["bash", V0 / "fetch_corpora.sh", original]))
        # Source set has a SHA256SUMS metadata file excluded from pack().
        files = sorted(p for p in original.rglob("*")
                       if p.is_file() and p.name != "SHA256SUMS")
        raw = b"".join(p.read_bytes() for p in files)
        if len(files) != 30 or len(raw) != SOURCE_BYTES or hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
            raise AssertionError("pinned source identity mismatch")
        evidence["source"] = {
            "documents": len(files), "logical_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        log.append(invoke([PY, V0 / "glyph_vault_cli_v0.py", "init", vault]))
        log.append(invoke([PY, V0 / "glyph_vault_cli_v0.py", "add", vault, original]))
        log.append(invoke([PY, V0 / "glyph_vault_cli_v0.py", "verify", vault]))
        log.append(invoke([PY, V0 / "proof_carrying_recall_v0.py", vault,
                           "--pattern", PATTERN, "--out", receipt]))
        made = json.loads(log[-1]["stdout_tail"])
        if not made.get("ok") or made.get("occurrences", 0) < 1:
            raise AssertionError("positive receipt not produced")
        held_out = workspace / "source-held-out"
        original.rename(held_out)  # preserves inputs; former source path becomes unavailable.
        if original.exists():
            raise AssertionError("original source path still accessible")
        log.append(invoke([PY, V0 / "verify_proof_carrying_recall_v0.py", vault, receipt]))
        verified = json.loads(log[-1]["stdout_tail"])
        if not (verified.get("ok") and verified.get("original_source_required") is False
                and verified.get("ai_required") is False):
            raise AssertionError("receipt replay did not complete")
        target = held_out / made["path"]
        if not target.is_file():
            raise AssertionError("expected source object missing in held-out copy")
        expected = target.read_bytes().count(PATTERN.encode())
        if expected != verified["occurrences"]:
            raise AssertionError("object count mismatch against source oracle")
        evidence["receipt"] = {
            "claimed_object": made["path"],
            "object_occurrences": verified["occurrences"],
            "independent_object_oracle_count": expected,
            "restored_without_original_source_path": True,
            "receipt_sha256": digest(receipt),
            "root_sha256": verified["root_sha256"],
            "source_tree_preserved_elsewhere": held_out.exists(),
        }
        # Scope-specific mutations: independent receipt, index and proof tamper.
        bad = json.loads(receipt.read_text())
        bad["evidence"]["object_offsets"][0] += 1
        badfile = workspace / "bad-receipt.json"
        badfile.write_text(json.dumps(bad, sort_keys=True, separators=(",", ":")) + "\n")
        p = invoke([PY, V0 / "verify_proof_carrying_recall_v0.py", vault, badfile],
                   allow_failure=True)
        log.append(p)
        if p["exit"] == 0 or "VERIFY_FAIL" not in p["stderr_tail"]+p["stdout_tail"]:
            raise AssertionError("hostile receipt mutation was not rejected")
        evidence["hostile_receipt_rejected"] = True
        evidence["storage"] = report_size(vault, receipt)
        evidence["status"] = "SCOPED_V0_PROOF_PASS_FULL_PROOF_INTEGRATION_NOT_CLOSED"
        evidence["limitations"] = [
            "Only receipt-bound selected-object literal occurrences were certified.",
            "Global absence, global candidate coverage and root authority/currentness not proved.",
            "Verifier/decoder Python source and host runtime are not packaged, thus not O1-O6 portable.",
            "Full retained size includes one receipt but excludes required unpackaged code, so no full-store ratio claim.",
            "Construction cache and held-out source are not retained archive state.",
            "No V2 generation semantics, no direct frozen Laptop V1 Phase A/B equivalence.",
            "One-corpus, one-pattern measurement; no performance or scalability generalization.",
        ]
    except Exception as e:
        evidence["status"] = "BLOCKED"
        evidence["error"] = str(e)
    finally:
        (out / "result.json").write_text(json.dumps(evidence, sort_keys=True,
                                     indent=2) + "\n")
        (out / "command_log.json").write_text(json.dumps(log, sort_keys=True,
                                         indent=2) + "\n")
    print(json.dumps({"status": evidence["status"], "output": str(out),
                      "error": evidence.get("error"),
                      "persistent_bytes": evidence.get("storage", {}).get(
                          "counted_persistent_bytes_in_v0_vault_plus_one_receipt")
                      }, sort_keys=True))
    return 0 if evidence["status"].startswith("SCOPED_V0_PROOF_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
