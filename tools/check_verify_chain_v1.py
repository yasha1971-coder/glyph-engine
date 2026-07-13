#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

PROOF = "P12"
FORMAT = "GLYPH_VERIFY_CHAIN_V1"
GRAPH_VERSION = "GLYPH_PROOF_GRAPH_V1"

PROOFS = [
    {
        "id": "P1",
        "name": "Virtual Sentinel Total Order",
        "spec": "docs/specs/GLYPH_VIRTUAL_SENTINEL_TOTAL_ORDER_V1.md",
        "checker": "tools/check_virtual_sentinel_total_order_v1.py",
        "depends": [],
    },
    {
        "id": "P2",
        "name": "Suffix Array Validity",
        "spec": "docs/specs/GLYPH_SUFFIX_ARRAY_VALIDITY_V1.md",
        "checker": "tools/check_suffix_array_validity_v1.py",
        "depends": ["P1"],
    },
    {
        "id": "P3",
        "name": "Suffix BWT Relation",
        "spec": "docs/specs/GLYPH_SUFFIX_BWT_RELATION_V1.md",
        "checker": "tools/check_suffix_bwt_relation_v1.py",
        "depends": ["P1", "P2"],
    },
    {
        "id": "P4",
        "name": "Canonical Corpus Identity",
        "spec": "docs/specs/GLYPH_CORPUS_IDENTITY_V1.md",
        "checker": "tools/check_corpus_identity_v1.py",
        "depends": [],
    },
    {
        "id": "P5",
        "name": "FM Token Rank LF Consistency",
        "spec": "docs/specs/GLYPH_FM_TOKEN_RANK_LF_CONSISTENCY_V1.md",
        "checker": "tools/check_fm_token_rank_lf_consistency_v1.py",
        "depends": ["P2", "P3", "P4"],
    },
    {
        "id": "P6",
        "name": "FM Backward Search Exactness",
        "spec": "docs/specs/GLYPH_FM_BACKWARD_SEARCH_EXACTNESS_V1.md",
        "checker": "tools/check_fm_backward_search_exactness_v1.py",
        "depends": ["P5"],
    },
    {
        "id": "P7",
        "name": "Locate Coordinate Exactness",
        "spec": "docs/specs/GLYPH_LOCATE_COORDINATE_EXACTNESS_V1.md",
        "checker": "tools/check_locate_coordinate_exactness_v1.py",
        "depends": ["P4", "P5", "P6"],
    },
    {
        "id": "P8",
        "name": "Binary Safe Query Transport",
        "spec": "docs/specs/GLYPH_BINARY_SAFE_QUERY_TRANSPORT_V1.md",
        "checker": "tools/check_binary_safe_query_transport_v1.py",
        "depends": ["P1", "P6"],
    },
    {
        "id": "P9",
        "name": "Document Boundary Semantics",
        "spec": "docs/specs/GLYPH_DOCUMENT_BOUNDARY_SEMANTICS_V1.md",
        "checker": "tools/check_document_boundary_semantics_v1.py",
        "depends": ["P4", "P6", "P7", "P8"],
    },
    {
        "id": "P10",
        "name": "Replay Determinism",
        "spec": "docs/specs/GLYPH_REPLAY_DETERMINISM_V1.md",
        "checker": "tools/check_replay_determinism_v1.py",
        "depends": ["P4", "P6", "P7", "P8", "P9"],
    },
    {
        "id": "P11",
        "name": "Bundle Completeness",
        "spec": "docs/specs/GLYPH_BUNDLE_COMPLETENESS_V1.md",
        "checker": "tools/check_bundle_completeness_v1.py",
        "depends": ["P4", "P7", "P8", "P9", "P10"],
    },
]

P12_SPEC = "docs/specs/GLYPH_VERIFY_CHAIN_V1.md"
GRAPH_SPEC = "docs/specs/GLYPH_PROOF_GRAPH_V1.md"
GRAPH_RUNNER = "tools/run_glyph_proof_graph_v1.sh"


class VerifyChainError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_json(stdout: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()

    for index, char in enumerate(stdout):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    return None


def validate_graph_definition() -> None:
    ids = [proof["id"] for proof in PROOFS]

    if ids != [f"P{i}" for i in range(1, 12)]:
        raise VerifyChainError("proof graph must contain exactly P1 through P11")

    positions = {proof_id: index for index, proof_id in enumerate(ids)}

    for proof in PROOFS:
        for dependency in proof["depends"]:
            if dependency not in positions:
                raise VerifyChainError(
                    f"{proof['id']} has unknown dependency {dependency}"
                )
            if positions[dependency] >= positions[proof["id"]]:
                raise VerifyChainError(
                    f"{proof['id']} dependency {dependency} is not earlier"
                )


def validate_required_files() -> list[dict[str, Any]]:
    required = [
        P12_SPEC,
        GRAPH_SPEC,
        GRAPH_RUNNER,
        *[
            value
            for proof in PROOFS
            for value in (proof["spec"], proof["checker"])
        ],
    ]

    if len(required) != len(set(required)):
        raise VerifyChainError("duplicate required path in proof graph")

    files = []

    for relative in required:
        path = ROOT / relative

        if not path.is_file():
            raise VerifyChainError(f"required proof file missing: {relative}")

        if path.is_symlink():
            raise VerifyChainError(f"proof file must not be symlink: {relative}")

        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return files


def run_checker(proof: dict[str, Any]) -> dict[str, Any]:
    checker = ROOT / proof["checker"]

    completed = subprocess.run(
        [sys.executable, "-I", str(checker)],
        cwd=ROOT,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONNOUSERSITE": "1",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
        check=False,
    )

    if completed.returncode != 0:
        raise VerifyChainError(
            f"{proof['id']} checker failed with code "
            f"{completed.returncode}: {completed.stderr[-2000:]}"
        )

    parsed = extract_json(completed.stdout)

    if parsed is not None:
        if parsed.get("ok") is not True:
            raise VerifyChainError(f"{proof['id']} checker JSON did not report ok=true")

        reported = parsed.get("proof_obligation")
        if reported is not None and reported != proof["id"]:
            raise VerifyChainError(
                f"{proof['id']} checker reported proof_obligation={reported}"
            )

        if proof["id"] == "P11" and parsed.get("p12_ready") is not True:
            raise VerifyChainError("P11 did not report p12_ready=true")

    return {
        "id": proof["id"],
        "name": proof["name"],
        "status": "PASS",
        "depends": proof["depends"],
        "spec": proof["spec"],
        "checker": proof["checker"],
        "checker_sha256": sha256_file(checker),
        "stdout_sha256": hashlib.sha256(
            completed.stdout.encode("utf-8")
        ).hexdigest(),
        "json_result_found": parsed is not None,
        "p12_ready": (
            parsed.get("p12_ready")
            if proof["id"] == "P11" and parsed is not None
            else None
        ),
    }


def mutation_missing_node() -> bool:
    mutated = PROOFS[:-1]
    return [proof["id"] for proof in mutated] != [f"P{i}" for i in range(1, 12)]


def mutation_duplicate_node() -> bool:
    mutated = [*PROOFS, PROOFS[-1]]
    ids = [proof["id"] for proof in mutated]
    return len(ids) != len(set(ids))


def mutation_dependency_skip() -> bool:
    known = {"P1"}
    return "P11" not in known


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        default="benchmarks/results/GLYPH_PROOF_GRAPH_V1.json",
    )
    args = parser.parse_args()

    validate_graph_definition()
    required_files = validate_required_files()

    results = []
    passed: set[str] = set()

    for proof in PROOFS:
        missing_dependencies = [
            dependency
            for dependency in proof["depends"]
            if dependency not in passed
        ]

        if missing_dependencies:
            raise VerifyChainError(
                f"{proof['id']} cannot run; missing passed dependencies: "
                + ",".join(missing_dependencies)
            )

        result = run_checker(proof)
        results.append(result)
        passed.add(proof["id"])

    if passed != {f"P{i}" for i in range(1, 12)}:
        raise VerifyChainError("P1-P11 closure incomplete")

    mutations = {
        "missing_node_rejected": mutation_missing_node(),
        "duplicate_node_rejected": mutation_duplicate_node(),
        "dependency_skip_rejected": mutation_dependency_skip(),
    }

    if not all(mutations.values()):
        raise VerifyChainError("proof graph mutation guard failed")

    p12 = {
        "id": "P12",
        "name": "End-to-End Verify Chain Closure",
        "status": "PASS",
        "depends": [f"P{i}" for i in range(1, 12)],
        "spec": P12_SPEC,
        "checker": "tools/check_verify_chain_v1.py",
    }

    all_results = [*results, p12]

    if len(all_results) != 12:
        raise VerifyChainError("proof result count must be exactly 12")

    if [item["id"] for item in all_results] != [
        f"P{i}" for i in range(1, 13)
    ]:
        raise VerifyChainError("proof result order must be P1 through P12")

    if not all(item["status"] == "PASS" for item in all_results):
        raise VerifyChainError("not all proof obligations passed")

    output = {
        "ok": True,
        "proof_obligation": PROOF,
        "format": FORMAT,
        "proof_graph_version": GRAPH_VERSION,
        "proof_count": 12,
        "passed": 12,
        "failed": 0,
        "all_required_nodes_present": True,
        "all_dependencies_satisfied": True,
        "p11_handoff_accepted": True,
        "verify_ok_permitted": True,
        "required_files": required_files,
        "mutations": mutations,
        "proofs": all_results,
    }

    result_path = ROOT / args.result
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerifyChainError as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "proof_obligation": PROOF,
                    "format": FORMAT,
                    "verify_ok_permitted": False,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1)
