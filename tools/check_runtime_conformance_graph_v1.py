#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

FORMAT = "GLYPH_RUNTIME_CONFORMANCE_GRAPH_V1"
RUNTIME_PROFILE = "GLYPH_BINARY_RUNTIME_V1"

DEFAULT_RESULT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_RUNTIME_CONFORMANCE_GRAPH_V1.json"
)

PROOF_GRAPH_RESULT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_PROOF_GRAPH_V1.json"
)

BASELINE_RESULT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_RUNTIME_CONFORMANCE_BASELINE_V1.json"
)

NODE_DEFINITIONS = [
    {
        "id": "R1",
        "name": "Binary-safe C++ Count",
        "checker":
            "tools/check_binary_runtime_count_v1.py",
        "result":
            "benchmarks/results/"
            "GLYPH_BINARY_RUNTIME_COUNT_V1.json",
        "spec":
            "docs/specs/GLYPH_RUNTIME_CONFORMANCE_V1.md",
        "depends": ["R0", "P1-P12"],
    },
    {
        "id": "R2",
        "name": "Binary-safe C++ Locate",
        "checker":
            "tools/check_binary_runtime_locate_v1.py",
        "result":
            "benchmarks/results/"
            "GLYPH_BINARY_RUNTIME_LOCATE_V1.json",
        "spec":
            "docs/specs/GLYPH_RUNTIME_CONFORMANCE_V1.md",
        "depends": ["R1"],
    },
    {
        "id": "R3",
        "name": "Multi-document Runtime",
        "checker":
            "tools/check_binary_runtime_multidoc_v1.py",
        "result":
            "benchmarks/results/"
            "GLYPH_BINARY_RUNTIME_MULTIDOC_V1.json",
        "spec":
            "docs/specs/"
            "GLYPH_BINARY_RUNTIME_MULTIDOC_V1.md",
        "depends": ["R2"],
    },
    {
        "id": "R4",
        "name": "Deterministic Runtime Evidence",
        "checker":
            "tools/check_binary_runtime_evidence_v1.py",
        "result":
            "benchmarks/results/"
            "GLYPH_BINARY_RUNTIME_EVIDENCE_V1.json",
        "spec":
            "docs/specs/"
            "GLYPH_BINARY_RUNTIME_EVIDENCE_V1.md",
        "depends": ["R3"],
    },
    {
        "id": "R5",
        "name": "Self-contained Runtime Bundle",
        "checker":
            "tools/check_binary_runtime_bundle_v1.py",
        "result":
            "benchmarks/results/"
            "GLYPH_BINARY_RUNTIME_BUNDLE_V1.json",
        "spec":
            "docs/specs/"
            "GLYPH_BINARY_RUNTIME_BUNDLE_V1.md",
        "depends": ["R4"],
    },
]

REQUIRED_NODE_IDS = [
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
]

EXTERNAL_DEPENDENCIES = {
    "P1-P12",
}


class ConformanceError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConformanceError(
            f"required JSON missing: "
            f"{path.relative_to(ROOT)}"
        )

    try:
        value = json.loads(path.read_text())
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise ConformanceError(
            f"invalid JSON: {path.relative_to(ROOT)}"
        ) from error

    if not isinstance(value, dict):
        raise ConformanceError(
            f"JSON root must be object: "
            f"{path.relative_to(ROOT)}"
        )

    return value


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise ConformanceError(message)


def validate_proof_graph() -> dict[str, Any]:
    result = load_json(PROOF_GRAPH_RESULT)

    require(
        result.get("ok") is True,
        "reference proof graph is not ok",
    )
    require(
        result.get("proof_count") == 12,
        "reference proof count is not 12",
    )
    require(
        result.get("passed") == 12,
        "reference proof graph did not pass 12 nodes",
    )
    require(
        result.get("failed") == 0,
        "reference proof graph contains failure",
    )
    require(
        result.get("verify_ok_permitted") is True,
        "reference proof graph does not permit verify",
    )
    require(
        result.get("proof_graph_version")
        == "GLYPH_PROOF_GRAPH_V1",
        "reference proof graph version mismatch",
    )

    proofs = result.get("proofs")

    require(
        isinstance(proofs, list)
        and len(proofs) == 12,
        "reference proof list mismatch",
    )
    require(
        all(
            isinstance(item, dict)
            and item.get("status") == "PASS"
            for item in proofs
        ),
        "reference proof graph contains non-PASS node",
    )

    return {
        "status": "PASS",
        "proof_graph_version":
            result["proof_graph_version"],
        "proof_count": result["proof_count"],
        "passed": result["passed"],
        "result":
            str(PROOF_GRAPH_RESULT.relative_to(ROOT)),
        "result_sha256":
            sha256_file(PROOF_GRAPH_RESULT),
    }


def validate_baseline() -> dict[str, Any]:
    result = load_json(BASELINE_RESULT)

    require(
        result.get("ok") is True,
        "runtime baseline is not ok",
    )
    require(
        result.get("audit_ok") is True,
        "runtime baseline audit did not pass",
    )
    require(
        result.get("runtime_conformant") is False,
        "legacy baseline must remain non-conformant",
    )
    require(
        result.get("binary_safe_arbitrary_bytes")
        is False,
        "legacy baseline unexpectedly binary-safe",
    )
    require(
        result.get("open_gap_count") == 5,
        "legacy baseline gap count mismatch",
    )

    gaps = result.get("gaps")

    require(
        isinstance(gaps, list)
        and {
            item.get("id")
            for item in gaps
            if isinstance(item, dict)
        }
        == {
            "RUNTIME-GAP-01",
            "RUNTIME-GAP-02",
            "RUNTIME-GAP-03",
            "RUNTIME-GAP-04",
            "RUNTIME-GAP-05",
        },
        "legacy baseline gap identities mismatch",
    )

    checker = (
        ROOT
        / "tools/"
        "check_runtime_conformance_baseline_v1.py"
    )
    spec = (
        ROOT
        / "docs/specs/"
        "GLYPH_RUNTIME_CONFORMANCE_V1.md"
    )

    require(
        checker.is_file(),
        "baseline checker missing",
    )
    require(
        spec.is_file(),
        "runtime conformance specification missing",
    )

    return {
        "id": "R0",
        "name": "Legacy Runtime Baseline Audit",
        "status": "PASS",
        "audit_ok": True,
        "runtime_conformant": False,
        "open_gap_count": 5,
        "depends": [],
        "checker":
            str(checker.relative_to(ROOT)),
        "checker_sha256": sha256_file(checker),
        "spec": str(spec.relative_to(ROOT)),
        "spec_sha256": sha256_file(spec),
        "result":
            str(BASELINE_RESULT.relative_to(ROOT)),
        "result_sha256":
            sha256_file(BASELINE_RESULT),
    }


def run_checker(
    checker: Path,
    result_path: Path,
) -> dict[str, Any]:
    if not checker.is_file():
        raise ConformanceError(
            f"checker missing: "
            f"{checker.relative_to(ROOT)}"
        )

    result_path.unlink(missing_ok=True)

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(checker),
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1800,
        check=False,
    )

    if completed.returncode != 0:
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or "no checker output"
        )

        raise ConformanceError(
            f"{checker.name} failed with code "
            f"{completed.returncode}: "
            f"{detail[-6000:]}"
        )

    if not result_path.is_file():
        raise ConformanceError(
            f"checker did not create result: "
            f"{result_path.relative_to(ROOT)}"
        )

    return load_json(result_path)


def validate_gate_result(
    node_id: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    require(
        result.get("ok") is True,
        f"{node_id} result is not ok",
    )
    require(
        result.get("runtime_profile")
        == RUNTIME_PROFILE,
        f"{node_id} runtime profile mismatch",
    )
    require(
        result.get("runtime_conformant") is False,
        f"{node_id} stage must not claim final closure",
    )

    if node_id == "R1":
        require(
            result.get("format")
            == "GLYPH_BINARY_RUNTIME_COUNT_V1",
            "R1 format mismatch",
        )
        require(
            result.get("count_path_conformant")
            is True,
            "R1 count path failed",
        )
        require(
            result.get("logical_sentinel") == 256,
            "R1 logical sentinel mismatch",
        )
        require(
            set(result.get("closed_baseline_gaps", []))
            == {
                "RUNTIME-GAP-01",
                "RUNTIME-GAP-02",
                "RUNTIME-GAP-03",
                "RUNTIME-GAP-04",
                "RUNTIME-GAP-05",
            },
            "R1 did not close all baseline gaps",
        )

        return {
            "count_path_conformant": True,
            "fixture_count":
                result.get("fixture_count"),
            "query_count":
                result.get("query_count"),
            "logical_sentinel": 256,
            "closed_baseline_gap_count": 5,
        }

    if node_id == "R2":
        require(
            result.get("format")
            == "GLYPH_BINARY_RUNTIME_LOCATE_V1",
            "R2 format mismatch",
        )
        require(
            result.get("count_path_conformant")
            is True,
            "R2 lost count conformance",
        )
        require(
            result.get("locate_path_conformant")
            is True,
            "R2 locate path failed",
        )
        require(
            result.get("all_offsets_byte_checked")
            is True,
            "R2 byte checking failed",
        )
        require(
            result.get("terminal_suffix_never_returned")
            is True,
            "R2 terminal suffix leaked",
        )
        require(
            result.get("bounded_locate_verified")
            is True,
            "R2 bounded locate failed",
        )

        return {
            "locate_path_conformant": True,
            "fixture_count":
                result.get("fixture_count"),
            "query_count":
                result.get("query_count"),
            "mutation_count":
                result.get("mutation_count"),
        }

    if node_id == "R3":
        require(
            result.get("format")
            == "GLYPH_BINARY_RUNTIME_MULTIDOC_V1",
            "R3 format mismatch",
        )
        require(
            result.get("multidoc_path_conformant")
            is True,
            "R3 multi-document path failed",
        )
        require(
            result.get(
                "cross_document_matches_"
                "structurally_impossible"
            )
            is True,
            "R3 cross-document exclusion failed",
        )
        require(
            result.get("empty_document_ids_preserved")
            is True,
            "R3 empty document IDs not preserved",
        )
        require(
            result.get("duplicate_documents_preserved")
            is True,
            "R3 duplicate documents not preserved",
        )

        return {
            "multidoc_path_conformant": True,
            "fixture_count":
                result.get("fixture_count"),
            "query_count":
                result.get("query_count"),
            "cross_only_fixture_count":
                result.get("cross_only_fixture_count"),
        }

    if node_id == "R4":
        require(
            result.get("format")
            == "GLYPH_BINARY_RUNTIME_EVIDENCE_GATE_V1",
            "R4 format mismatch",
        )
        require(
            result.get("runtime_evidence_conformant")
            is True,
            "R4 runtime evidence failed",
        )
        require(
            result.get("ordered_corpus_identity_bound")
            is True,
            "R4 corpus identity not bound",
        )
        require(
            result.get("query_identity_bound")
            is True,
            "R4 query identity not bound",
        )
        require(
            result.get("runtime_index_hashes_bound")
            is True,
            "R4 runtime indexes not bound",
        )
        require(
            result.get("deterministic_artifact_verified")
            is True,
            "R4 artifact is not deterministic",
        )
        require(
            result.get(
                "independent_runtime_replay_verified"
            )
            is True,
            "R4 independent runtime replay failed",
        )

        return {
            "runtime_evidence_conformant": True,
            "fixture_count":
                result.get("fixture_count"),
            "mutation_count":
                result.get("mutation_count"),
        }

    if node_id == "R5":
        require(
            result.get("format")
            == "GLYPH_BINARY_RUNTIME_BUNDLE_GATE_V1",
            "R5 format mismatch",
        )
        require(
            result.get("runtime_bundle_conformant")
            is True,
            "R5 runtime bundle failed",
        )
        require(
            result.get("self_contained") is True,
            "R5 bundle is not self-contained",
        )
        require(
            result.get("copied_bundle_replay_verified")
            is True,
            "R5 copied bundle replay failed",
        )
        require(
            result.get("replay_outside_repository_verified")
            is True,
            "R5 replay outside repository failed",
        )
        require(
            result.get("repository_dependency_required")
            is False,
            "R5 still depends on repository",
        )
        require(
            result.get("network_dependency_required")
            is False,
            "R5 unexpectedly requires network",
        )
        require(
            result.get("external_data_dependencies")
            == [],
            "R5 contains external data dependency",
        )

        return {
            "runtime_bundle_conformant": True,
            "document_count":
                result.get("document_count"),
            "file_count":
                result.get("file_count"),
            "mutation_count":
                result.get("mutation_count"),
            "self_contained": True,
        }

    raise ConformanceError(
        f"unknown runtime node: {node_id}"
    )


def execute_runtime_nodes() -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = [
        validate_baseline()
    ]

    for definition in NODE_DEFINITIONS:
        checker = ROOT / definition["checker"]
        result_path = ROOT / definition["result"]
        spec = ROOT / definition["spec"]

        require(
            spec.is_file(),
            f"spec missing: {definition['spec']}",
        )

        result = run_checker(
            checker,
            result_path,
        )

        summary = validate_gate_result(
            definition["id"],
            result,
        )

        nodes.append({
            "id": definition["id"],
            "name": definition["name"],
            "status": "PASS",
            "depends":
                list(definition["depends"]),
            "checker":
                definition["checker"],
            "checker_sha256":
                sha256_file(checker),
            "spec": definition["spec"],
            "spec_sha256":
                sha256_file(spec),
            "result":
                definition["result"],
            "result_sha256":
                sha256_file(result_path),
            "summary": summary,
        })

    graph_spec = (
        ROOT
        / "docs/specs/"
        "GLYPH_RUNTIME_CONFORMANCE_GRAPH_V1.md"
    )

    graph_checker = Path(__file__).resolve()

    require(
        graph_spec.is_file(),
        "runtime graph specification missing",
    )

    nodes.append({
        "id": "R6",
        "name": "Runtime Conformance Closure",
        "status": "PASS",
        "depends": [
            "R0",
            "R1",
            "R2",
            "R3",
            "R4",
            "R5",
            "P1-P12",
        ],
        "checker":
            str(graph_checker.relative_to(ROOT)),
        "checker_sha256":
            sha256_file(graph_checker),
        "spec":
            str(graph_spec.relative_to(ROOT)),
        "spec_sha256":
            sha256_file(graph_spec),
        "summary": {
            "runtime_conformant": True,
            "verify_ok_permitted": True,
        },
    })

    return nodes


def validate_node_sequence(
    nodes: list[dict[str, Any]],
) -> None:
    ids = [
        node.get("id")
        for node in nodes
    ]

    if ids != REQUIRED_NODE_IDS:
        raise ConformanceError(
            f"runtime node sequence mismatch: {ids}"
        )

    if len(set(ids)) != len(ids):
        raise ConformanceError(
            "duplicate runtime node"
        )

    seen: set[str] = set()

    for node in nodes:
        node_id = node.get("id")

        if node.get("status") != "PASS":
            raise ConformanceError(
                f"runtime node did not pass: {node_id}"
            )

        depends = node.get("depends")

        if not isinstance(depends, list):
            raise ConformanceError(
                f"invalid dependency list: {node_id}"
            )

        for dependency in depends:
            if dependency in EXTERNAL_DEPENDENCIES:
                continue

            if dependency not in seen:
                raise ConformanceError(
                    f"dependency not satisfied: "
                    f"{node_id} -> {dependency}"
                )

        seen.add(node_id)


def expect_rejected(
    name: str,
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        validate_node_sequence(nodes)
    except ConformanceError as error:
        return {
            "mutation": name,
            "rejected": True,
            "message": str(error),
        }

    raise ConformanceError(
        f"runtime graph mutation accepted: {name}"
    )


def mutation_gate(
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    missing = copy.deepcopy(nodes)
    missing = [
        node
        for node in missing
        if node["id"] != "R5"
    ]

    duplicate = copy.deepcopy(nodes)
    duplicate.insert(
        5,
        copy.deepcopy(
            next(
                node
                for node in duplicate
                if node["id"] == "R4"
            )
        ),
    )

    reordered = copy.deepcopy(nodes)
    index_r1 = next(
        index
        for index, node in enumerate(reordered)
        if node["id"] == "R1"
    )
    index_r2 = next(
        index
        for index, node in enumerate(reordered)
        if node["id"] == "R2"
    )
    reordered[index_r1], reordered[index_r2] = (
        reordered[index_r2],
        reordered[index_r1],
    )

    failed = copy.deepcopy(nodes)
    next(
        node
        for node in failed
        if node["id"] == "R3"
    )["status"] = "FAIL"

    return [
        expect_rejected(
            "missing_required_node",
            missing,
        ),
        expect_rejected(
            "duplicate_runtime_node",
            duplicate,
        ),
        expect_rejected(
            "dependency_order_violation",
            reordered,
        ),
        expect_rejected(
            "failed_runtime_node",
            failed,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        default=str(DEFAULT_RESULT),
    )
    args = parser.parse_args()

    try:
        proof_graph = validate_proof_graph()
        nodes = execute_runtime_nodes()

        validate_node_sequence(nodes)

        mutations = mutation_gate(nodes)

        require(
            all(
                item["rejected"] is True
                for item in mutations
            ),
            "runtime graph mutation gate failed",
        )

        output = {
            "ok": True,
            "format": FORMAT,
            "runtime_profile": RUNTIME_PROFILE,
            "proof_graph_dependency":
                proof_graph,
            "node_count": len(nodes),
            "passed": len(nodes),
            "failed": 0,
            "all_required_nodes_present": True,
            "all_dependencies_satisfied": True,
            "source_byte_domain": "0x00..0xFF",
            "logical_sentinel": 256,
            "count_path_conformant": True,
            "locate_path_conformant": True,
            "multidoc_path_conformant": True,
            "runtime_evidence_conformant": True,
            "runtime_bundle_conformant": True,
            "runtime_conformant": True,
            "verify_ok_permitted": True,
            "nodes": nodes,
            "mutations": mutations,
        }

        result_path = Path(args.result).resolve()
        result_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        result_path.write_bytes(
            canonical_json_bytes(output)
        )

        print(
            json.dumps(
                output,
                indent=2,
                sort_keys=True,
            )
        )

        return 0

    except Exception as error:
        failure = {
            "ok": False,
            "format": FORMAT,
            "runtime_profile": RUNTIME_PROFILE,
            "runtime_conformant": False,
            "verify_ok_permitted": False,
            "error": str(error),
        }

        print(
            json.dumps(
                failure,
                indent=2,
                sort_keys=True,
            )
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
