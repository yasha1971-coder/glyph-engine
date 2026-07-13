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

FORMAT = "GLYPH_OPERATOR_CONFORMANCE_GRAPH_V1"
OPERATOR_PROFILE = "GLYPH_OPERATOR_PATH_V1"

DEFAULT_RESULT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_OPERATOR_CONFORMANCE_GRAPH_V1.json"
)

PROOF_GRAPH_RESULT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_PROOF_GRAPH_V1.json"
)

RUNTIME_GRAPH_RESULT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_RUNTIME_CONFORMANCE_GRAPH_V1.json"
)

NODE_DEFINITIONS = [
    {
        "id": "O1",
        "name": "Deterministic Filesystem Manifest",
        "checker": "tools/check_operator_manifest_v1.py",
        "result": (
            "benchmarks/results/"
            "GLYPH_OPERATOR_MANIFEST_V1.json"
        ),
        "spec": (
            "docs/specs/"
            "GLYPH_OPERATOR_CORPUS_MANIFEST_V1.md"
        ),
        "depends": ["P1-P12", "R0-R6"],
    },
    {
        "id": "O2",
        "name": "Deterministic Runtime Index",
        "checker": "tools/check_operator_index_v1.py",
        "result": (
            "benchmarks/results/"
            "GLYPH_OPERATOR_RUNTIME_INDEX_V1.json"
        ),
        "spec": (
            "docs/specs/"
            "GLYPH_OPERATOR_RUNTIME_INDEX_V1.md"
        ),
        "depends": ["O1"],
    },
    {
        "id": "O3",
        "name": "Binary Query and Source Mapping",
        "checker": "tools/check_operator_query_v1.py",
        "result": (
            "benchmarks/results/"
            "GLYPH_OPERATOR_QUERY_V1.json"
        ),
        "spec": (
            "docs/specs/"
            "GLYPH_OPERATOR_QUERY_V1.md"
        ),
        "depends": ["O2"],
    },
    {
        "id": "O4",
        "name": "Self-contained Operator Evidence Bundle",
        "checker": "tools/check_operator_bundle_v1.py",
        "result": (
            "benchmarks/results/"
            "GLYPH_OPERATOR_EVIDENCE_BUNDLE_V1.json"
        ),
        "spec": (
            "docs/specs/"
            "GLYPH_OPERATOR_EVIDENCE_BUNDLE_V1.md"
        ),
        "depends": ["O3"],
    },
    {
        "id": "O5",
        "name": "One-command Operator Workflow",
        "checker": "tools/check_operator_workflow_v1.py",
        "result": (
            "benchmarks/results/"
            "GLYPH_OPERATOR_WORKFLOW_V1.json"
        ),
        "spec": (
            "docs/specs/"
            "GLYPH_OPERATOR_WORKFLOW_V1.md"
        ),
        "depends": ["O4"],
    },
]

REQUIRED_NODE_IDS = [
    "O1",
    "O2",
    "O3",
    "O4",
    "O5",
    "O6",
]

EXPECTED_DEPENDENCIES = {
    "O1": ["P1-P12", "R0-R6"],
    "O2": ["O1"],
    "O3": ["O2"],
    "O4": ["O3"],
    "O5": ["O4"],
    "O6": [
        "O1",
        "O2",
        "O3",
        "O4",
        "O5",
        "P1-P12",
        "R0-R6",
    ],
}

EXTERNAL_DEPENDENCIES = {
    "P1-P12",
    "R0-R6",
}


class OperatorConformanceError(RuntimeError):
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


def validate_sha256(
    value: Any,
    field: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise OperatorConformanceError(
            f"invalid SHA256: {field}"
        )

    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OperatorConformanceError(
            "required JSON missing: "
            + str(path.relative_to(ROOT))
        )

    try:
        value = json.loads(path.read_text())
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise OperatorConformanceError(
            "invalid JSON: "
            + str(path.relative_to(ROOT))
        ) from error

    if not isinstance(value, dict):
        raise OperatorConformanceError(
            "JSON root must be object: "
            + str(path.relative_to(ROOT))
        )

    return value


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise OperatorConformanceError(message)


def require_true_fields(
    node_id: str,
    result: dict[str, Any],
    fields: list[str],
) -> None:
    for field in fields:
        require(
            result.get(field) is True,
            f"{node_id} required field is not true: "
            f"{field}",
        )


def validate_mutations(
    node_id: str,
    result: dict[str, Any],
    minimum_count: int,
) -> int:
    mutations = result.get("mutations")
    declared_count = result.get(
        "mutation_count"
    )

    require(
        isinstance(mutations, list),
        f"{node_id} mutations must be a list",
    )
    require(
        isinstance(declared_count, int)
        and not isinstance(declared_count, bool),
        f"{node_id} mutation_count must be an integer",
    )
    require(
        declared_count >= minimum_count,
        f"{node_id} mutation_count regressed below "
        f"minimum {minimum_count}: {declared_count}",
    )
    require(
        len(mutations) == declared_count,
        f"{node_id} mutation_count differs from "
        "mutation list size",
    )
    require(
        all(
            isinstance(item, dict)
            and item.get("rejected") is True
            for item in mutations
        ),
        f"{node_id} contains accepted mutation",
    )

    return declared_count

def validate_reference_graphs() -> dict[str, Any]:
    proof = load_json(PROOF_GRAPH_RESULT)

    require(
        proof.get("ok") is True,
        "reference proof graph is not ok",
    )
    require(
        proof.get("proof_count") == 12,
        "reference proof count mismatch",
    )
    require(
        proof.get("passed") == 12,
        "reference proof graph did not pass",
    )
    require(
        proof.get("failed") == 0,
        "reference proof graph contains failure",
    )
    require(
        proof.get("verify_ok_permitted") is True,
        "reference proof graph does not permit verify",
    )
    require(
        proof.get("proof_graph_version")
        == "GLYPH_PROOF_GRAPH_V1",
        "reference proof graph version mismatch",
    )

    runtime = load_json(RUNTIME_GRAPH_RESULT)

    require(
        runtime.get("ok") is True,
        "runtime conformance graph is not ok",
    )
    require(
        runtime.get("format")
        == "GLYPH_RUNTIME_CONFORMANCE_GRAPH_V1",
        "runtime graph format mismatch",
    )
    require(
        runtime.get("node_count") == 7,
        "runtime node count mismatch",
    )
    require(
        runtime.get("passed") == 7,
        "runtime graph did not pass seven nodes",
    )
    require(
        runtime.get("failed") == 0,
        "runtime graph contains failure",
    )
    require(
        runtime.get("runtime_conformant") is True,
        "runtime is not conformant",
    )
    require(
        runtime.get("verify_ok_permitted") is True,
        "runtime graph does not permit verify",
    )

    return {
        "proof_graph": {
            "id": "P1-P12",
            "status": "PASS",
            "version":
                proof["proof_graph_version"],
            "passed": proof["passed"],
            "result": str(
                PROOF_GRAPH_RESULT.relative_to(ROOT)
            ),
            "result_sha256":
                sha256_file(PROOF_GRAPH_RESULT),
        },
        "runtime_graph": {
            "id": "R0-R6",
            "status": "PASS",
            "version":
                runtime["format"],
            "passed": runtime["passed"],
            "result": str(
                RUNTIME_GRAPH_RESULT.relative_to(ROOT)
            ),
            "result_sha256":
                sha256_file(RUNTIME_GRAPH_RESULT),
        },
    }


def run_checker(
    checker: Path,
    result_path: Path,
) -> dict[str, Any]:
    if not checker.is_file():
        raise OperatorConformanceError(
            "checker missing: "
            + str(checker.relative_to(ROOT))
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
        timeout=3600,
        check=False,
    )

    if completed.returncode != 0:
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or "no checker output"
        )

        raise OperatorConformanceError(
            f"{checker.name} failed with code "
            f"{completed.returncode}: "
            f"{detail[-8000:]}"
        )

    if not result_path.is_file():
        raise OperatorConformanceError(
            "checker did not create result: "
            + str(result_path.relative_to(ROOT))
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
        result.get("operator_obligation")
        == node_id,
        f"{node_id} obligation mismatch",
    )

    if node_id == "O1":
        require_true_fields(
            node_id,
            result,
            [
                "deterministic_source_discovery",
                "raw_path_byte_ordering_verified",
                "stable_doc_id_assignment",
                "invalid_utf8_filename_supported",
                "embedded_nul_content_supported",
                "byte_ff_content_supported",
                "empty_files_preserved",
                "duplicate_documents_preserved",
                "source_stability_verified",
                "source_mutation_rejected",
                "symlink_rejected",
            ],
        )
        mutation_count = validate_mutations(
            node_id,
            result,
            11,
        )
        validate_sha256(
            result.get("corpus_id"),
            "O1 corpus_id",
        )
        validate_sha256(
            result.get("source_manifest_id"),
            "O1 source_manifest_id",
        )
        require(
            result.get("next_operator_obligation")
            == "O2_RUNTIME_INDEX_BUILD",
            "O1 handoff mismatch",
        )

        return {
            "deterministic_manifest": True,
            "document_count":
                result.get("document_count"),
            "total_source_bytes":
                result.get("total_source_bytes"),
            "mutation_count": mutation_count,
        }

    if node_id == "O2":
        require_true_fields(
            node_id,
            result,
            [
                "built_from_committed_snapshot",
                "private_verified_inputs_used",
                "binary_safe_arbitrary_bytes",
                "one_index_per_document",
                "empty_document_indexed",
                "duplicate_documents_preserved",
                "runtime_binary_commitments_bound",
                "source_manifest_bound",
                "corpus_identity_bound",
                "source_manifest_identity_bound",
                "per_document_index_commitments",
                "deterministic_manifest_verified",
                "deterministic_payloads_verified",
                "deterministic_rebuild_verified",
                "atomic_publication_verified",
                "interrupted_build_rejected",
                "snapshot_mutation_rejected",
                "structural_verification_passed",
            ],
        )
        mutation_count = validate_mutations(
            node_id,
            result,
            13,
        )

        require(
            result.get("original_source_directory_used")
            is False,
            "O2 unexpectedly used original source",
        )
        require(
            result.get("logical_sentinel") == 256,
            "O2 logical sentinel mismatch",
        )
        require(
            result.get("alphabet_size") == 257,
            "O2 alphabet size mismatch",
        )
        validate_sha256(
            result.get("runtime_index_id"),
            "O2 runtime_index_id",
        )
        require(
            result.get("next_operator_obligation")
            == "O3_BINARY_QUERY_AND_SOURCE_MAPPING",
            "O2 handoff mismatch",
        )

        return {
            "runtime_index_conformant": True,
            "document_count":
                result.get("document_count"),
            "total_runtime_bytes":
                result.get("total_runtime_bytes"),
            "mutation_count": mutation_count,
        }

    if node_id == "O3":
        require_true_fields(
            node_id,
            result,
            [
                "compiled_count_used",
                "compiled_locate_used",
                "count_locate_agreement_verified",
                "query_file_transport_verified",
                "query_hex_transport_verified",
                "query_file_equals_hex",
                "embedded_nul_query_verified",
                "byte_ff_query_verified",
                "all_256_byte_query_verified",
                "invalid_utf8_path_mapping_verified",
                "zero_match_verified",
                "cross_document_match_rejected",
                "bounded_multidoc_locate_verified",
                "global_coordinate_order_verified",
                "independent_byte_check_verified",
                "query_binary_commitments_bound",
                "source_mutation_rejected",
                "runtime_mutation_rejected",
                "query_file_mutation_rejected",
                "deterministic_results_verified",
            ],
        )
        mutation_count = validate_mutations(
            node_id,
            result,
            15,
        )
        require(
            result.get("next_operator_obligation")
            == "O4_OPERATOR_EVIDENCE_BUNDLE",
            "O3 handoff mismatch",
        )

        return {
            "binary_query_conformant": True,
            "fixture_count":
                result.get("fixture_count"),
            "bounded_fixture_count":
                result.get("bounded_fixture_count"),
            "mutation_count": mutation_count,
        }

    if node_id == "O4":
        require_true_fields(
            node_id,
            result,
            [
                "deterministic_bundle_verified",
                "self_contained",
                "source_documents_bundled",
                "runtime_indexes_bundled",
                "runtime_builder_binaries_bundled",
                "query_binaries_bundled",
                "query_bytes_bundled",
                "query_artifact_bundled",
                "replay_code_bundled",
                "exact_manifest_coverage_verified",
                "payload_hashes_verified",
                "bundle_root_verified",
                "source_manifest_verified",
                "runtime_manifest_verified",
                "compiled_query_replay_verified",
                "copied_bundle_replay_verified",
                "replay_outside_repository_verified",
                "original_source_paths_removed_before_replay",
                "atomic_publication_verified",
                "interrupted_build_rejected",
            ],
        )
        mutation_count = validate_mutations(
            node_id,
            result,
            16,
        )

        require(
            result.get("repository_dependency_required")
            is False,
            "O4 still requires repository",
        )
        require(
            result.get("network_dependency_required")
            is False,
            "O4 unexpectedly requires network",
        )
        require(
            result.get("external_data_dependencies")
            == [],
            "O4 has external data dependencies",
        )
        validate_sha256(
            result.get("bundle_root_sha256"),
            "O4 bundle root",
        )
        require(
            result.get("next_operator_obligation")
            == "O5_ONE_COMMAND_OPERATOR_WORKFLOW",
            "O4 handoff mismatch",
        )

        return {
            "portable_bundle_conformant": True,
            "file_count":
                result.get("file_count"),
            "document_count":
                result.get("document_count"),
            "mutation_count": mutation_count,
        }

    if node_id == "O5":
        require_true_fields(
            node_id,
            result,
            [
                "one_command_workflow_verified",
                "o1_snapshot_executed",
                "o2_runtime_index_executed",
                "o3_binary_query_executed",
                "o4_evidence_bundle_executed",
                "query_file_cli_verified",
                "query_hex_cli_verified",
                "query_file_equals_query_hex",
                "embedded_nul_query_verified",
                "bounded_result_verified",
                "deterministic_case_tree_verified",
                "case_verify_verified",
                "source_removed_before_verify",
                "query_file_removed_before_verify",
                "bundle_replay_outside_repository_verified",
                "exact_case_root_coverage_verified",
                "source_snapshot_verified",
                "runtime_index_verified",
                "bundle_replay_verified",
                "workflow_complete_verified",
                "atomic_publication_verified",
                "interrupted_workflow_rejected",
                "invalid_utf8_filename_supported",
            ],
        )
        mutation_count = validate_mutations(
            node_id,
            result,
            19,
        )

        require(
            result.get("network_dependency_required")
            is False,
            "O5 unexpectedly requires network",
        )
        require(
            result.get(
                "bundle_repository_dependency_required"
            )
            is False,
            "O5 bundle still requires repository",
        )
        validate_sha256(
            result.get("workflow_result_id"),
            "O5 workflow result",
        )
        validate_sha256(
            result.get("query_result_id"),
            "O5 query result",
        )
        validate_sha256(
            result.get("runtime_index_id"),
            "O5 runtime index",
        )
        validate_sha256(
            result.get("bundle_root_sha256"),
            "O5 bundle root",
        )
        require(
            result.get("next_operator_obligation")
            == "O6_OPERATOR_CONFORMANCE_CLOSURE",
            "O5 handoff mismatch",
        )

        return {
            "one_command_workflow_conformant":
                True,
            "document_count":
                result.get("document_count"),
            "match_count":
                result.get("match_count"),
            "returned_count":
                result.get("returned_count"),
            "mutation_count": mutation_count,
        }

    raise OperatorConformanceError(
        f"unknown operator node: {node_id}"
    )


def execute_operator_nodes() -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []

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
            "checker": definition["checker"],
            "checker_sha256":
                sha256_file(checker),
            "spec": definition["spec"],
            "spec_sha256":
                sha256_file(spec),
            "result": definition["result"],
            "result_sha256":
                sha256_file(result_path),
            "summary": summary,
        })

    graph_spec = (
        ROOT
        / "docs/specs/"
        / "GLYPH_OPERATOR_CONFORMANCE_GRAPH_V1.md"
    )
    graph_checker = Path(__file__).resolve()

    require(
        graph_spec.is_file(),
        "operator graph specification missing",
    )

    nodes.append({
        "id": "O6",
        "name": "Operator Conformance Closure",
        "status": "PASS",
        "depends":
            list(EXPECTED_DEPENDENCIES["O6"]),
        "checker":
            str(graph_checker.relative_to(ROOT)),
        "checker_sha256":
            sha256_file(graph_checker),
        "spec":
            str(graph_spec.relative_to(ROOT)),
        "spec_sha256":
            sha256_file(graph_spec),
        "summary": {
            "operator_conformant": True,
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
        raise OperatorConformanceError(
            f"operator node sequence mismatch: {ids}"
        )

    if len(set(ids)) != len(ids):
        raise OperatorConformanceError(
            "duplicate operator node"
        )

    seen: set[str] = set()

    for node in nodes:
        node_id = node.get("id")

        if node.get("status") != "PASS":
            raise OperatorConformanceError(
                "operator node did not pass: "
                + str(node_id)
            )

        depends = node.get("depends")

        if (
            not isinstance(depends, list)
            or depends
            != EXPECTED_DEPENDENCIES[node_id]
        ):
            raise OperatorConformanceError(
                "operator dependency declaration "
                f"mismatch: {node_id}"
            )

        for dependency in depends:
            if dependency in EXTERNAL_DEPENDENCIES:
                continue

            if dependency not in seen:
                raise OperatorConformanceError(
                    "operator dependency not satisfied: "
                    f"{node_id} -> {dependency}"
                )

        seen.add(node_id)


def expect_rejected(
    name: str,
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        validate_node_sequence(nodes)
    except OperatorConformanceError as error:
        return {
            "mutation": name,
            "rejected": True,
            "message": str(error),
        }

    raise OperatorConformanceError(
        "operator graph mutation accepted: "
        + name
    )


def mutation_gate(
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    missing = [
        node
        for node in copy.deepcopy(nodes)
        if node["id"] != "O4"
    ]

    duplicate = copy.deepcopy(nodes)
    duplicate.insert(
        3,
        copy.deepcopy(
            next(
                node
                for node in duplicate
                if node["id"] == "O3"
            )
        ),
    )

    reordered = copy.deepcopy(nodes)
    o2_index = next(
        index
        for index, node in enumerate(reordered)
        if node["id"] == "O2"
    )
    o3_index = next(
        index
        for index, node in enumerate(reordered)
        if node["id"] == "O3"
    )
    reordered[o2_index], reordered[o3_index] = (
        reordered[o3_index],
        reordered[o2_index],
    )

    failed = copy.deepcopy(nodes)
    next(
        node
        for node in failed
        if node["id"] == "O5"
    )["status"] = "FAIL"

    unknown_dependency = copy.deepcopy(nodes)
    next(
        node
        for node in unknown_dependency
        if node["id"] == "O1"
    )["depends"].append("R7")

    closure_skip = copy.deepcopy(nodes)
    next(
        node
        for node in closure_skip
        if node["id"] == "O6"
    )["depends"].remove("O4")

    return [
        expect_rejected(
            "missing_required_node",
            missing,
        ),
        expect_rejected(
            "duplicate_operator_node",
            duplicate,
        ),
        expect_rejected(
            "dependency_order_violation",
            reordered,
        ),
        expect_rejected(
            "failed_operator_node",
            failed,
        ),
        expect_rejected(
            "undeclared_external_dependency",
            unknown_dependency,
        ),
        expect_rejected(
            "closure_dependency_skip",
            closure_skip,
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
        external = validate_reference_graphs()

        nodes = execute_operator_nodes()
        validate_node_sequence(nodes)

        mutations = mutation_gate(nodes)

        require(
            all(
                item["rejected"] is True
                for item in mutations
            ),
            "operator graph mutation gate failed",
        )

        output = {
            "ok": True,
            "format": FORMAT,
            "operator_profile":
                OPERATOR_PROFILE,
            "proof_graph_dependency":
                external["proof_graph"],
            "runtime_graph_dependency":
                external["runtime_graph"],
            "node_count": len(nodes),
            "passed": len(nodes),
            "failed": 0,
            "all_required_nodes_present":
                True,
            "all_dependencies_satisfied":
                True,
            "deterministic_manifest_conformant":
                True,
            "runtime_index_conformant":
                True,
            "binary_query_conformant":
                True,
            "portable_bundle_conformant":
                True,
            "one_command_workflow_conformant":
                True,
            "operator_conformant": True,
            "verify_ok_permitted": True,
            "nodes": nodes,
            "mutation_count":
                len(mutations),
            "mutations": mutations,
        }

        result_path = Path(
            args.result
        ).resolve()
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
            "operator_profile":
                OPERATOR_PROFILE,
            "operator_conformant": False,
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
