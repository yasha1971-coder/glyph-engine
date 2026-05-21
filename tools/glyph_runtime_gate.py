#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CAPABILITY_PROBE = ROOT / "tools" / "glyph_capability_probe.py"
MINI_MANIFEST = ROOT / "examples" / "mini" / "out" / "manifest.json"
FM_BIN = ROOT / "examples" / "mini" / "out" / "fm.bin"


REQUIRED = {
    "capability_contract_version": "GLYPH_CAPABILITY_CONTRACT_V1",
    "retrieval_contract_version": "GLYPH_RETRIEVAL_CONTRACT_V1",
    "query_protocol_version": "GLYPH_QUERY_PROTOCOL_V1",
    "server_protocol_version": "GLYPH_SERVER_PROTOCOL_V1",
    "http_protocol_version": "GLYPH_HTTP_QUERY_V1",
}


def run_capability_probe():
    out = subprocess.run(
        [str(CAPABILITY_PROBE)],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout

    return json.loads(out)


def check_required_versions(cap):
    failures = []

    for key, expected in REQUIRED.items():
        got = cap.get(key)
        if got != expected:
            failures.append(
                {
                    "field": key,
                    "expected": expected,
                    "got": got,
                }
            )

    return failures


def check_manifest_exists():
    return MINI_MANIFEST.exists()


def check_fm_exists():
    return FM_BIN.exists()


def main():
    result = {
        "runtime_gate_version": "GLYPH_RUNTIME_GATE_V1",
        "capability_contract": "UNKNOWN",
        "retrieval_contract": "UNKNOWN",
        "manifest_integrity": "UNKNOWN",
        "fm_artifact": "UNKNOWN",
        "ready": False,
        "failures": [],
    }

    try:
        cap = run_capability_probe()
    except Exception as e:
        result["capability_contract"] = "FAIL"
        result["failures"].append(
            {
                "check": "capability_probe",
                "error": str(e),
            }
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 2

    version_failures = check_required_versions(cap)

    if version_failures:
        result["capability_contract"] = "FAIL"
        result["retrieval_contract"] = "FAIL"
        result["failures"].extend(version_failures)
    else:
        result["capability_contract"] = "PASS"
        result["retrieval_contract"] = "PASS"

    if check_manifest_exists():
        result["manifest_integrity"] = "PASS"
    else:
        result["manifest_integrity"] = "FAIL"
        result["failures"].append(
            {
                "check": "manifest_exists",
                "path": str(MINI_MANIFEST),
            }
        )

    if check_fm_exists():
        result["fm_artifact"] = "PASS"
    else:
        result["fm_artifact"] = "FAIL"
        result["failures"].append(
            {
                "check": "fm_exists",
                "path": str(FM_BIN),
            }
        )

    result["ready"] = (
        result["capability_contract"] == "PASS"
        and result["retrieval_contract"] == "PASS"
        and result["manifest_integrity"] == "PASS"
        and result["fm_artifact"] == "PASS"
    )

    print(json.dumps(result, sort_keys=True, indent=2))

    if result["ready"]:
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
