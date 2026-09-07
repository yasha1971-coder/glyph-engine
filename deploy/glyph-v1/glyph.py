#!/usr/bin/env python3
"""Single GLYPH V1 command surface.

Existing Personal Vault lifecycle commands stay on the proven V0.1 substrate;
authenticated natural-language retrieval uses the portable V1 adapter.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V0 = HERE.parent / "personal-vault-v0.1" / "glyph.py"
V1 = HERE / "portable_runtime.py"
V1_COMMANDS = {"verify-profile", "doctor-v1", "query", "select", "materialize"}


def git_sha():
    release = HERE / "RELEASE.json"
    if release.is_file():
        try:
            return json.loads(release.read_text(encoding="utf-8")).get("git_sha")
        except Exception:
            return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: glyph <version|doctor|init|add|verify|status|list|search|restore|"
            "free-space|verify-profile|doctor-v1|query|select|materialize> ..."
        )
    command = sys.argv[1]
    if command == "version":
        print(
            json.dumps(
                {
                    "format": "GLYPH_RELEASE_INFO_V1",
                    "version": "1",
                    "git_sha": git_sha(),
                    "exact_vault_substrate": "Personal Vault V0.1",
                    "authenticated_retrieval": "Portable frozen V6.13 profile",
                    "human_gated_materialization": True,
                    "production_claimed": False,
                },
                sort_keys=True,
            )
        )
        return
    if command in V1_COMMANDS:
        translated = "doctor" if command == "doctor-v1" else command
        raise SystemExit(subprocess.call([sys.executable, str(V1), translated, *sys.argv[2:]]))
    if not V0.is_file():
        raise SystemExit(f"GLYPH V0.1 substrate missing: {V0}")
    environment = os.environ.copy()
    raise SystemExit(subprocess.call([sys.executable, str(V0), *sys.argv[1:]], env=environment))


if __name__ == "__main__":
    main()
