#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_bytes_at(path: Path, off: int, n: int) -> bytes:
    with path.open("rb") as f:
        f.seek(off)
        return f.read(n)


def parse_server_line(line: str):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 9:
        raise RuntimeError(f"bad server line: {line!r}")
    if parts[0] != "OK":
        raise RuntimeError(f"server error: {line!r}")

    offsets = []
    if parts[8]:
        offsets = [int(x) for x in parts[8].split(",") if x]

    return {
        "l": int(parts[1]),
        "r": int(parts[2]),
        "match_count": int(parts[3]),
        "located_count": int(parts[4]),
        "bounded": parts[5] == "true",
        "total_steps": int(parts[6]),
        "max_steps": int(parts[7]),
        "offsets": offsets,
    }


def run_server_once(runtime_dir: Path, query_hex: str, max_offsets: int):
    server = ROOT / "build" / "rlbwt_full_query_locate_server_v1"
    if not server.exists():
        raise FileNotFoundError(f"missing server binary: {server}")

    proc = subprocess.run(
        [
            str(server),
            str(runtime_dir / "bwt.rlbwt"),
            str(runtime_dir / "bwt.rlbwt.rank"),
            str(runtime_dir / "locate_core_s128.bin"),
        ],
        input=f"{query_hex}\t{max_offsets}\nQUIT\n",
        text=True,
        capture_output=True,
        check=True,
    )

    lines = [x for x in proc.stdout.splitlines() if x.strip()]
    if not lines:
        raise RuntimeError(f"server produced no stdout, stderr={proc.stderr}")

    return parse_server_line(lines[0])


def fail(errors, msg):
    errors.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay-verify RLBWT bounded evidence artifact V1.")
    ap.add_argument("--artifact", required=True)
    args = ap.parse_args()

    artifact_path = Path(args.artifact).resolve()
    artifact = json.loads(artifact_path.read_text())

    errors = []

    if artifact.get("artifact_version") != "RLBWT_BOUNDED_EVIDENCE_V1":
        fail(errors, "bad artifact_version")

    if artifact.get("profile") != "RLBWT_FULL_RUNTIME_PROFILE_V1":
        fail(errors, "bad profile")

    query = artifact["query"]
    query_text = query["text"]
    query_bytes = query_text.encode("utf-8")
    query_hex = query_bytes.hex()
    query_sha256 = hashlib.sha256(query_bytes).hexdigest()

    if query_hex != query["hex"]:
        fail(errors, "query hex mismatch")

    if query_sha256 != query["sha256"]:
        fail(errors, "query sha256 mismatch")

    corpus_info = artifact["source_corpus"]
    corpus_path = Path(corpus_info["path"]).resolve()

    if not corpus_path.exists():
        fail(errors, f"missing source corpus: {corpus_path}")
    else:
        if corpus_path.stat().st_size != corpus_info["bytes"]:
            fail(errors, "source corpus size mismatch")
        if sha256_file(corpus_path) != corpus_info["sha256"]:
            fail(errors, "source corpus sha256 mismatch")

    runtime_dir = Path(artifact["runtime_dir"]).resolve()

    for rf in artifact["runtime_files"]:
        p = Path(rf["path"]).resolve()
        if not p.exists():
            fail(errors, f"missing runtime file: {p}")
            continue
        if p.stat().st_size != rf["bytes"]:
            fail(errors, f"runtime file size mismatch: {rf['name']}")
        if sha256_file(p) != rf["sha256"]:
            fail(errors, f"runtime file sha256 mismatch: {rf['name']}")

    retrieval = artifact["retrieval"]
    replay = None

    if not errors:
        replay = run_server_once(runtime_dir, query_hex, int(retrieval["max_offsets"]))

        if [replay["l"], replay["r"]] != retrieval["fm_interval"]:
            fail(errors, "FM interval mismatch")

        if replay["match_count"] != retrieval["match_count"]:
            fail(errors, "match_count mismatch")

        if replay["located_count"] != retrieval["returned_count"]:
            fail(errors, "returned_count mismatch")

        if replay["bounded"] != retrieval["bounded"]:
            fail(errors, "bounded flag mismatch")

        if replay["offsets"] != retrieval["offsets"]:
            fail(errors, "offset list mismatch")

    byte_checks = []
    byte_ok = True

    if corpus_path.exists():
        for off in retrieval["offsets"]:
            got = read_bytes_at(corpus_path, off, len(query_bytes))
            ok = got == query_bytes
            byte_ok = byte_ok and ok
            byte_checks.append({
                "offset": off,
                "expected_hex": query_hex,
                "observed_hex": got.hex(),
                "ok": ok,
            })

    if not byte_ok:
        fail(errors, "byte_check failed")

    if artifact.get("byte_check", {}).get("all_returned_offsets_match_query") is not True:
        fail(errors, "artifact byte_check was not true")

    result = {
        "ok": not errors,
        "artifact": str(artifact_path),
        "artifact_version": artifact.get("artifact_version"),
        "profile": artifact.get("profile"),
        "query": query_text,
        "fm_interval": retrieval["fm_interval"],
        "match_count": retrieval["match_count"],
        "max_offsets": retrieval["max_offsets"],
        "returned_count": retrieval["returned_count"],
        "bounded": retrieval["bounded"],
        "offsets": retrieval["offsets"],
        "byte_check": byte_ok,
        "replay_result": replay,
        "errors": errors,
    }

    print(json.dumps(result, indent=2, sort_keys=True))

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
