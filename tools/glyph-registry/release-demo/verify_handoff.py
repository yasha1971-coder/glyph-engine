#!/usr/bin/env python3
"""Trusted receiver for a data-only repository result, pinned before delivery."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
DATA_FILES = {
    "index/registry.json", "index/documents.json", "index/core/manifest.json",
    "index/core/bwt.rlbwt", "index/core/bwt.rlbwt.rank", "index/core/locate_core_s128.bin",
    "query.bin", "results/index-result.json", "results/registry-result.json",
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def check_trusted_installation():
    entries = json.loads((ROOT / "TRUSTED_CODE.json").read_bytes())
    require(entries and "verify_handoff.py" in entries and "trust/Git-anchor.json" in entries, "incomplete trusted-code manifest")
    for name, spec in entries.items():
        rel = PurePosixPath(name)
        require(not rel.is_absolute() and ".." not in rel.parts and str(rel) == name, "unsafe trusted path")
        path = ROOT / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == spec["bytes"] and digest(path) == spec["sha256"], "trusted installation differs: " + name)


def sandbox(root, command):
    return ["bwrap", "--unshare-all", "--die-with-parent", "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/etc",
            "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache", "--bind", str(root), "/work",
            "--ro-bind", str(root / "kit"), "/work/kit", "--ro-bind", str(root / "trusted.json"),
            "/work/trusted.json", "--chdir", "/work", *command]


def run(args):
    # ROOT, code manifest and anchor come from the publisher's separately trusted
    # archive. None are loaded from a sender's transport or RUN.sh.
    check_trusted_installation()
    anchor_path = ROOT / "trust/Git-anchor.json"
    require(digest(anchor_path) == args.expected_anchor_sha, "independent publisher anchor SHA mismatch")
    anchor = json.loads(anchor_path.read_bytes())
    require(anchor["engine"] == "stock", "only the released GLYPH format is accepted here")
    require(len(args.expected_query_sha) == 64 and set(args.expected_query_sha) <= set("0123456789abcdef"), "invalid agreed query SHA")
    require(1 <= args.limit <= 65536, "invalid agreed limit")
    root = Path(args.output).resolve()
    require(root != ROOT and ROOT not in root.parents, "receiver output must be outside trusted installation")
    root.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    transport = root / "input.tar.gz"
    h = hashlib.sha256()
    size = 0
    with Path(args.archive).open("rb") as source, transport.open("xb") as dest:
        for block in iter(lambda: source.read(1 << 20), b""):
            size += len(block)
            require(size <= 128 << 20, "transport outside size limit")
            h.update(block)
            dest.write(block)
    require(h.hexdigest() == args.transport_sha, "transport SHA mismatch")
    kit = root / "kit"
    shutil.copytree(ROOT / "runtime", kit)
    seen = set()
    expanded = 0
    # Exact data allow-list: the receiver never extracts or executes sender code.
    with tarfile.open(transport, "r:gz") as archive:
        for entry in archive:
            path = PurePosixPath(entry.name)
            require(entry.isfile() and not path.is_absolute() and ".." not in path.parts and str(path) == entry.name and path.parts[0] == "kit", "unsafe data member")
            name = "/".join(path.parts[1:])
            require(name in DATA_FILES and name not in seen, "unexpected/duplicate data member: " + name)
            expanded += entry.size
            require(0 <= entry.size <= 64 << 20 and expanded <= 128 << 20, "expanded data outside size limit")
            seen.add(name)
            dest = kit / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(entry) as source, dest.open("xb") as output:
                shutil.copyfileobj(source, output, 1 << 20)
            require(dest.stat().st_size == entry.size, "short data extraction")
    require(seen == DATA_FILES, "incomplete data capsule")
    query = (kit / "query.bin").read_bytes()
    require(1 <= len(query) <= 1024 and 0 not in query and 1 not in query, "unsupported query bytes")
    query.decode("utf-8")
    require(hashlib.sha256(query).hexdigest() == args.expected_query_sha, "agreed query SHA mismatch")
    trusted = dict(anchor, query_sha256=args.expected_query_sha, query_hex=query.hex(), limit=args.limit)
    save(root / "trusted.json", trusted)
    command = ["python3", "kit/chain.py", "verify", "--trusted", "trusted.json", "--state", "state"]
    if args.isolate:
        command = sandbox(root, command)
    save(root / "verify.command.json", {"command": command, "state": "running", "trusted_code_from_sender": False})
    begin = time.perf_counter()
    p = subprocess.run(command, cwd=root, capture_output=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1"))
    (root / "verify.stdout").write_bytes(p.stdout)
    (root / "verify.stderr").write_bytes(p.stderr)
    save(root / "verify.command.json", {"command": command, "exit_code": p.returncode, "process_seconds": time.perf_counter() - begin, "trusted_code_from_sender": False})
    require(p.returncode == 0, "trusted verifier rejected: " + p.stderr.decode(errors="replace").strip())
    result = json.loads(p.stdout)
    require(result["ok"] and result["byte_check"] and result["native_replay"] and not result["canonical_runtime_rebuilt"], "missing pinned-profile PASS")
    return {"ok": True, "scenario": "3->2->1", "count": result["count"], "matches": result["matches"],
            "files_verified": result["files_verified"], "byte_check": True, "native_replay": True,
            "registry_sha256": anchor["registry_sha256"], "corpus_sha256": anchor["corpus_sha256"],
            "publisher_anchor_sha256": args.expected_anchor_sha, "query_sha256": args.expected_query_sha,
            "sender_code_executed": False, "canonical_runtime_rebuilt": False,
            "receiver_seconds": time.perf_counter() - start, "restored_tree": str(root / "state/restored")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["archive", "transport-sha", "expected-anchor-sha", "expected-query-sha", "output"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--isolate", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error), "type": type(error).__name__}), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
