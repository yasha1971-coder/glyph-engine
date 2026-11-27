#!/usr/bin/env python3
"""Ready-to-run registry frontend for the accepted release GLYPH index."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")


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
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def check_files():
    manifest = json.loads((ROOT / "FILES.json").read_bytes())
    require(bool(manifest), "empty package manifest")
    for name, item in manifest.items():
        path = ROOT / name
        require(not path.is_symlink() and path.is_file(), "missing package file: " + name)
        require(path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], "package file SHA/length mismatch: " + name)


def checked_example(dataset, query, result):
    expected = json.loads((ROOT / "examples" / dataset / "expected.json").read_bytes())
    if hashlib.sha256(query).hexdigest() != expected["query_sha256"]:
        return False
    known = {(x["path"], x["offset"]) for x in expected["full_matches"]}
    require(len(known) == expected["count"] == result["count"], "example full count mismatch")
    require(all((x["path"], x["offset"]) in known for x in result["matches"]), "example coordinate differs from independent direct-search oracle")
    return True


def search_once(runtime, index, pins, query, limit, state):
    state.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(runtime / "registry.py"), "serve", "--index", str(index),
               "--expected-index", pins["registry_sha256"], "--expected-docs", pins["documents_sha256"],
               "--recovered", str(state / "restored")]
    start = time.perf_counter()
    with (state / "stderr").open("wb") as err:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, env=ENV, text=True, encoding="utf-8", bufsize=1)
        lines = []
        try:
            line = proc.stdout.readline()
            lines.append(line)
            require(bool(line), "registry did not become READY; see search-state/stderr")
            ready = json.loads(line)
            require(ready.get("ready") and ready["files_verified"] == pins["file_count"], "registry READY verification mismatch")
            proc.stdin.write(json.dumps({"op": "query", "hex": query.hex(), "limit": limit}) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            lines.append(line)
            require(bool(line), "query response missing")
            result = json.loads(line)
            require(result.get("ok") and result.get("byte_check"), "native byte check did not PASS")
            require(type(result["count"]) is int and len(result["matches"]) == min(limit, result["count"]), "invalid bounded result cardinality")
            proc.stdin.write('{"op":"quit"}\n')
            proc.stdin.flush()
            proc.stdin.close()
            require(proc.wait() == 0, "registry exited with an error")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            (state / "stdout").write_text("".join(lines), encoding="utf-8")
            save(state / "command.json", {"command": command, "exit_code": proc.returncode, "process_seconds": time.perf_counter() - start})
    return ready, result


def pack_kit(kit, target):
    with target.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|", format=tarfile.GNU_FORMAT) as tar:
            for path in sorted(kit.rglob("*")):
                require(not path.is_symlink(), "unexpected symlink in transfer kit")
                if not path.is_file():
                    continue
                entry = tar.gettarinfo(str(path), "kit/" + path.relative_to(kit).as_posix())
                entry.uid = entry.gid = 0
                entry.uname = entry.gname = "root"
                entry.mtime = 0
                with path.open("rb") as source:
                    tar.addfile(entry, source)


def run(args):
    check_files()
    source = ROOT / "datasets" / args.dataset
    pins = json.loads((source / "pins.json").read_bytes())
    qpath = Path(args.query_file) if args.query_file else ROOT / "examples" / args.dataset / "query.bin"
    query = qpath.read_bytes()
    require(query and 0 not in query and 1 not in query, "query must be nonempty UTF-8 bytes without 00/01; original bytes are preserved")
    query.decode("utf-8")
    require(len(query) <= 1024, "this user profile supports queries of at most 1024 bytes, matching the pinned-result verifier")
    require(0 <= args.limit <= 65536, "limit must be in 0..65536")
    require(args.command != "handoff" or args.limit >= 1, "the pinned-result profile requires limit >= 1")
    output = Path(args.output).resolve()
    require(output != ROOT and ROOT not in output.parents, "choose an output folder outside the immutable user package")
    output.mkdir(parents=True, exist_ok=False)
    runtime = ROOT / "runtime"
    index = source / "index"
    if args.command == "handoff":
        runtime = output / "kit"
        shutil.copytree(ROOT / "runtime", runtime)
        shutil.copytree(index, runtime / "index")
        index = runtime / "index"
        (runtime / "query.bin").write_bytes(query)
    ready, result = search_once(runtime, index, pins, query, args.limit, output / "search-state")
    example = checked_example(args.dataset, query, result)
    search = {"ok": True, "dataset": args.dataset, "query_sha256": hashlib.sha256(query).hexdigest(), "limit": args.limit,
              "registry_sha256": pins["registry_sha256"], "corpus_sha256": pins["corpus_sha256"], "ready": ready,
              **result, "example_oracle_check": example}
    save(output / "search.json", search)
    if args.command == "search":
        return {"ok": True, "scenario": "3", "count": result["count"], "matches": result["matches"],
                "byte_check": True, "files_verified": ready["files_verified"], "example_oracle_check": example,
                "report": str(output / "search.json")}
    trusted = {**pins, "query_sha256": hashlib.sha256(query).hexdigest(), "query_hex": query.hex(), "limit": args.limit}
    save(output / "trusted.json", trusted)
    (runtime / "results").mkdir()
    command = [sys.executable, str(runtime / "chain.py"), "create", "--trusted", str(output / "trusted.json"),
               "--state", str(output / "create-state"), "--search-result", str(output / "search.json")]
    start = time.perf_counter()
    p = subprocess.run(command, capture_output=True, env=ENV)
    (output / "create.stdout").write_bytes(p.stdout)
    (output / "create.stderr").write_bytes(p.stderr)
    save(output / "create.command.json", {"command": command, "exit_code": p.returncode, "process_seconds": time.perf_counter() - start})
    require(p.returncode == 0 and json.loads(p.stdout)["ok"], "actual glyph-evidence create-index failed; see create.stderr")
    transfer = output / "transfer"
    transfer.mkdir()
    archive = transfer / "kit.tar.gz"
    pack_kit(runtime, archive)
    archive_sha = digest(archive)
    (transfer / "kit.tar.gz.sha256").write_text(archive_sha + "  kit.tar.gz\n", encoding="ascii")
    shutil.copy2(output / "trusted.json", transfer / "trusted.json")
    trusted_sha = digest(transfer / "trusted.json")
    shutil.copy2(ROOT / "receive.py", transfer / "receive.py")
    script = '''#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then echo "usage: bash RUN.sh /new/receiver [--isolate]" >&2; exit 2; fi
receiver=$1
shift
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$here/receive.py" --archive "$here/kit.tar.gz" --expected-sha ARCHIVE_SHA --trusted "$here/trusted.json" --expected-trusted-sha TRUSTED_SHA --receiver "$receiver" "$@"
'''.replace("ARCHIVE_SHA", archive_sha).replace("TRUSTED_SHA", trusted_sha)
    (transfer / "RUN.sh").write_text(script, encoding="utf-8")
    (transfer / "RUN.sh").chmod(0o755)
    receipt = {"ok": True, "scenario": "1+2+3", "count": result["count"], "matches": result["matches"],
               "byte_check": True, "transfer_directory": str(transfer), "archive_sha256": archive_sha,
               "trusted_sha256": trusted_sha, "transferred_bytes": sum(x.stat().st_size for x in transfer.iterdir()),
               "files": {x.name: {"bytes": x.stat().st_size, "sha256": digest(x)} for x in sorted(transfer.iterdir())},
               "number1_profile": "glyph-index-result-v1 pinned index; canonical_runtime_rebuilt=false"}
    save(output / "handoff.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["search", "handoff"])
    parser.add_argument("--dataset", choices=["Git", "CPython"], default="Git")
    parser.add_argument("--query-file", help="exact UTF-8 query bytes; default is the fixed example")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", required=True, help="new folder outside the immutable package")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error), "type": type(error).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
