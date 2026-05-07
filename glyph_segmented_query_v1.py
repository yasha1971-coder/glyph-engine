#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path


def start_backend(fm: str, bwt: str):
    p = subprocess.Popen(
        ["./build/query_fm_server_v1", fm, bwt],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    ready = p.stderr.readline().strip()
    if ready != "READY":
        raise RuntimeError(f"backend not ready: {ready}")
    return p


def query_backend(proc, hex_pattern: str) -> int:
    proc.stdin.write(hex_pattern + "\n")
    proc.stdin.flush()

    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("backend returned no output")

    # expected compact line from query_fm_server_v1; count is first/last numeric depending backend format
    nums = []
    for part in line.replace(":", " ").replace(",", " ").split():
        if part.isdigit():
            nums.append(int(part))

    if not nums:
        raise RuntimeError(f"could not parse backend output: {line!r}")

    return nums[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hex", required=True)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    shards = manifest["shards"]

    backends = []
    try:
        for shard in shards:
            backends.append((shard, start_backend(shard["fm"], shard["bwt"])))

        total = 0
        results = []

        t0 = time.time()

        for shard, proc in backends:
            count = query_backend(proc, args.hex)
            total += count
            results.append({
                "shard_id": shard["id"],
                "global_offset": shard["global_offset"],
                "count": count,
            })

        dt = time.time() - t0

        print(json.dumps({
            "manifest": manifest["name"],
            "shards": len(shards),
            "total_count": total,
            "query_time_sec": dt,
            "results": results,
        }, indent=2))

    finally:
        for _, proc in backends:
            proc.kill()


if __name__ == "__main__":
    main()
