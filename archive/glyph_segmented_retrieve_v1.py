#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys


def run_shard(args, shard_id, chunks_per_shard):
    cmd = [
        "./glyph_live_retrieve_v5.py",
        "--fm", args.fm,
        "--bwt", args.bwt,
        "--sa", args.sa,
        "--server-bin", args.server_bin,
    ]

    if args.query_file:
        cmd += ["--query-file", args.query_file]
    else:
        cmd += ["--query-text", args.query_text]

    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        return {
            "shard_id": shard_id,
            "error": True,
            "stderr": p.stderr,
            "returncode": p.returncode,
        }

    out = json.loads(p.stdout)

    local = out.get("shortlist_top", [])
    global_ids = [shard_id * chunks_per_shard + x for x in local]

    return {
        "shard_id": shard_id,
        "outcome": out.get("outcome"),
        "local_shortlist": local,
        "global_shortlist": global_ids,
        "total_count": out.get("total_count"),
        "timings": out.get("timings", {}),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--sa", required=True)
    ap.add_argument("--server-bin", required=True)

    ap.add_argument("--query-file")
    ap.add_argument("--query-text")

    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--chunks-per-shard", type=int, default=244141)

    args = ap.parse_args()

    if not args.query_file and args.query_text is None:
        raise SystemExit("provide --query-file or --query-text")

    shard_results = []
    merged = []

    for sid in range(args.shards):
        r = run_shard(args, sid, args.chunks_per_shard)
        shard_results.append(r)
        merged.extend(r.get("global_shortlist", []))

    merged = sorted(set(merged))

    result = {
        "mode": "segmented_prototype",
        "shards": args.shards,
        "chunks_per_shard": args.chunks_per_shard,
        "total_merged": len(merged),
        "merged_shortlist": merged[:100],
        "shard_results": shard_results,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
