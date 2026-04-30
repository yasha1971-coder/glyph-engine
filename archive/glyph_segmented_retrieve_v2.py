#!/usr/bin/env python3
import argparse
import json
import subprocess


def run_shard(args, shard, chunks_per_shard):
    sid = int(shard["id"])

    cmd = [
        "./glyph_live_retrieve_v5.py",
        "--fm", shard["fm"],
        "--bwt", shard["bwt"],
        "--sa", shard["sa"],
        "--server-bin", args.server_bin,
    ]

    if args.query_file:
        cmd += ["--query-file", args.query_file]
    else:
        cmd += ["--query-text", args.query_text]

    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        return {
            "shard_id": sid,
            "error": True,
            "stderr": p.stderr,
            "returncode": p.returncode,
        }

    out = json.loads(p.stdout)
    local = out.get("shortlist_top", [])
    global_ids = [sid * chunks_per_shard + x for x in local]

    return {
        "shard_id": sid,
        "outcome": out.get("outcome"),
        "local_shortlist": local,
        "global_shortlist": global_ids,
        "total_count": out.get("total_count"),
        "timings": out.get("timings", {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--server-bin", required=True)
    ap.add_argument("--query-file")
    ap.add_argument("--query-text")
    args = ap.parse_args()

    if not args.query_file and args.query_text is None:
        raise SystemExit("provide --query-file or --query-text")

    with open(args.config, "r") as f:
        cfg = json.load(f)

    chunks_per_shard = int(cfg["chunks_per_shard"])
    shard_results = []
    merged = []

    for shard in cfg["shards"]:
        r = run_shard(args, shard, chunks_per_shard)
        shard_results.append(r)
        merged.extend(r.get("global_shortlist", []))

    merged = sorted(set(merged))

    result = {
        "mode": "segmented_config_v2",
        "name": cfg.get("name"),
        "shards": len(cfg["shards"]),
        "chunks_per_shard": chunks_per_shard,
        "total_merged": len(merged),
        "merged_shortlist": merged[:100],
        "shard_results": shard_results,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
