#!/usr/bin/env python3
import argparse
import json
import time

from glyph_live_retrieve_v5 import FMServer, MMapSA, retrieve_bytes


def load_query_bytes(args):
    if args.query_text is not None:
        return args.query_text.encode("utf-8")
    if args.query_file is not None:
        with open(args.query_file, "rb") as f:
            return f.read()
    raise ValueError("provide --query-text or --query-file")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True)
    ap.add_argument("--server-bin", required=True)
    ap.add_argument("--query-file")
    ap.add_argument("--query-text")

    # passthrough retrieval params
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--window-step", type=int, default=64)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--pick-k", type=int, default=3)
    ap.add_argument("--entropy-min", type=float, default=2.0)
    ap.add_argument("--non-selective-threshold", type=int, default=16)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-query-bytes", type=int, default=1048576)
    ap.add_argument("--max-range-scan", type=int, default=10000)

    args = ap.parse_args()

    query = load_query_bytes(args)

    with open(args.config, "r") as f:
        cfg = json.load(f)

    chunks_per_shard = int(cfg["chunks_per_shard"])

    shards = []
    t_startup0 = time.time()

    for shard in cfg["shards"]:
        sid = int(shard["id"])
        sa_map = MMapSA(shard["sa"])
        server = FMServer(args.server_bin, shard["fm"], shard["bwt"])
        shards.append({
            "id": sid,
            "sa_map": sa_map,
            "server": server,
        })

    startup_time = time.time() - t_startup0

    t0 = time.time()
    shard_results = []
    merged = []

    try:
        for sh in shards:
            sid = sh["id"]
            out = retrieve_bytes(query, sh["server"], sh["sa_map"], args)

            local = out.get("shortlist_top", [])
            global_ids = [sid * chunks_per_shard + x for x in local]
            merged.extend(global_ids)

            shard_results.append({
                "shard_id": sid,
                "outcome": out.get("outcome"),
                "local_shortlist": local,
                "global_shortlist": global_ids,
                "total_count": out.get("total_count"),
                "timings": out.get("timings", {}),
            })

        merged = sorted(set(merged))

        result = {
            "mode": "segmented_live_v1",
            "name": cfg.get("name"),
            "shards": len(shards),
            "chunks_per_shard": chunks_per_shard,
            "startup_time_sec": startup_time,
            "query_wall_time_sec": time.time() - t0,
            "total_merged": len(merged),
            "merged_shortlist": merged[:100],
            "shard_results": shard_results,
        }

        print(json.dumps(result, indent=2))

    finally:
        for sh in shards:
            sh["server"].close()
            sh["sa_map"].close()


if __name__ == "__main__":
    main()
