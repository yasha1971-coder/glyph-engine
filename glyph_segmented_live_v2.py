#!/usr/bin/env python3
import argparse
import json
import time
import sys

from glyph_live_retrieve_v5 import FMServer, MMapSA, retrieve_bytes


def load_query_bytes_line(line):
    line = line.strip()
    if line.startswith("HEX "):
        return bytes.fromhex(line[4:])
    return line.encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--server-bin", required=True)

    # retrieval params
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

    with open(args.config, "r") as f:
        cfg = json.load(f)

    chunks_per_shard = int(cfg["chunks_per_shard"])

    print("LOADING_SHARDS...", file=sys.stderr)
    t0 = time.time()

    shards = []
    for shard in cfg["shards"]:
        sid = int(shard["id"])
        sa_map = MMapSA(shard["sa"])
        server = FMServer(args.server_bin, shard["fm"], shard["bwt"])
        shards.append({
            "id": sid,
            "sa_map": sa_map,
            "server": server,
        })

    print(f"READY {len(shards)} shards in {time.time()-t0:.2f}s", file=sys.stderr)
    sys.stderr.flush()

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            if line == "__EXIT__":
                break

            query = load_query_bytes_line(line)

            t_query = time.time()

            merged = []
            shard_results = []

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
                })

            merged = sorted(set(merged))

            result = {
                "total_merged": len(merged),
                "merged_shortlist": merged[:100],
                "query_time_sec": time.time() - t_query,
                "shards": shard_results,
            }

            print(json.dumps(result))
            sys.stdout.flush()

    finally:
        for sh in shards:
            sh["server"].close()
            sh["sa_map"].close()


if __name__ == "__main__":
    main()
