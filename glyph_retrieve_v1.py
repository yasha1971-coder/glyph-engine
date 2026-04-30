#!/usr/bin/env python3
import argparse
import subprocess
import sys
import json

def run_query(args):
    cmd = [
        "python3",
        "tools/rare_anchor_retrieval_strict_v3.py",
        "--fm", args.fm,
        "--bwt", args.bwt,
        "--chunk-map", args.chunk_map,
        "--server-bin", args.server_bin,
    ]

    if args.query_text:
        cmd += ["--query-text", args.query_text]
    elif args.query_file:
        cmd += ["--query-file", args.query_file]
    else:
        return {"error": "no_query_provided"}

    if args.explain:
        cmd.append("--explain")

    result = subprocess.run(cmd, capture_output=True, text=True)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--server-bin", required=True)

    ap.add_argument("--query-text")
    ap.add_argument("--query-file")

    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--json", action="store_true")

    args = ap.parse_args()

    res = run_query(args)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(res["stdout"])
        if res["stderr"]:
            print(res["stderr"], file=sys.stderr)

    sys.exit(res["returncode"])


if __name__ == "__main__":
    main()
