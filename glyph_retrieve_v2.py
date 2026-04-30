#!/usr/bin/env python3
import argparse
import ast
import json
import subprocess
import sys


def parse_bool(s):
    return s.strip().lower() == "true"


def parse_strict_output(text):
    result = {
        "outcome": None,
        "shortlist_size": None,
        "total_count": None,
        "truncated": None,
        "shortlist_top": [],
        "timings": {},
        "raw_stdout": text,
    }

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("outcome ="):
            result["outcome"] = line.split("=", 1)[1].strip()

        elif line.startswith("shortlist_size ="):
            result["shortlist_size"] = int(line.split("=", 1)[1].strip())

        elif line.startswith("total_count ="):
            result["total_count"] = int(line.split("=", 1)[1].strip())

        elif line.startswith("truncated ="):
            result["truncated"] = parse_bool(line.split("=", 1)[1].strip())

        elif line.startswith("shortlist_top ="):
            val = line.split("=", 1)[1].strip()
            try:
                result["shortlist_top"] = ast.literal_eval(val)
            except Exception:
                result["shortlist_top"] = []

        elif line.startswith("total_time_sec ="):
            result["timings"]["total_time_sec"] = float(line.split("=", 1)[1].strip())

        elif line.startswith("server_time_sec ="):
            result["timings"]["server_time_sec"] = float(line.split("=", 1)[1].strip())

        elif line.startswith("fm_calls ="):
            result["timings"]["fm_calls"] = int(line.split("=", 1)[1].strip())

    return result


def run_query(args):
    cmd = [
        "python3",
        "tools/rare_anchor_retrieval_strict_v3.py",
        "--fm", args.fm,
        "--bwt", args.bwt,
        "--chunk-map", args.chunk_map,
        "--server-bin", args.server_bin,
    ]

    if args.query_text is not None:
        cmd += ["--query-text", args.query_text]
    elif args.query_file is not None:
        cmd += ["--query-file", args.query_file]
    else:
        return {
            "returncode": 1,
            "error": "no_query_provided",
        }

    if args.explain:
        cmd.append("--explain")

    res = subprocess.run(cmd, capture_output=True, text=True)

    parsed = parse_strict_output(res.stdout)
    parsed["returncode"] = res.returncode
    parsed["stderr"] = res.stderr

    return parsed


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
    out = run_query(args)

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(out.get("raw_stdout", ""))
        if out.get("stderr"):
            print(out["stderr"], file=sys.stderr)

    sys.exit(out.get("returncode", 1))


if __name__ == "__main__":
    main()
