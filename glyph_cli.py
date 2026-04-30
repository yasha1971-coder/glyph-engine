#!/usr/bin/env python3
import argparse
import subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--hex", action="store_true")
    args = ap.parse_args()

    if args.hex:
        hex_query = args.query
    else:
        hex_query = args.query.encode("utf-8").hex()

    cmd = [
        "./glyph_segmented_live_v3.py",
        "--config", "shards_8gb_demo.json",
        "--server-bin", "build/query_fm_server_v1"
    ]

    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    while True:
        line = p.stderr.readline().strip()
        if line.startswith("READY"):
            break

    p.stdin.write("HEX " + hex_query + "\n")
    p.stdin.flush()

    result = p.stdout.readline()
    print(result.strip())

    p.kill()

if __name__ == "__main__":
    main()
