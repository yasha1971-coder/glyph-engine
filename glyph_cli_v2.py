#!/usr/bin/env python3
import argparse
import requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--hex", action="store_true")
    args = ap.parse_args()

    if args.hex:
        hex_query = args.query
    else:
        hex_query = args.query.encode("utf-8").hex()

    r = requests.post(
        "http://127.0.0.1:18080/query",
        json={"hex": hex_query},
        timeout=5
    )

    print(r.json())

if __name__ == "__main__":
    main()
