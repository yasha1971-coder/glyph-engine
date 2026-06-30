#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def measure_runs(path: Path, chunk_size: int = 1024 * 1024):
    n = 0
    runs = 0
    prev = None
    alphabet = set()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            for b in chunk:
                alphabet.add(b)
                if prev is None or b != prev:
                    runs += 1
                    prev = b
                n += 1

    if n == 0:
        return {
            "ok": False,
            "error": "empty file",
            "path": str(path),
        }

    return {
        "ok": True,
        "path": str(path),
        "bytes": n,
        "runs": runs,
        "r_over_n": runs / n,
        "avg_run_len": n / runs if runs else None,
        "alphabet_size": len(alphabet),
        "sha256": sha256_file(path),
    }


def classify(r_over_n: float) -> str:
    if r_over_n <= 0.001:
        return "extremely_repetitive"
    if r_over_n <= 0.01:
        return "highly_repetitive"
    if r_over_n <= 0.05:
        return "repetitive"
    if r_over_n <= 0.15:
        return "moderately_repetitive"
    return "not_run_compressible"


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure BWT run count r and r/n.")
    ap.add_argument("bwt", nargs="+", help="Path(s) to bwt.bin")
    ap.add_argument("--jsonl-out", default="", help="Optional JSONL output.")
    args = ap.parse_args()

    results = []

    for item in args.bwt:
        path = Path(item).resolve()
        if not path.exists():
            res = {
                "ok": False,
                "path": str(path),
                "error": "missing",
            }
        else:
            res = measure_runs(path)
            if res.get("ok"):
                res["classification"] = classify(res["r_over_n"])
        results.append(res)
        print(json.dumps(res, indent=2, sort_keys=True))

    if args.jsonl_out:
        out = Path(args.jsonl_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, sort_keys=True) + "\n")

    bad = [r for r in results if not r.get("ok")]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
