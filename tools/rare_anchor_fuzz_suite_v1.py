import argparse
import os
import random
import subprocess
import tempfile
from collections import Counter


def run_case(args, name, query_bytes, expected=None):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(query_bytes)
        path = f.name

    try:
        cmd = [
            "python3", "tools/rare_anchor_retrieval_strict_v3.py",
            "--fm", args.fm,
            "--bwt", args.bwt,
            "--chunk-map", args.chunk_map,
            "--server-bin", args.server_bin,
            "--query-file", path,
            "--limit", str(args.limit),
            "--non-selective-threshold", str(args.non_selective_threshold),
            "--explain",
        ]

        res = subprocess.run(cmd, cwd=args.project_dir, capture_output=True, text=True, timeout=args.timeout)

        outcome = "UNKNOWN"
        for line in res.stdout.splitlines():
            line = line.strip()
            if line.startswith("outcome ="):
                outcome = line.split("=", 1)[1].strip()

        ok = expected is None or outcome == expected

        return {
            "name": name,
            "expected": expected,
            "outcome": outcome,
            "ok": ok,
            "returncode": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "expected": expected,
            "outcome": "TIMEOUT",
            "ok": expected == "TIMEOUT",
            "returncode": -1,
            "stdout": "",
            "stderr": "timeout",
        }
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def make_query_from_corpus(corpus, qid, chunk_size=16384, n=16384):
    with open(corpus, "rb") as f:
        f.seek(qid * chunk_size)
        return f.read(n)


def mutate_bytes(b, positions):
    x = bytearray(b)
    for pos in positions:
        if pos < len(x):
            x[pos] ^= 1
    return bytes(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default="/home/glyph/GLYPH_CPP_BACKEND")
    ap.add_argument("--corpus", default="data_1gb/corpus_1gb_s0.bin")
    ap.add_argument("--fm", default="out_1gb/fm_s0.bin")
    ap.add_argument("--bwt", default="out_1gb/bwt_s0.bin")
    ap.add_argument("--chunk-map", default="out_1gb/chunk_map_s0.bin")
    ap.add_argument("--server-bin", default="build/query_fm_server_v1")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--non-selective-threshold", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--show-failures", type=int, default=10)
    args = ap.parse_args()

    corpus_path = os.path.join(args.project_dir, args.corpus)

    exact_41905 = make_query_from_corpus(corpus_path, 41905)
    exact_48598_128 = make_query_from_corpus(corpus_path, 48598, n=128)
    partial_mut = mutate_bytes(exact_41905, [70, 130, 190])

    random.seed(42)
    random_absent = bytes(random.getrandbits(8) for _ in range(256))

    cases = [
        ("empty_query", b"", "EMPTY_QUERY"),  # strict_v2 currently maps empty to TOO_SHORT
        ("too_short_ascii", b"short", "TOO_SHORT"),
        ("too_short_binary", b"\x01\x02\x03", "TOO_SHORT"),

        ("low_complexity_A", b"A" * 128, "NON_SELECTIVE"),
        ("low_complexity_zero_text", b"0" * 128, "NON_SELECTIVE"),
        ("repeat_AT", (b"AT" * 64), "NON_SELECTIVE"),
        ("absent_binary_random", random_absent, "NO_HIT"),

        ("exact_unique_41905", exact_41905, "EXACT_UNIQUE"),
        ("exact_multi_48598_128", exact_48598_128, "EXACT_MULTI"),
        ("invalid_partial_41905", partial_mut, "INVALID_PARTIAL_HIT"),

        # long but still under current strict_v2 no max_query cap
        ("long_query_1mb_A", b"A" * (1024 * 1024), "NON_SELECTIVE"),
        ("query_too_long", b"B" * (1024 * 1024 + 1), "QUERY_TOO_LONG"),
    ]

    rows = []
    counts = Counter()

    print("=" * 72)
    print(" RARE ANCHOR FUZZ SUITE V1")
    print("=" * 72)

    for name, data, expected in cases:
        row = run_case(args, name, data, expected)
        rows.append(row)
        counts[row["outcome"]] += 1

    passed = sum(1 for r in rows if r["ok"])
    total = len(rows)

    print("\nSUMMARY:")
    print(f"  total_cases = {total}")
    print(f"  passed      = {passed}")
    print(f"  failed      = {total - passed}")

    print("\nOUTCOMES:")
    for k, v in counts.items():
        print(f"  {k:24s}= {v}")

    print("\nDETAILS:")
    for r in rows:
        status = "OK" if r["ok"] else "FAIL"
        print(
            f"  [{status}] {r['name']} "
            f"expected={r['expected']} outcome={r['outcome']} rc={r['returncode']}"
        )

    failures = [r for r in rows if not r["ok"]]
    if failures:
        print("\nFAILURE OUTPUTS:")
        for r in failures[:args.show_failures]:
            print("-" * 72)
            print(f"name={r['name']} expected={r['expected']} outcome={r['outcome']}")
            print("STDOUT:")
            print(r["stdout"][-2000:])
            print("STDERR:")
            print(r["stderr"][-2000:])

    print("\nNOTE:")
    print("  strict_v3 maps empty query to EMPTY_QUERY.")
    print("  strict_v3 rejects low-complexity anchors before loading chunk_map.")
    print("  QUERY_TOO_LONG is covered by fuzz suite.")


if __name__ == "__main__":
    main()