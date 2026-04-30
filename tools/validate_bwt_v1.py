import argparse
import os
import struct
from collections import Counter


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def read_u32(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        raise ValueError("sa.bin size is not multiple of 4")
    return list(struct.unpack("<" + "I" * (len(data) // 4), data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--sa", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--sentinel", type=int, default=0)
    ap.add_argument("--sample", type=int, default=10)
    args = ap.parse_args()

    corpus = read_bytes(args.corpus)
    sa = read_u32(args.sa)
    bwt = read_bytes(args.bwt)

    print("=" * 60)
    print(" VALIDATE BWT V1")
    print("=" * 60)
    print(f"corpus_bytes={len(corpus)}")
    print(f"sa_len={len(sa)}")
    print(f"bwt_bytes={len(bwt)}")

    if len(corpus) != len(sa):
        raise ValueError("corpus length != sa length")
    if len(corpus) != len(bwt):
        raise ValueError("corpus length != bwt length")

    # check sentinel count
    sent_count = bwt.count(args.sentinel.to_bytes(1, "little"))
    print(f"sentinel_count={sent_count}")

    # sample exact BWT relation
    bad = 0
    n = len(corpus)
    step = max(1, n // args.sample)
    checked = 0

    for i in range(0, n, step):
        s = sa[i]
        expected = args.sentinel if s == 0 else corpus[s - 1]
        got = bwt[i]
        checked += 1
        if expected != got:
            bad += 1
            print(f"mismatch at i={i}: sa={s}, expected={expected}, got={got}")

    print(f"sample_checks={checked}")
    print(f"sample_mismatches={bad}")

    # histogram sanity: BWT multiset should match corpus except one preceding sentinel substitution
    corp_hist = Counter(corpus)
    bwt_hist = Counter(bwt)

    # expected: one position where sa[i]==0 gets sentinel instead of corpus[-1]
    last_byte = corpus[-1]
    expected_hist = corp_hist.copy()
    expected_hist[last_byte] -= 1
    if expected_hist[last_byte] == 0:
        del expected_hist[last_byte]
    expected_hist[args.sentinel] += 1

    hist_ok = (bwt_hist == expected_hist)
    print(f"histogram_ok={hist_ok}")

    if bad == 0 and hist_ok:
        print("validation_ok")
    else:
        print("validation_failed")


if __name__ == "__main__":
    main()