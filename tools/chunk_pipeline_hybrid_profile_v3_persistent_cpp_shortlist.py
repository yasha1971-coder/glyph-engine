import argparse
import pickle
import random
import struct
import subprocess
import time
import zlib


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
def load_chunks(glyph_path):
    chunks = []
    with open(glyph_path, "rb") as f:
        if f.read(6) != b"GLYPH1":
            raise ValueError("bad glyph")
        _v, ng = struct.unpack("<BH", f.read(3))
        for _ in range(ng):
            sz = struct.unpack("<I", f.read(4))[0]
            chunks.append(zlib.decompress(f.read(sz)))
    return chunks


def load_filter(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------
# FRAGMENTS
# ------------------------------------------------------------
def choose_fragment_starts(L, frag_len, nfrag, min_gap, rng):
    need = nfrag * frag_len + (nfrag - 1) * min_gap
    if L < need:
        return None
    base = rng.randint(0, L - need)
    return [base + i * (frag_len + min_gap) for i in range(nfrag)]


def build_fragments(chunk, frag_len, nfrag, min_gap, rng):
    starts = choose_fragment_starts(len(chunk), frag_len, nfrag, min_gap, rng)
    if starts is None:
        return None, None
    return [chunk[s:s+frag_len] for s in starts], starts


# ------------------------------------------------------------
# REQUEST / RESPONSE
# ------------------------------------------------------------
def write_request_to_pipe(pipe, frags, topk):
    pipe.write(b"SLREQV1\0")
    pipe.write(struct.pack("<I", len(frags)))
    pipe.write(struct.pack("<I", topk))
    for frag in frags:
        pipe.write(struct.pack("<I", len(frag)))
        pipe.write(frag)
    pipe.flush()


def read_exact(pipe, n):
    buf = b""
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            raise RuntimeError("unexpected EOF from backend")
        buf += chunk
    return buf


def read_response_from_pipe(pipe):
    magic = read_exact(pipe, 8)
    if magic != b"SLRESP1\0":
        raise ValueError(f"bad response magic: {magic!r}")

    n = struct.unpack("<I", read_exact(pipe, 4))[0]
    rows = []
    for _ in range(n):
        cid = struct.unpack("<I", read_exact(pipe, 4))[0]
        score = struct.unpack("<d", read_exact(pipe, 8))[0]
        rows.append((cid, score))
    return rows


# ------------------------------------------------------------
# EXACT RERANK
# ------------------------------------------------------------
def exact_score(frags, chunk):
    present = 0
    total = 0
    for frag in frags:
        c = chunk.count(frag)
        if c > 0:
            present += 1
            total += c
    return present, total


def rerank_exact(chunks, frags, shortlist):
    shortlist = [cid for cid in shortlist if 0 <= cid < len(chunks)]
    out = []
    for cid in shortlist:
        p, t = exact_score(frags, chunks[cid])
        out.append((cid, p, t))
    out.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return out


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--dense-bin", required=True)
    ap.add_argument("--rare-bin", required=True)
    ap.add_argument("--filter", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("=" * 60)
    print(" HYBRID PROFILE V3 (PERSISTENT C++ SHORTLIST)")
    print("=" * 60)

    filt = load_filter(args.filter)
    indexed_limit = len(filt["rare"]["chunks"])
    chunks = load_chunks(args.glyph)[:indexed_limit]

    t_shortlist = 0.0
    t_rerank = 0.0

    shortlist_hit = 0
    hit1 = 0
    total = 0

    ids = list(range(len(chunks)))
    if args.trials < len(ids):
        ids = rng.sample(ids, args.trials)

    proc = subprocess.Popen(
        [args.backend, args.dense_bin, args.rare_bin],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    try:
        for qid in ids:
            frags, _starts = build_fragments(
                chunks[qid],
                args.frag_len,
                args.nfrag,
                args.min_gap,
                random.Random(args.seed + qid),
            )
            if frags is None:
                continue

            t0 = time.perf_counter()
            write_request_to_pipe(proc.stdin, frags, args.top_k)
            rows = read_response_from_pipe(proc.stdout)
            shortlist = [cid for cid, _score in rows]
            t1 = time.perf_counter()

            if qid in shortlist:
                shortlist_hit += 1

            t2 = time.perf_counter()
            rr = rerank_exact(chunks, frags, shortlist)
            t3 = time.perf_counter()

            if rr and rr[0][0] == qid:
                hit1 += 1

            t_shortlist += (t1 - t0)
            t_rerank += (t3 - t2)
            total += 1

    finally:
        if proc.stdin:
            proc.stdin.close()
        stderr_text = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        proc.wait()

    if total == 0:
        raise ValueError("no executed trials")

    print()
    print("RESULT:")
    print(f"  indexed_chunks       = {len(chunks)}")
    print(f"  trials               = {total}")
    print(f"  shortlist_hit@{args.top_k:<2} = {shortlist_hit / total:.4f}")
    print(f"  hit@1                = {hit1 / total:.4f}")
    print(f"  avg_shortlist_time   = {t_shortlist / total:.6f} sec")
    print(f"  avg_rerank_time      = {t_rerank / total:.6f} sec")
    print(f"  avg_total_time       = {(t_shortlist + t_rerank) / total:.6f} sec")

    print()
    print("backend_stderr:")
    print(stderr_text.strip())


if __name__ == "__main__":
    main()