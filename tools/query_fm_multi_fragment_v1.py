import argparse
import subprocess


def load_chunks(corpus_path, chunk_size=16384):
    chunks = []
    with open(corpus_path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            chunks.append(block)
    return chunks


def query_one(query_bin, fm, bwt, frag: bytes):
    pattern_hex = frag.hex()
    cmd = [query_bin, fm, bwt, pattern_hex]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr or res.stdout)

    l = r = cnt = None
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("interval:"):
            # interval: [88036406, 88067605)
            part = line.split("[", 1)[1].split(")", 1)[0]
            left, right = part.split(",", 1)
            l = int(left.strip())
            r = int(right.strip())
        elif line.startswith("count:"):
            cnt = int(line.split(":", 1)[1].strip())

    return l, r, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--query-bin", required=True)
    ap.add_argument("--chunk-id", type=int, required=True)
    ap.add_argument("--starts", required=True, help="comma-separated starts")
    ap.add_argument("--frag-len", type=int, default=48)
    args = ap.parse_args()

    chunks = load_chunks(args.corpus)
    chunk = chunks[args.chunk_id]
    starts = [int(x) for x in args.starts.split(",")]

    print("=" * 60)
    print(" QUERY FM MULTI FRAGMENT V1")
    print("=" * 60)
    print(f"chunk_id={args.chunk_id}")
    print(f"starts={starts}")
    print(f"frag_len={args.frag_len}")

    for i, s in enumerate(starts, 1):
        frag = chunk[s:s + args.frag_len]
        if len(frag) != args.frag_len:
            raise ValueError(f"fragment {i} too short")

        l, r, cnt = query_one(args.query_bin, args.fm, args.bwt, frag)
        print(f"frag{i}: start={s} interval=[{l}, {r}) count={cnt}")


if __name__ == "__main__":
    main()