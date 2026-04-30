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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--query-bin", required=True)
    ap.add_argument("--chunk-id", type=int, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    args = ap.parse_args()

    chunks = load_chunks(args.corpus)
    chunk = chunks[args.chunk_id]

    frag = chunk[args.start:args.start + args.frag_len]
    if len(frag) != args.frag_len:
        raise ValueError("fragment shorter than frag_len")

    pattern_hex = frag.hex()

    print("=" * 60)
    print(" QUERY FM FROM CORPUS V1")
    print("=" * 60)
    print(f"chunk_id={args.chunk_id}")
    print(f"start={args.start}")
    print(f"frag_len={args.frag_len}")
    print(f"pattern_hex_prefix={pattern_hex[:64]}...")

    cmd = [
        args.query_bin,
        args.fm,
        args.bwt,
        pattern_hex,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    print("\nRETURN CODE:", res.returncode)
    print("\nSTDOUT:")
    print(res.stdout)
    if res.stderr:
        print("\nSTDERR:")
        print(res.stderr)


if __name__ == "__main__":
    main()