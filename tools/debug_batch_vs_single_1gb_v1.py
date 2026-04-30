import argparse
import struct
import subprocess


def load_chunks(corpus_path: str, chunk_size: int = 16384):
    chunks = []
    with open(corpus_path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            chunks.append(block)
    return chunks


def run_single(single_bin, fm, bwt, hex_pat):
    res = subprocess.run(
        [str(single_bin), str(fm), str(bwt), hex_pat],
        capture_output=True,
        text=True,
    )
    return res.returncode, res.stdout, res.stderr


def run_batch(batch_bin, fm, bwt, hex_patterns):
    proc = subprocess.Popen(
        [str(batch_bin), str(fm), str(bwt)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = "\n".join(hex_patterns) + "\n"
    stdout, stderr = proc.communicate(payload)
    return proc.returncode, stdout, stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--single-bin", required=True)
    ap.add_argument("--batch-bin", required=True)
    ap.add_argument("--qid", type=int, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    args = ap.parse_args()

    chunks = load_chunks(args.corpus)
    chunk = chunks[args.qid]
    frag = chunk[args.start:args.start + args.frag_len]
    if len(frag) != args.frag_len:
        raise ValueError("fragment too short")

    hex_pat = frag.hex()

    print("============================================")
    print(" DEBUG BATCH VS SINGLE 1GB V1")
    print("============================================")
    print(f"qid={args.qid}")
    print(f"start={args.start}")
    print(f"frag_len={args.frag_len}")
    print(f"hex={hex_pat}")

    rc1, out1, err1 = run_single(args.single_bin, args.fm, args.bwt, hex_pat)
    print("\n[SINGLE]")
    print("returncode:", rc1)
    print("stdout:")
    print(out1)
    print("stderr:")
    print(err1)

    rc2, out2, err2 = run_batch(args.batch_bin, args.fm, args.bwt, [hex_pat])
    print("\n[BATCH]")
    print("returncode:", rc2)
    print("stdout:")
    print(out2)
    print("stderr:")
    print(err2)


if __name__ == "__main__":
    main()