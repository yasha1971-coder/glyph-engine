import argparse
import os
import struct
import subprocess
import tempfile


def write_request(path, ranges, topk):
    with open(path, "wb") as f:
        f.write(b"FMHREQ1\0")
        f.write(struct.pack("<I", len(ranges)))
        f.write(struct.pack("<I", topk))
        for l, r, w in ranges:
            f.write(struct.pack("<I", l))
            f.write(struct.pack("<I", r))
            f.write(struct.pack("<I", w))


def read_response(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"FMHRES1\0":
            raise ValueError(f"bad response magic: {magic!r}")
        n = struct.unpack("<I", f.read(4))[0]
        rows = []
        for _ in range(n):
            cid = struct.unpack("<I", f.read(4))[0]
            score = struct.unpack("<I", f.read(4))[0]
            rows.append((cid, score))
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--top-k", type=int, default=16)
    args = ap.parse_args()

    # synthetic ranges for smoke test
    # just verify backend runs and returns chunks
    ranges = [
        (100, 200, 1),
        (1000, 1200, 1),
        (50000, 50300, 1),
    ]

    with tempfile.TemporaryDirectory() as td:
        req = os.path.join(td, "req.bin")
        resp = os.path.join(td, "resp.bin")
        write_request(req, ranges, args.top_k)

        proc = subprocess.run(
            [args.backend, args.chunk_map],
            input=open(req, "rb").read(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        with open(resp, "wb") as f:
            f.write(proc.stdout)

        rows = read_response(resp)

    print("=" * 60)
    print(" FM CHUNK HIST SMOKE V1")
    print("=" * 60)
    print(f" ranges={ranges}")
    print(" top chunks:")
    for i, (cid, score) in enumerate(rows, 1):
        print(f"  #{i:>2}: chunk={cid} score={score}")

    print()
    print("backend_stderr:")
    print(proc.stderr.decode("utf-8", errors="replace").strip())


if __name__ == "__main__":
    main()