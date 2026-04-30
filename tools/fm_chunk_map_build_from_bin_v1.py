import argparse
import os
import struct

INVALID_CHUNK = 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sa-bin", required=True)
    ap.add_argument("--chunk-size", type=int, default=16384)
    ap.add_argument("--num-chunks", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sa_path = args.sa_bin
    sa_size_bytes = os.path.getsize(sa_path)
    if sa_size_bytes % 4 != 0:
        raise ValueError(f"sa.bin size is not divisible by 4: {sa_size_bytes}")

    sa_len = sa_size_bytes // 4
    covered_bytes = args.chunk_size * args.num_chunks

    num_valid = 0
    num_invalid = 0

    with open(sa_path, "rb") as f_in, open(args.out, "wb") as f_out:
        f_out.write(b"CHMAPV1\0")
        f_out.write(struct.pack("<Q", sa_len))
        f_out.write(struct.pack("<I", args.chunk_size))
        f_out.write(struct.pack("<I", args.num_chunks))

        for _ in range(sa_len):
            raw = f_in.read(4)
            if len(raw) != 4:
                raise RuntimeError("unexpected EOF while reading sa.bin")
            pos = struct.unpack("<I", raw)[0]

            if pos < covered_bytes:
                cid = pos // args.chunk_size
                num_valid += 1
            else:
                cid = INVALID_CHUNK
                num_invalid += 1

            f_out.write(struct.pack("<I", cid))

    print("=" * 60)
    print(" FM CHUNK MAP BUILD FROM BIN V1")
    print("=" * 60)
    print(f" sa_bin={args.sa_bin}")
    print(f" sa_len={sa_len:,}")
    print(f" chunk_size={args.chunk_size}")
    print(f" num_chunks={args.num_chunks}")
    print(f" covered_bytes={covered_bytes:,}")
    print(f" valid_suffixes={num_valid:,}")
    print(f" invalid_suffixes={num_invalid:,}")
    print(f" out={args.out}")


if __name__ == "__main__":
    main()