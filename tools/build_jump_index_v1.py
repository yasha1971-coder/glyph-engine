import argparse
import os
import numpy as np


UINT32_MAX = np.uint32(0xFFFFFFFF)


def build_isa(sa_path: str, isa_path: str, chunk_size: int = 5_000_000):
    sa = np.memmap(sa_path, dtype=np.uint32, mode="r")
    n = len(sa)

    print("=" * 60)
    print(" BUILD ISA")
    print("=" * 60)
    print(f"sa_path={sa_path}")
    print(f"n={n:,}")
    print(f"isa_path={isa_path}")

    isa = np.memmap(isa_path, dtype=np.uint32, mode="w+", shape=(n,))

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        idx = np.asarray(sa[start:end], dtype=np.uint32)
        vals = np.arange(start, end, dtype=np.uint32)
        isa[idx] = vals
        if start == 0 or end == n or ((start // chunk_size) % 10 == 0):
            print(f"  ISA progress: {end:,}/{n:,}")

    isa.flush()
    print("ISA done")
    return n


def build_jump(sa_path: str, isa_path: str, jump_path: str, step: int, chunk_size: int = 5_000_000):
    sa = np.memmap(sa_path, dtype=np.uint32, mode="r")
    isa = np.memmap(isa_path, dtype=np.uint32, mode="r")
    n = len(sa)

    if len(isa) != n:
        raise ValueError(f"ISA length mismatch: sa={n} isa={len(isa)}")

    print("=" * 60)
    print(" BUILD JUMP")
    print("=" * 60)
    print(f"sa_path={sa_path}")
    print(f"isa_path={isa_path}")
    print(f"jump_path={jump_path}")
    print(f"step={step}")
    print(f"n={n:,}")

    jump = np.memmap(jump_path, dtype=np.uint32, mode="w+", shape=(n,))

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        sa_chunk = np.asarray(sa[start:end], dtype=np.int64)
        out = np.full(end - start, UINT32_MAX, dtype=np.uint32)

        valid = (sa_chunk + step) < n
        if np.any(valid):
            tgt_pos = sa_chunk[valid] + step
            out[valid] = np.asarray(isa[tgt_pos], dtype=np.uint32)

        jump[start:end] = out

        if start == 0 or end == n or ((start // chunk_size) % 10 == 0):
            print(f"  JUMP progress: {end:,}/{n:,}")

    jump.flush()
    print("JUMP done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sa-bin", required=True, help="raw uint32 SA binary")
    ap.add_argument("--step", type=int, default=176)
    ap.add_argument("--jump-out", required=True, help="raw uint32 jump array")
    ap.add_argument("--isa-out", default=None, help="build ISA here if --isa-in not given")
    ap.add_argument("--isa-in", default=None, help="reuse existing ISA")
    ap.add_argument("--chunk-size", type=int, default=5_000_000)
    args = ap.parse_args()

    if not os.path.exists(args.sa_bin):
        raise FileNotFoundError(args.sa_bin)

    if args.isa_in is None and args.isa_out is None:
        raise ValueError("provide either --isa-in or --isa-out")

    isa_path = args.isa_in

    if isa_path is None:
        isa_path = args.isa_out
        build_isa(args.sa_bin, isa_path, chunk_size=args.chunk_size)
    else:
        if not os.path.exists(isa_path):
            raise FileNotFoundError(isa_path)

    build_jump(
        sa_path=args.sa_bin,
        isa_path=isa_path,
        jump_path=args.jump_out,
        step=args.step,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()