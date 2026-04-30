import argparse
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--rare", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.dense, "rb") as f:
        dense = pickle.load(f)

    with open(args.rare, "rb") as f:
        rare = pickle.load(f)

    hybrid = {
        "version": "chunk_filter_hybrid_v1",
        "dense": dense,
        "rare": rare,
    }

    with open(args.out, "wb") as f:
        pickle.dump(hybrid, f, protocol=pickle.HIGHEST_PROTOCOL)

    dense_inv_keys = len(dense.get("inv", {}))
    rare_chunks = len(rare.get("chunks", []))

    print("=" * 60)
    print("  HYBRID FILTER BUILT")
    print("=" * 60)
    print(f"  dense_version={dense.get('version')}")
    print(f"  dense_inv_keys={dense_inv_keys}")
    print(f"  rare_version={rare.get('version')}")
    print(f"  rare_chunks={rare_chunks}")
    print(f"  saved={args.out}")


if __name__ == "__main__":
    main()