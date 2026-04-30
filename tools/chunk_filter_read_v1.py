import argparse
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True)
    args = ap.parse_args()

    with open(args.filter, "rb") as f:
        obj = pickle.load(f)

    print("=" * 60)
    print("  CHUNK FILTER READ V1")
    print("=" * 60)
    print(f"  version={obj['version']}")
    print(f"  sig_len={obj['sig_len']}")
    print(f"  stride={obj['stride']}")
    print(f"  top_per_chunk={obj['top_per_chunk']}")
    print(f"  chunks={len(obj['chunks'])}")

    if obj["chunks"]:
        first = obj["chunks"][0]
        print("")
        print("  first_chunk:")
        print(f"    chunk_id={first['chunk_id']}")
        print(f"    raw_len={first['raw_len']}")
        print(f"    num_unique_signatures={first['num_unique_signatures']}")
        print(f"    top_signatures[:10]={first['top_signatures'][:10]}")


if __name__ == "__main__":
    main()