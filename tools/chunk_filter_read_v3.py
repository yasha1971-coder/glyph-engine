import argparse
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True)
    args = ap.parse_args()

    with open(args.filter, "rb") as f:
        obj = pickle.load(f)

    print("=" * 60)
    print("  CHUNK FILTER READ V3")
    print("=" * 60)
    print(f"  version={obj['version']}")
    print(f"  sig_len={obj['sig_len']}")
    print(f"  stride={obj['stride']}")
    print(f"  max_df={obj['max_df']}")
    print(f"  top_per_chunk={obj['top_per_chunk']}")
    print(f"  chunks={len(obj['chunks'])}")
    print(f"  global_df_size={len(obj['global_df'])}")

    if obj["chunks"]:
        row = obj["chunks"][0]
        print("")
        print("  first_chunk:")
        print(f"    chunk_id={row['chunk_id']}")
        print(f"    num_unique_signatures={row['num_unique_signatures']}")
        print(f"    num_rare_candidates={row['num_rare_candidates']}")
        print(f"    rare_anchors[:10]={row['rare_anchors'][:10]}")


if __name__ == "__main__":
    main()