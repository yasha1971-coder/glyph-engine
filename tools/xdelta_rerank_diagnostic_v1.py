import argparse
import os
import pickle
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_xdelta(query_bytes, cand_bytes):
    """
    Returns:
      ("OK", delta_size) on success
      ("ERR", short_error_message) on failure
    """
    src_path = None
    tgt_path = None
    out_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            src_path = f1.name
            f1.write(query_bytes)
            f1.flush()

        with tempfile.NamedTemporaryFile(delete=False) as f2:
            tgt_path = f2.name
            f2.write(cand_bytes)
            f2.flush()

        out_fd, out_path = tempfile.mkstemp(prefix="xdelta_out_", suffix=".vcdiff")
        os.close(out_fd)

        # xdelta3 prefers to create/overwrite itself
        if os.path.exists(out_path):
            os.unlink(out_path)

        cmd = ["xdelta3", "-f", "-e", "-s", src_path, tgt_path, out_path]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="ignore").strip()
            if not err:
                err = f"xdelta3 failed with code {result.returncode}"
            return ("ERR", err[:300])

        size = Path(out_path).stat().st_size
        return ("OK", size)

    except Exception as e:
        return ("ERR", f"{type(e).__name__}: {e}")

    finally:
        for p in (src_path, tgt_path, out_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


def load_chunks(corpus_path, chunk_size=16384):
    chunks = []
    with open(corpus_path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            chunks.append(block)
    return chunks


def build_query(chunk, starts, frag_len=48):
    parts = [chunk[s:s + frag_len] for s in starts]
    return b"".join(parts)


def shortlist_from_any(vals, top_k):
    out = []
    if not isinstance(vals, (list, tuple)):
        return out

    for item in vals:
        cid = None
        if isinstance(item, int):
            cid = item
        elif isinstance(item, (list, tuple)) and len(item) >= 1:
            cid = int(item[0])
        elif isinstance(item, dict):
            for k in ("cid", "chunk", "chunk_id", "id"):
                if k in item:
                    cid = int(item[k])
                    break

        if cid is not None:
            out.append(cid)

        if len(out) >= top_k:
            break

    dedup = []
    seen = set()
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup[:top_k]


def get_shortlist(row, top_k):
    for key in ("shortlist", "ranked", "top", "topk", "pair_top", "fusion_top", "candidates"):
        if key in row:
            sl = shortlist_from_any(row[key], top_k)
            if sl:
                return sl

    pair_keys = [k for k in row.keys() if k.startswith("pair") and isinstance(row[k], list)]
    if pair_keys:
        merged = []
        seen = set()
        for pk in sorted(pair_keys):
            for cid in shortlist_from_any(row[pk], top_k):
                if cid not in seen:
                    seen.add(cid)
                    merged.append(cid)
                if len(merged) >= top_k:
                    return merged
        return merged[:top_k]

    return []


def get_qid(row):
    for k in ("qid", "query_chunk", "q_chunk", "chunk_id"):
        if k in row:
            return int(row[k])
    raise KeyError("no qid/query_chunk field in row")


def get_starts(row):
    if "starts" not in row:
        raise KeyError("no starts field in row")
    return [int(x) for x in row["starts"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-results", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    with open(args.gap_results, "rb") as f:
        results = pickle.load(f)

    if isinstance(results, dict):
        if "results" in results:
            results = results["results"]
        elif "queries" in results:
            results = results["queries"]
        else:
            results = list(results.values())

    chunks = load_chunks(args.corpus)

    hit1 = 0
    hit4 = 0
    hit8 = 0
    hit16 = 0
    total = 0
    examples = []
    err_examples = []

    print("=" * 60)
    print(" XDELTA RERANK DIAGNOSTIC V1")
    print("=" * 60)

    for row in results:
        qid = get_qid(row)
        starts = get_starts(row)
        shortlist = get_shortlist(row, args.top_k)

        if not shortlist:
            continue

        query = build_query(chunks[qid], starts)
        scores = []

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {}
            for cid in shortlist:
                futures[ex.submit(run_xdelta, query, chunks[cid])] = cid

            for fut in as_completed(futures):
                cid = futures[fut]
                status, payload = fut.result()
                if status == "OK":
                    score = payload
                else:
                    score = 10**9
                    if len(err_examples) < 5:
                        err_examples.append((qid, cid, payload))
                scores.append((cid, score))

        scores.sort(key=lambda x: x[1])
        ranked = [cid for cid, _ in scores]

        rank = None
        for i, cid in enumerate(ranked):
            if cid == qid:
                rank = i + 1
                break

        total += 1

        if rank == 1:
            hit1 += 1
        if rank is not None and rank <= 4:
            hit4 += 1
        if rank is not None and rank <= 8:
            hit8 += 1
        if rank is not None and rank <= 16:
            hit16 += 1

        if len(examples) < args.show:
            examples.append({
                "qid": qid,
                "starts": starts,
                "rank": rank,
                "top5": scores[:5],
            })

    if total == 0:
        raise ValueError("no usable rows found in gap_results")

    print("\nRESULTS:")
    print(f"  xdelta_hit@1  = {hit1 / total:.2%}")
    print(f"  xdelta_hit@4  = {hit4 / total:.2%}")
    print(f"  xdelta_hit@8  = {hit8 / total:.2%}")
    print(f"  xdelta_hit@16 = {hit16 / total:.2%}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(f"  query_chunk={ex['qid']} starts={ex['starts']} rank={ex['rank']}")
        print(f"    top5={ex['top5']}")

    if err_examples:
        print("\nXDELTA ERRORS (sample):")
        for qid, cid, msg in err_examples:
            print(f"  qid={qid} cid={cid} err={msg}")


if __name__ == "__main__":
    main()