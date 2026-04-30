import argparse
import math
import pickle
from typing import Any, Dict, List

import numpy as np
import torch
import maxsim_cpu
from transformers import AutoTokenizer, AutoModel


# ============================================================
# LOADERS
# ============================================================
def load_chunks(corpus_path: str, chunk_size: int = 16384) -> List[bytes]:
    chunks = []
    with open(corpus_path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            chunks.append(block)
    return chunks


def normalize_rows(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "results" in obj and isinstance(obj["results"], list):
            return obj["results"]
        if "queries" in obj and isinstance(obj["queries"], list):
            return obj["queries"]
        if all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())
    raise ValueError("unsupported results format")


def shortlist_from_any(vals: Any, top_k: int) -> List[int]:
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

    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup[:top_k]


def get_shortlist(row: Dict[str, Any], top_k: int) -> List[int]:
    for key in ("shortlist", "ranked", "top", "topk", "pair_top", "fusion_top", "candidates"):
        if key in row:
            vals = shortlist_from_any(row[key], top_k)
            if vals:
                return vals

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


def get_qid(row: Dict[str, Any]) -> int:
    for k in ("qid", "query_chunk", "q_chunk", "chunk_id"):
        if k in row:
            return int(row[k])
    raise KeyError("no qid/query_chunk field")


def get_starts(row: Dict[str, Any]) -> List[int]:
    if "starts" not in row:
        raise KeyError("no starts field")
    return [int(x) for x in row["starts"]]


def build_query(chunk: bytes, starts: List[int], frag_len: int = 48) -> bytes:
    return b"".join(chunk[s:s + frag_len] for s in starts if s + frag_len <= len(chunk))


# ============================================================
# EMBEDDING
# ============================================================
def decode_bytes(x: bytes) -> str:
    return x.decode("utf-8", errors="ignore")


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    x = x / norms
    return x.astype(np.float32, copy=False)


def sanitize_token_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected 2D token matrix, got shape={x.shape}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # drop all-zero rows
    if x.shape[0] > 0:
        mask = np.any(np.abs(x) > 0, axis=1)
        x = x[mask]
    if x.shape[0] == 0:
        # keep one tiny row so maxsim has valid input
        x = np.zeros((1, x.shape[1]), dtype=np.float32)
    x = l2_normalize_rows(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return x


@torch.inference_mode()
def embed_texts_tokenwise(
    texts: List[str],
    tokenizer,
    model,
    device: str = "cpu",
    max_length: int = 256,
) -> List[np.ndarray]:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(**batch)
    hidden = outputs.last_hidden_state  # [B, T, D]
    attn = batch["attention_mask"]      # [B, T]

    out = []
    for i in range(hidden.size(0)):
        mask = attn[i].bool()
        vecs = hidden[i][mask].detach().cpu().to(torch.float32).numpy()
        vecs = sanitize_token_matrix(vecs)
        out.append(vecs)
    return out


def finite_score(x: float) -> float:
    if not math.isfinite(x):
        return -1e30
    return float(x)


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-results", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--limit-queries", type=int, default=20)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    print("=" * 60)
    print(" LATE INTERACTION REAL FEASIBILITY V2")
    print("=" * 60)
    print(f"model={args.model_name}")
    print(f"top_k={args.top_k}")
    print(f"max_length={args.max_length}")
    print(f"limit_queries={args.limit_queries}")

    with open(args.gap_results, "rb") as f:
        rows = normalize_rows(pickle.load(f))

    chunks = load_chunks(args.corpus)

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()
    model.to(device)

    hit1 = 0
    hit4 = 0
    hit8 = 0
    hit16 = 0
    total = 0
    examples = []

    for row in rows[:args.limit_queries]:
        qid = get_qid(row)
        starts = get_starts(row)
        shortlist = get_shortlist(row, args.top_k)
        if not shortlist:
            continue

        query_bytes = build_query(chunks[qid], starts, args.frag_len)
        query_text = decode_bytes(query_bytes)
        cand_texts = [decode_bytes(chunks[cid]) for cid in shortlist]

        q_emb = embed_texts_tokenwise(
            [query_text],
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
        )[0]

        d_embs = embed_texts_tokenwise(
            cand_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
        )

        scores = maxsim_cpu.maxsim_scores_variable(q_emb, d_embs)
        ranked_pairs = [(cid, finite_score(float(score))) for cid, score in zip(shortlist, scores)]
        ranked_pairs.sort(key=lambda x: (-x[1], x[0]))

        rank = None
        for i, (cid, _score) in enumerate(ranked_pairs):
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
                "rank": rank,
                "top5": ranked_pairs[:5],
            })

        print(f"[{total}/{args.limit_queries}] qid={qid} rank={rank}")

    if total == 0:
        raise ValueError("no usable rows")

    print("\nRESULTS:")
    print(f"  hit@1  = {hit1 / total:.2%}")
    print(f"  hit@4  = {hit4 / total:.2%}")
    print(f"  hit@8  = {hit8 / total:.2%}")
    print(f"  hit@16 = {hit16 / total:.2%}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(ex)


if __name__ == "__main__":
    main()