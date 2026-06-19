#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def safe_text(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(description="Create GLYPH_EVIDENCE_CASE_V1 from audit artifact V0.")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--context-bytes", type=int, default=24)
    args = ap.parse_args()

    artifact_path = Path(args.artifact).resolve()
    out_path = Path(args.output).resolve()

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    if artifact.get("artifact_version") != "GLYPH_AUDIT_ARTIFACT_V0":
        raise SystemExit("unsupported artifact_version")

    corpus_path = resolve_path(artifact["corpus"]["path"])
    corpus = corpus_path.read_bytes()

    query_hex = artifact["query"]["hex"]
    query_bytes = bytes.fromhex(query_hex)
    query_text = safe_text(query_bytes)

    records = []
    for offset in artifact["result"].get("offsets", []):
        start = max(0, offset - args.context_bytes)
        end = min(len(corpus), offset + len(query_bytes) + args.context_bytes)
        snippet = corpus[start:end]

        records.append({
            "offset": offset,
            "match_hex": query_hex,
            "match_text": query_text,
            "snippet_start": start,
            "snippet_end": end,
            "snippet_text": safe_text(snippet),
            "byte_check": corpus[offset:offset + len(query_bytes)] == query_bytes,
        })

    case = {
        "case_version": "GLYPH_EVIDENCE_CASE_V1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source_artifact": str(artifact_path.relative_to(ROOT) if artifact_path.is_relative_to(ROOT) else artifact_path),
        "corpus": artifact["corpus"],
        "query": {
            "hex": query_hex,
            "text": query_text,
            "sha256": artifact["query"]["sha256"],
            "length_bytes": artifact["query"]["length_bytes"],
        },
        "result_summary": {
            "match_count": artifact["result"]["match_count"],
            "fm_interval": artifact["result"]["fm_interval"],
            "offset_mode": artifact["result"]["offset_mode"],
            "offsets": artifact["result"]["offsets"],
        },
        "evidence_records": records,
        "boundary": {
            "claim": "Human-readable evidence case derived from a reproducible audit artifact.",
            "not_claimed": [
                "legal proof",
                "zero-knowledge proof",
                "cryptographic completeness proof",
                "cryptographic non-membership proof"
            ]
        }
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[evidence-case-v1] wrote {out_path}")
    print(f"[evidence-case-v1] records={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
