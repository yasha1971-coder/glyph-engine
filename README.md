# GLYPH

GLYPH is a byte-exact substring retrieval engine over raw bytes.

It is designed for high-speed exact matching without tokenization or scoring.

It is NOT a search engine:
- no ranking
- no fuzzy matching
- no scoring

It performs deterministic exact matches at scale.

---

## What problem it solves

Most systems trade accuracy for flexibility:

- grep → scans (slow at scale)
- Elasticsearch → ranks (approximate)
- vector search → approximate similarity

GLYPH does the opposite:

- exact byte matches
- no interpretation
- deterministic results

---

## When to use

- large-scale log search
- binary corpus lookup
- forensic / debugging analysis
- RAG pre-filtering (exact stage before embedding)

---

## Performance

- ~1.3–1.7 ms (warm, segmented)
- ~4 ms p99 (4GB shard)
- segmented parallel fan-out

---

## Example

CLI:

    ./glyph_cli_v2.py "error 500"

HTTP:

    POST /query

---

## Status

Experimental prototype.

---

## IP Notice

No patent clearance is claimed.  
Commercial use requires independent legal review.

See:
- PATENT_RISK_AUDIT_v2.md

---

## License

Apache-2.0
---

## Quick Start

Minimal example:

    echo "error 500 test" > test.txt
    ./glyph_cli_v2.py "error"

Expected:
- exact byte match positions returned

Core guarantees:
- deterministic results
- no ranking
- no fuzzy matching

See:
- PRODUCT_BASELINE_v1.md

