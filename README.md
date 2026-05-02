python3 - << 'PY'
from pathlib import Path

Path("README.md").write_text("""# GLYPH

GLYPH is a byte-exact substring retrieval engine over raw bytes.

It is designed for high-speed exact matching without tokenization or scoring.

It is NOT a search engine:
- no ranking
- no fuzzy matching
- no scoring

It performs deterministic exact matches at scale.

---

## ⚡ Try it in 10 seconds

    git clone https://github.com/yasha1971-coder/glyph-engine
    cd glyph-engine
    ./examples/mini/build_mini.sh

Expected output:

    count:    2

This runs a full pipeline:

- builds suffix array (SA)
- builds BWT
- builds FM-index
- runs a real query

No large datasets required.

---

## Quick Start (full system)

GLYPH currently expects prepared index artifacts and a running local HTTP server.

Note:
- run.sh expects local prepared demo artifacts
- large corpus/index artifacts are not included

Check service:

    curl http://127.0.0.1:18080/health

Query prepared demo data:

    ./glyph_cli.py --hex "$(xxd -p -c 999999 /tmp/query_41905.bin)"

Expected:
- JSON response with exact byte-match shortlist

---

## Core guarantees

- deterministic results
- no ranking
- no fuzzy matching

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

- ~1.3–1.7 ms (warm)
- ~4 ms p99 (4GB shard)
- mmap-based index

---

## Status

Experimental prototype.

---

## Patent Status

GLYPH uses public algorithms and provides no patent-safety guarantee.

No patent clearance is claimed.
Commercial use requires independent legal review.

See:
- PATENT_RISK_AUDIT_v2.md

---

## License

Apache-2.0
""")

print("README OK")
PY