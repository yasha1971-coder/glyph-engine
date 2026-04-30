# GLYPH

GLYPH is a byte-exact substring retrieval engine.

It is NOT a search engine.
- no ranking
- no fuzzy matching
- no scoring

It performs deterministic exact matches over binary data.

Performance:
- ~1.3–1.7 ms (warm)
- ~4 ms p99 (4GB shard)

## Example

CLI:

    ./glyph_cli_v2.py "error 500"

HTTP:

    POST /query

## Status

Experimental prototype.

## IP Notice

No patent clearance is claimed.

Commercial use requires independent legal review.

See:
- PATENT_RISK_AUDIT_v2.md

## License

Apache-2.0
