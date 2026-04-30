# GLYPH PATENT / IP RISK AUDIT v2

Status: PRE-PUBLICATION CHECK

Decision:
- GitHub public release: DONE (experimental basis)
- encode.su post: PENDING
- commercial claims: HOLD

1. Core

GLYPH v1.2:
- SA32u
- BWT
- FM
- v5 retrieval
- no chunk_map
- segmented router
- HTTP API
- CLI

2. Own code

- glyph_live_retrieve_v5.py
- glyph_segmented_live_v3.py
- glyph_http_server_v2.py
- glyph_cli_v2.py

Status:
- own implementation
- no patent-safety claim

3. Dependencies

libsais:
- Apache-2.0
- verify LICENSE

xxhash:
- BSD
- verify LICENSE

FastAPI / uvicorn:
- verify licenses

4. Algorithms

- BWT (1992) → expired
- FM-index (~1999) → expired

Conclusion:
- core algorithm risk LOW
- implementation risk UNKNOWN (needs review)

5. Rules

Allowed:
- experimental
- exact search
- prototype

Forbidden:
- patent-safe
- legally cleared
- production-ready commercial

6. Gate

Before GitHub:
- verify licenses
- add LICENSE
- add THIRD_PARTY_NOTICES.md
