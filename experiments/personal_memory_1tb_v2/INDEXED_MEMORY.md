# Verified content retrieval — bounded session pilot

Implemented `indexed_memory.ContentIndex` and optional
`llm_mediator.mediator.execute_indexed`. The existing GUI and linear evidence
bridge are unchanged. This is not a deployed GUI feature or a semantic model.

The host constructs pinned `ArchiveView` instances and explicitly grants
`(receipt_sha256, path)` pairs. Only granted files are read into an in-memory
SQLite FTS5 trigram index. Queries are case-sensitive exact UTF-8 substrings,
3+ Unicode characters, at most 256 UTF-8 bytes. Operators are literal, not SQL
or an FTS expression supplied by the user. No stemming, OCR, PDF extraction,
audio transcription, embeddings, or language translation is claimed.

Each returned match is independently re-read with the existing bounded decoder
and SHA-256 verification, then bound to its version, path, digest and original
byte coordinates. A failed candidate aborts the response without partial
evidence. Quotes remain untrusted document content, not instructions or proof
of factual truth. The model has no mutation rights.

`NO_MATCH_IN_INDEXED_SCOPE` concerns only the indexed pinned versions, not all
personal data and not current disk health. Unsupported codecs, binary text,
file-size and source budgets produce explicit skipped entries and incomplete
coverage. Empty grants never imply proven absence. A changed current grant set
rejects the query: the host must rebuild (including on revocation/deletion or
new-version selection). No automatic latest-version resolution is implemented.

This deliberately has no persistent index-file import path or database sidecar
that could silently be stale or modified. Process memory and the host are
trusted. No claim is made against malicious code inside the same process.
Closing the session releases the database; this is not secure memory erasure,
and OS swap/core dumps are outside this component's protections.

Default source budget: 32 MiB; maximum 64 MiB, 10,000 granted files, SQLite main
database 16,384 pages x 4096 bytes. SQLite pages do NOT bound total RSS: Python
objects, decoded files, FTS working memory and archive manifests are additional.
Existing ArchiveView receipt/file/decompression limits still apply. Deadlines
are cooperative SQL/procedural checks, not hard cancellation of blocking I/O or
decompression. Worker-process isolation and persistent scale indexing remain
necessary before untrusted, large-scale production deployment.

Reproduce targeted tests from repository root:

```sh
python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -p test_indexed_memory.py -v
python3 -m unittest discover -s experiments/personal_memory_1tb_v2/llm_mediator -v
```

Tests create real compressed GLYPH archives from synthetic text, then exercise
multilingual exact phrases, literal operators, separate versions, grant filtering,
revocation, corruption before/after indexing, incomplete coverage, limits and
the planner adapter. These do not execute a real LLM, the 40 product acceptance
scenarios, 1 TB indexing, UI acceptance, or industrial certification.

Design reference: SQLite's official FTS5 documentation, including the trigram
tokenizer and its minimum-three-character matching limitation:
https://www.sqlite.org/fts5.html#the_trigram_tokenizer (consulted 2026-09-14).
