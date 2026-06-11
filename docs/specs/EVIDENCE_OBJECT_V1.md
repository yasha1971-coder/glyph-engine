# EVIDENCE_OBJECT_V1

Status:
DRAFT

Purpose:

Define the canonical evidence object produced by GLYPH retrieval.

GLYPH must not only return a match.

GLYPH must return reproducible evidence.

Core principle:

LLM may reason.
GLYPH must verify.

Evidence Object V1 represents a deterministic proof that a byte-exact query
was found inside a specific corpus at a specific offset.

Required fields:

query:
Original user query string.

query_hex:
Byte representation of the query.

match:
Boolean.

count:
Number of matches returned by FM search.

interval:
FM interval as [l, r).

corpus_path:
Path of the corpus used for retrieval.

corpus_sha256:
SHA-256 hash of the corpus file.

bwt_path:
Path of BWT artifact.

fm_path:
Path of FM artifact.

sa_path:
Path of SA artifact.

offset:
Corpus byte offset of the selected match.

length:
Length of the query in bytes.

snippet:
Human-readable context window around the match.

snippet_begin:
Start offset of snippet.

snippet_end:
End offset of snippet.

method:
Retrieval method.

Expected value:
sentinel-safe-fm-sa-v1

index_tag:
Human-assigned index or milestone tag.

Recommended value:
retrieval-v1

reproduce_command:
Command needed to reproduce the retrieval.

verified:
Boolean.

timestamp_utc:
UTC timestamp when evidence was generated.

Non-goals:

Evidence Object V1 does not prove legal infringement.
Evidence Object V1 does not prove model training usage.
Evidence Object V1 does not perform semantic search.
Evidence Object V1 does not rank relevance.

It proves only:

This exact byte sequence occurs in this exact corpus at this exact offset.

Next:

Implement --json and --evidence-out in tools/retrieve_v1.py.
