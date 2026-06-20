# Public Evidence Demo

This directory defines the first public GLYPH evidence demo path.

Status:

* design skeleton
* no large corpus committed
* no generated index artifacts committed

Purpose:

Show how GLYPH can produce reproducible exact-byte retrieval evidence over a fixed public corpus.

This demo is not intended to prove a whole incident, authorship, intent, legal responsibility, or cryptographic completeness.

It demonstrates a narrow verifiable retrieval chain:

    fixed public corpus
    → GLYPH index
    → exact byte query
    → FM interval
    → offsets
    → Audit Artifact V0
    → audit verifier
    → Evidence Case V1
    → human-readable snippets

## What this demo claims

GLYPH can produce a reproducible audit artifact for exact byte-level retrieval over a fixed committed corpus.

A reviewer can verify that:

* the corpus hash matches
* the index manifest hash matches
* the query hash matches
* the recorded match count matches replay
* the recorded FM interval matches replay
* each returned offset points to bytes equal to the query
* the evidence case snippets are derived from verified offsets

## What this demo does not claim

This demo does not claim:

* legal proof
* zero-knowledge proof
* cryptographic completeness proof
* cryptographic non-membership proof
* full incident reconstruction
* attribution
* semantic truth
* replacement for SIEM, ELK, Splunk, grep, or forensic suites

## Candidate public corpus

The first real public corpus should be:

* small enough for an external reviewer to reproduce
* publicly downloadable
* stable or content-addressable
* free of licensing ambiguity
* suitable for exact string queries
* explainable without hype

Candidate directions:

* small public source-code snapshot
* fixed public text corpus
* small public log-like corpus
* later: XZ Utils / CVE-2024-3094 / JiaT75 / GitHub or GH Archive exact-string evidence demo

The XZ/GH Archive case should be treated carefully.

GLYPH would not prove the whole CVE-2024-3094 incident.

GLYPH would only show reproducible exact byte-level occurrences in a fixed committed corpus.

## Demo procedure

1. Obtain fixed public corpus.
2. Record corpus source and hash.
3. Build GLYPH index.
4. Run exact queries.
5. Generate Audit Artifact V0:

    python3 tools/glyph_make_audit_artifact_v0.py \
      --index-dir <index_dir> \
      --query <exact_string> \
      --output <audit_artifact_v0.json>

6. Verify Audit Artifact V0:

    python3 tools/glyph_verify_audit_artifact_v0.py \
      <audit_artifact_v0.json>

Expected result:

    VERIFY AUDIT ARTIFACT OK

7. Generate Evidence Case V1:

    python3 tools/glyph_make_evidence_case_v1.py \
      --artifact <audit_artifact_v0.json> \
      --output <evidence_case_v1.json>

8. Inspect human-readable evidence records:

    python3 -m json.tool <evidence_case_v1.json> | head -120

## Locate layer requirement

Audit Artifact V0 can verify FM interval and match count.

Evidence Case V1 requires exact offsets.

For human-readable snippets, the index directory must include the locate layer:

    fm_core.bin
    locate_core_s16.bin

Without the locate layer, an audit artifact may still verify successfully, but Evidence Case V1 can contain zero records because no offsets are available.

For local experiments, these generated files stay under:

    examples/public-evidence-demo/work/

and must not be committed.

## Success criterion

The demo succeeds when an external reviewer can reproduce:

    VERIFY AUDIT ARTIFACT OK

and inspect an Evidence Case V1 with:

    byte_check=true

for each evidence record.

## Local validation: Pizza & Chili English 50MB

This demo path has been locally validated on a 50MB prefix of the Pizza & Chili English corpus.

Local corpus:

    REFERENCE_BENCH/OUT/pizza_sentinel_test/english_50mb.txt

Working copy used for the demo:

    examples/public-evidence-demo/work/pizza_english_50mb/corpus.bin

Corpus properties:

    size_bytes: 50000000
    null_bytes: 0

Query:

    Ten Days that Shook the World

Observed Audit Artifact V0 result after building the locate layer:

    reproduce_status: PASS
    match_count: 1
    fm_interval: [12587658, 12587659]
    offset_mode: locate_backend_v2
    offsets: [53]

Observed Evidence Case V1 result:

    records: 1
    byte_check: true

The evidence snippet contains:

    Ten Days that Shook the World

Important lesson from this validation:

Audit Artifact V0 can verify FM interval and match count without offsets.

Evidence Case V1 requires the locate layer to recover exact offsets and human-readable snippets.

## Current next step

Choose the smallest public corpus that makes the evidence path meaningful.

Do not start with a heavy XZ/GH Archive corpus until the small public demo path is clean.

The purpose of this directory is to prevent a jump from mini example directly into a complex incident-scale corpus.
