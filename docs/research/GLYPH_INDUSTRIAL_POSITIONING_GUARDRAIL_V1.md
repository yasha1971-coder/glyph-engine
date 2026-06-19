# GLYPH_INDUSTRIAL_POSITIONING_GUARDRAIL_V1

Status: Positioning Guardrail  
Date: 2026-06-19

## Purpose

This note prevents false comparison between GLYPH and industrial search / SIEM systems such as ELK, Splunk, OpenSearch, and similar log analytics stacks.

GLYPH must be positioned honestly.

## Core Correction

GLYPH is not "search without indexing".

GLYPH is a pre-indexed exact-byte retrieval engine.

Current GLYPH requires a built index:

- suffix array
- BWT
- FM-index
- locate core
- manifest

The correct statement is:

GLYPH works without a heavy always-running database or SIEM runtime after its index has been built.

It does not work without indexing.

## Correct System Type

GLYPH is:

- pre-indexed
- exact-byte
- fixed-corpus
- reproducible
- artifact-producing
- evidence-oriented

GLYPH is not:

- grep replacement for cold-start search
- SIEM replacement
- ELK/Splunk replacement
- realtime monitoring engine
- alerting engine
- semantic search engine
- legal proof system
- zero-knowledge proof system

## Cold Start vs Hot Search

### Cold Start

Cold start means the corpus has not yet been indexed.

In cold start, GLYPH must build its index before it can provide fast exact retrieval.

Current approximate build-time orientation:

- 100 MB corpus: ~10–60 seconds
- 1 GB corpus: ~1–10 minutes
- 4 GB corpus: ~10–40 minutes
- 5 GB+ corpus: must be measured case by case

In cold start, ELK/OpenSearch/grep-like workflows may be faster or more operationally convenient.

### Hot Search

Hot search means the GLYPH index already exists.

In hot search, GLYPH can answer exact byte queries quickly.

Current approximate query-time orientation after index build:

- 1 exact query: milliseconds to seconds
- 10 exact queries: seconds
- offsets/snippets: depends on number of matches and corpus I/O

GLYPH's strength appears after a corpus has become fixed evidence or archive.

## Disk Reality

Current GLYPH indexing expands storage heavily.

Approximate disk expansion:

- ~8–10× corpus size during current pipeline
- recommended safe free disk: ~12× corpus size

Examples:

- 1 GB corpus requires roughly 8–12 GB free disk
- 5 GB corpus requires roughly 40–60 GB free disk
- 10 GB corpus requires roughly 80–120 GB free disk

This is a major current limitation.

Any comparison with ELK/Splunk must mention this.

## Fair Comparison With ELK / Splunk / OpenSearch

### ELK / Splunk Strengths

Industrial log systems are strong at:

- ingestion
- dashboards
- realtime search
- alerting
- correlation rules
- operational monitoring
- multi-user workflows
- continuous log streams

They are designed for live telemetry.

### GLYPH Strengths

GLYPH is stronger for:

- exact byte-level retrieval
- special characters
- raw tokens / keys / domains / IPs / errors
- exact offsets
- snippets around byte matches
- corpus hash
- query hash
- replay command
- audit artifact
- human-readable evidence case
- fixed corpus verification

GLYPH is designed for fixed evidence/archive review.

## Correct Comparison Frame

Do not compare:

ELK cold ingest + search  
versus  
GLYPH hot search over prebuilt index

That is misleading.

Correct comparison:

### Cold Corpus

If a fresh corpus arrives and no GLYPH index exists:

- GLYPH must build the index
- disk expansion matters
- build time matters
- ELK/grep may be more convenient

### Archived / Fixed Corpus

If the corpus is already fixed, hashed, and indexed:

- GLYPH can provide fast exact search
- GLYPH can return exact byte offsets
- GLYPH can produce audit artifacts
- GLYPH can produce human-readable evidence cases
- GLYPH can operate without a heavy always-running search cluster

## Correct Claim

GLYPH can produce a reproducible exact-byte evidence chain over a fixed committed corpus.

Current chain:

corpus
→ index
→ exact query
→ FM interval
→ offsets
→ audit artifact
→ verifier byte-check
→ evidence case
→ snippets

## Forbidden Claims

Do not claim:

- GLYPH works without indexing
- GLYPH is generally faster than ELK
- GLYPH replaces SIEM
- GLYPH is a legal proof system
- GLYPH is a zero-knowledge proof system
- GLYPH has cryptographic completeness proof today
- GLYPH solves realtime log analytics
- GLYPH is better for cold-start search

## Best Current Use Cases

GLYPH should be tested against fixed corpora where exact evidence matters:

1. Incident response archives  
   Check whether an IP, token, domain, or error string appeared in fixed logs.

2. Malware analysis  
   Check whether exact byte strings or signatures appear in a file set.

3. Supply-chain audit  
   Check whether exact forbidden strings, license headers, keys, or vulnerable function names appear in source archives.

4. RAG / AI dataset audit  
   Check whether an exact phrase existed in a fixed knowledge corpus.

5. Forensics-lite  
   Check whether exact emails, wallets, domains, or tokens appear in a dump.

## Best First External Test

Recommended test:

- corpus: 1–5 GB of real logs or source/text files
- queries: 5–10 exact strings
- output:
  - audit_artifact_v0.json
  - evidence_case_v1.json
- goal:
  - determine whether another human can verify the result faster and with more trust than a normal search screenshot or pasted grep output

## One-Sentence Positioning

GLYPH does not replace SIEM.

GLYPH adds a reproducible exact-byte evidence layer for fixed, committed corpora.
