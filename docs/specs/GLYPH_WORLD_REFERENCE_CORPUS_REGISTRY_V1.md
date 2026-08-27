# GLYPH_WORLD_REFERENCE_CORPUS_REGISTRY_V1

Status: normative research registry
Version: 1
Date: 2026-08-27
Identity dependency: `GLYPH_CORPUS_IDENTITY_V1`

## Purpose

Define the exact whole-file public references permitted in official
`RLBWT_BINARY_SAFE_V2` comparative matrices.

The registry prevents same-name substitutions, local slices, derived files and
unidentified benchmark inputs from entering a comparative result.

## Canonical registry

The canonical machine-readable registry is:

    benchmarks/corpora/GLYPH_WORLD_REFERENCE_CORPUS_REGISTRY_V1.json

It contains exactly seven records.

Each record has exactly these fields:

- `reference_id`;
- `class`;
- `document_name`;
- `bytes`;
- `md5`;
- `sha256`;
- `source`;
- `whole_file`.

## Identity boundary

A reference is accepted only when all of the following match:

1. approved `reference_id`;
2. approved public source;
3. canonical `document_name`;
4. exact byte length;
5. full lowercase MD5;
6. full lowercase SHA-256;
7. `whole_file` is `true`.

MD5 preserves compatibility with the established public corpus identity.
SHA-256 provides the stronger GLYPH integrity binding.

Neither an absolute local path nor a hostname is identity-bearing.

On ace-core the files are held read-only under a shared golden root. Another
host may use another path, but it must reproduce the exact registered bytes.

## Default reference set

The V1 default set is:

- UCSC hg38 chromosome 1;
- Matt Mahoney enwik8;
- Matt Mahoney enwik9;
- canonical Silesia tar corpus;
- TEXMEX SIFT1M base vectors;
- TEXMEX GIST1M base vectors;
- BIG ANN DEEP10M base vectors.

## Exclusions

The following are not V1 world-reference inputs:

- locally defined FASTQ slices;
- locally compressed derivatives;
- index sidecars such as `.fai`;
- unregistered files;
- subsets of registered files;
- files accepted only because their basename matches;
- the noncanonical same-name Silesia artifact whose MD5 begins `b918a44a`.

An official accession used for a separate domain experiment does not
automatically become part of the default world-reference matrix.

## Verification order

Before any expensive BWT construction, the verifier must:

1. validate canonical registry serialization and exact schema;
2. reject duplicate, missing, reordered or additional records;
3. resolve the requested `reference_id`;
4. reject symbolic links and non-regular files;
5. verify byte length;
6. stream and verify MD5 and SHA-256;
7. confirm that the file did not change during verification;
8. emit a path-independent canonical result.

Any mismatch fails closed before benchmark construction.

## Relationship to GLYPH corpus identity

The registered `document_name` is supplied as the canonical V1 document name
when the reference becomes a one-document GLYPH corpus.

The local storage path is not supplied as a document name and must not enter
`corpus_id`.

## Non-claims

This registry does not:

- select or implement RLB2;
- prove that a file is suitable for RLBWT;
- claim any compression ratio;
- include rank, locate, metadata or evidence;
- promote RLBWT into required top-level verification;
- modify the live demo.

Its only claim is exact selection and identity of approved reference inputs.
