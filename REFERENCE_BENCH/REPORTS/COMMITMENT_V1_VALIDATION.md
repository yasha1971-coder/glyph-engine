# COMMITMENT V1 VALIDATION

Milestone: Corpus Commitment Layer V1 — minimal hash-only commitment prototype
Scope: Commitment Object construction and verification ONLY. No proofs
(presence, absence, count, interval) are implemented or claimed.

## What this prototype proves

A Commitment Object V1 deterministically binds:

1. corpus bytes      -> corpus_root      (chunked Merkle tree, SHA256)
1. suffix array bytes -> sa_root          (chunked Merkle tree, SHA256)
1. build specification -> build_spec_hash (canonical JSON, SHA256)

into a single root:

```
commitment_root = SHA256( utf8(corpus_root_hex)
                        || utf8(sa_root_hex)
                        || utf8(build_spec_hash_hex) )
```

An independent verifier recomputes every value from raw bytes and confirms
agreement. `created_at_utc` is metadata only and does not enter the root:
rebuilding at a different time yields a byte-identical commitment_root.

## Normative construction parameters (committed via build_spec_hash)

- commitment_version: commitment-v1
- hash_function: sha256
- chunk_size: 4096
- leaf format (both trees): sha256(0x00 || chunk_bytes)
- internal format (both trees): sha256(0x01 || left_digest32 || right_digest32)
- odd-leaf rule: promote-last-node-unchanged
- empty file rule: one leaf over the empty chunk

## Commands

Build:

```
python3 tools/build_commitment_v1.py \
    --corpus /home/glyph/GLYPH_LEGAL_PILOT/corpora/sanity.txt \
    --sa     /home/glyph/GLYPH_LEGAL_PILOT/out/sanity/sa.bin \
    --out    REFERENCE_BENCH/REPORTS/commitment_sanity_v1.json
```

Verify:

```
python3 tools/verify_commitment_v1.py \
    --commitment REFERENCE_BENCH/REPORTS/commitment_sanity_v1.json
```

Expected corpus_sha256 (known sanity value, must appear in build output and
pass verification):

```
531a90f4cb8d299a91400ee38a917e0d1066f417830f6e63fd706b11ddaf6ca3
```

## Official run results (fill in from the sanity run)

- corpus_sha256   : 531a90f4cb8d299a91400ee38a917e0d1066f417830f6e63fd706b11ddaf6ca3
- sa_sha256       : 495e53b47d5c7fe5c28df2574dccd56abc94007269af0c5f2932fefa3c5230fd
- corpus_root     : a1705a782515e0756b572284e7be204d5a232f4feb8547bf20e099a2ab8a482b
- sa_root         : b6e0899cd1ba4927b4379361d28bcb3c40e7983def01472f77d77f8be95d538d
- build_spec_hash : 2ce7024ea16b717843148304fa59b25504781a937ec5ee4b12d6c2eddff36800
- commitment_root : c5313bd7b25e8ed9e38a23d7f42ed3b4a36a550fd643ecd9059830e3819494d0
- Result          : COMMITMENT VERIFIED
- Verifier exit code: 0

## Pre-delivery validation (performed on a synthetic stand-in corpus,

## since the real sanity artifacts live only on the GLYPH machine)

Test corpus: 227 bytes containing “right to privacy” at byte offset 49;
real suffix array (uint64 LE) generated for it. Results:

- T1 build + verify: all six checks OK, printed COMMITMENT VERIFIED, exit 0.
  Example commitment_root from this stand-in run:
  c03ad6b75c0a94b719bb1dc95b37de730f8d296b54a74c86cfdd9ba7b3511f77
- T2 tamper test: flipping one corpus byte at offset 49 ->
  COMMITMENT VERIFICATION FAILED, exit 1.
- T3 multi-chunk odd-leaf test: 6 leaves at chunk-size 64 (odd interior
  levels exercised) -> COMMITMENT VERIFIED, exit 0.
- T4 determinism: rebuild at a later timestamp -> created_at_utc differs,
  commitment_root byte-identical.

build_spec_hash for the normative parameters (chunk_size=4096) is constant
across machines:

```
2ce7024ea16b717843148304fa59b25504781a937ec5ee4b12d6c2eddff36800
```

(For the stand-in run above chunk_size=4096 was used; any GLYPH run with
default parameters must reproduce this exact build_spec_hash.)

## Explicit non-claims

This prototype does NOT provide membership proofs, non-membership proofs,
count proofs, interval proofs, or any per-query verification. It proves only
that corpus bytes, SA bytes, and the build specification can be bound into
one deterministic, independently recomputable root. The SA is committed as
opaque bytes; its correctness as the suffix array of the corpus is NOT
attested by this commitment.