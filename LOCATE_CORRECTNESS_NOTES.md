# LOCATE CORRECTNESS NOTES

GLYPH currently has regression coverage for FM-count correctness.

This means the system can verify:

    pattern -> occurrence count

However, count correctness is not the same as locate correctness.

Locate correctness means verifying:

    pattern -> exact byte offsets

These are different correctness layers.

---

## Current state

Current regression tests validate:
- single occurrence counts
- multiple occurrence counts
- absent pattern counts
- full-corpus match counts
- single-byte queries
- overlapping occurrence counts
- deterministic repeated count queries
- terminal sentinel behavior
- rejection of invalid zero-byte input corpus
- rejection of corrupted FM magic

This provides coverage for the count layer.

It does not yet prove the locate layer.

---

## Why locate is a separate risk

FM-count can be correct while locate offsets are wrong.

Typical failure modes include:
- off-by-one LF traversal
- incorrect SA sampling step
- stale locate-core files
- sentinel position returned as valid corpus offset
- mismatch between FM interval and locate backend
- duplicated offsets from segmented retrieval
- missing offsets across shard boundaries

A system can return:

    count = 17

while returning 17 incorrect offsets.

Therefore locate must be tested independently.

---

## Required locate invariant

For every returned offset:

    corpus[offset : offset + len(pattern)] == pattern

And:

    len(offsets) == FM_count

If either condition fails, retrieval is corrupt even if FM-count is correct.

---

## Minimal future test

A future T_LOCATE_VERIFY test should:

1. Build a small sentinel-safe corpus.
2. Build SA/BWT/FM artifacts.
3. Generate a known query.
4. Obtain FM interval.
5. Locate all offsets from that interval.
6. Compare offsets against Python byte-search oracle.
7. Verify every returned offset slices back to the original pattern.

Required assertions:

    sorted(offsets) == oracle_offsets
    len(offsets) == fm_count
    all(corpus[o:o+len(pattern)] == pattern for o in offsets)

---

## Current implementation note

The current locate backend exists as:

    build/locate_backend_v2
    src/locate_backend_v2.cpp

It currently consumes:
- fm_core.bin
- locate_core_sN.bin
- bwt.bin

and receives FM ranges over stdin.

Existing locate tooling appears to use older pickle/export paths:

    tools/fm_export_v1.py
    tools/fm_true_locate_prepare.py
    tools/test_locate_backend_v2.py

These paths are useful for investigation but should not be added to CI until the locate artifact build path is made clean and reproducible.

---

## Shard boundary risk

Segmented retrieval may miss patterns crossing shard boundaries unless overlap or cross-shard stitching is explicitly implemented.

This is not necessarily a bug, but it must become an explicit contract.

A future document should define:

    SHARD_BOUNDARY_SEMANTICS.md

and clarify whether cross-shard matches are:
- unsupported
- supported via overlap
- supported via stitching
- explicitly out of scope for v0.x

---

## Current conclusion

The current regression suite proves FM-count behavior, not full retrieval correctness.

The next correctness frontier is locate verification.

Until locate verification exists, GLYPH should avoid claiming complete offset-level retrieval correctness.
