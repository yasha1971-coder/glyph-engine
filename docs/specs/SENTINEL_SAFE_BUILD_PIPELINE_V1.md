# SENTINEL_SAFE_BUILD_PIPELINE_V1

Status:
OPEN

Goal:

Create a strict FM-index build pipeline using a real appended sentinel.

Legacy pipeline:

build_sa
build_bwt
build_fm

Problem:

Legacy build_sa builds SA over text length n.

Legacy build_bwt injects a synthetic sentinel when SA[i] == 0.

This creates inconsistent FM semantics.

New pipeline:

build_sa_sentinel_v1
build_bwt_sentinel_v1
build_fm

Model:

Input corpus:
text

Internal indexed text:
text + 0x00

Requirements:

1. Input text must not contain 0x00.
2. build_sa_sentinel_v1 appends real 0x00 before SA construction.
3. SA size becomes n + 1.
4. SA values range from 0 to n.
5. build_bwt_sentinel_v1 builds BWT over text + 0x00.
6. BWT size becomes n + 1.
7. build_fm can consume the new BWT without changes.
8. Query tools must ignore matches that cross or equal sentinel-only suffixes.
9. Locate/snippet tools must map offsets back to original text length n.

Artifact naming:

sa_sentinel.bin
bwt_sentinel.bin
fm_sentinel.bin

Do not overwrite legacy artifacts.

Pizza & Chili target:

REFERENCE_BENCH/CORPORA/english_2gb_prefix_no_nulls.txt

Expected result:

Pattern:

Ten Days th

must return count 1.

Full phrase:

Ten Days that Shook the World

must return count 1.
