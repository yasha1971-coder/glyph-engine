# GLYPH Personal Vault V0 corpus fixture

Status: experiment only. This branch does not modify the Runtime V2 formats, main, demo deployment, OVH, or yasha-context.

## Purpose

Create a reproducible multi-object fixture for the first Personal Vault closed-loop experiment:

ADD -> SEARCH -> RESTORE -> SHA-256 VERIFY

The fixture uses four classic lossless-compression corpora: Canterbury, Calgary, Artificial, and Miscellaneous.

Important correction: these four sets contain **30 files**, not 33:
- Canterbury: 11
- Calgary: 14
- Artificial: 4
- Miscellaneous: 1

Expected payload: **7,252,407 bytes**.

The three-file Large Corpus is deliberately not included in V0. Adding it later would make the combined fixture exactly 33 files.

## Source pin

The downloader uses zlib-ng/corpora only as a reproducible mirror and pins commit:

`5583ca94d1643b6dcd6b6dd2ad0c5704a4afa094`

The upstream descriptions are maintained by the Canterbury Corpus project.

## Safety

Corpus binaries are not committed to glyph-engine. The script downloads them into a disposable directory, verifies file count and total payload bytes, and writes SHA256SUMS.

No Personal Vault storage format is claimed or selected by this fixture. No original may be deleted until a later closed-loop restore gate proves byte-identical reconstruction for every object.
