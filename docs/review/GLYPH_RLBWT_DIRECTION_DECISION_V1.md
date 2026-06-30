# GLYPH_RLBWT_DIRECTION_DECISION_V1

Status: current decision  
Date: 2026-06-28

## Purpose

Record the current decision about RLBWT/r-index direction for GLYPH.

## What was tested

GLYPH added:

- `tools/measure_bwt_runs_v1.py`
- `tools/run_bwt_repetitiveness_probe_v1.sh`
- `tools/trace_bwt_source_v1.py`
- `tools/fastq_rlbwt_viability_probe_v1.py`
- `tools/run_fastq_rlbwt_viability_probe_v1.sh`

The goal was to measure whether real corpora have low BWT run ratio:

    r/n = BWT run count / BWT length

Lower `r/n` means stronger potential for an RLBWT/r-index style compressed index.

## Key result: FASTQ NA12878

Measured:

    source: /tmp/NA12878_1gb.fastq
    sample: 256 MiB
    BWT r/n: 0.15165326
    avg_run_len: 6.594
    verdict: poor RLBWT signal

Decision:

    Do not continue FASTQ as the first GLYPH compact-index direction.

Reason:

    The measured r/n is too high for a strong RLBWT memory argument.
    FASTQ remains important for ACEAPEX compression/decode work, but it is not currently a strong GLYPH RLBWT evidence target.

## Earlier internal signal

An old local artifact showed:

    out/bwt.bin
    r/n = 0.0394347396056526
    avg_run_len = 25.358

But this artifact is tied to `corpus2_true_backend`, an internal/old corpus family, not a named reproducible public corpus.

Decision:

    Useful as engineering signal.
    Not usable as public claim until corpus origin is named and reproducible.

## Current GLYPH direction

Do not pivot GLYPH to deduplication.

Use RLBWT only as a measured viability path for:

    replayable exact-byte evidence over repetitive fixed corpora

## Next target corpora

Priority should move to named real corpora closer to GLYPH use:

- HDFS logs
- large static logs
- versioned source snapshots
- backup/version corpora
- legal/audit document corpora

## Rule

If a named real corpus gives:

    r/n <= 0.05

then continue RLBWT engineering for that corpus class.

If:

    r/n > 0.15

do not use that corpus as a compact-index argument.
