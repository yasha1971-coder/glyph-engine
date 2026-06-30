# BWT_REPETITIVENESS_SOURCE_TRACE_V1

Status: measured
Date: 2026-06-28

## Purpose

Trace the local source/provenance of the strongest BWT repetitiveness signal found by `BWT_REPETITIVENESS_PROBE_V1`.

The strongest measured signal was:

    out/bwt.bin
    n = 100,000,001
    r/n = 0.0394347396056526
    avg_run_len = 25.358

## File identity

| name | path | bytes | sha256 |
|---|---|---:|---|
| main_bwt | `out/bwt.bin` | 100000001 | `69f5a6708eb49c1f0f6f70e5fb285d94a162fc6efd252947d87519cbdfc2311b` |
| corpus2_true_bwt | `out/corpus2_true_bwt.bin` | 100000001 | `69f5a6708eb49c1f0f6f70e5fb285d94a162fc6efd252947d87519cbdfc2311b` |
| py_true_bwt | `out/py_true/corpus2_true_backend.bwt.bin` | 100000001 | `69f5a6708eb49c1f0f6f70e5fb285d94a162fc6efd252947d87519cbdfc2311b` |
| source_corpus | `out/py_true/corpus2_true_backend.corpus.bin` | 100000001 | `4ad7c92379421daee73f1e99f75d12ad83b70a61d9975feebf3c3a11469191c0` |
| sa | `out/sa.bin` | 400000004 | `1c0911abf67ad5b25d71d8846471d658d2a9175b9a8d46659e17b37beddb4546` |
| corpus2_true_sa | `out/corpus2_true_sa.bin` | 400000004 | `1c0911abf67ad5b25d71d8846471d658d2a9175b9a8d46659e17b37beddb4546` |
| py_true_sa | `out/py_true/corpus2_true_backend.sa.bin` | 400000004 | `1c0911abf67ad5b25d71d8846471d658d2a9175b9a8d46659e17b37beddb4546` |

## BWT equality

All available BWT hashes equal: `True`

## Source corpus stats

- path: `out/py_true/corpus2_true_backend.corpus.bin`
- bytes: 100000001
- sha256: `4ad7c92379421daee73f1e99f75d12ad83b70a61d9975feebf3c3a11469191c0`
- alphabet_size: 112
- nul_bytes: 1
- newline_bytes: 0
- printable_fraction: 0.777025
- entropy_bits_per_byte: 5.535348

## Exact block duplicate probe

| block size | total blocks | unique blocks | duplicate blocks | duplicate fraction |
|---:|---:|---:|---:|---:|
| 64 | 1562501 | 304676 | 1257825 | 0.80500748 |
| 256 | 390626 | 162181 | 228445 | 0.58481770 |
| 1024 | 97657 | 79487 | 18170 | 0.18605937 |
| 4096 | 24415 | 24415 | 0 | 0.00000000 |
| 16384 | 6104 | 6104 | 0 | 0.00000000 |

## Interpretation

This confirms that the strongest local RLBWT signal is tied to the old `corpus2_true_backend` artifact family.

It does not yet identify the external origin or semantic domain of the corpus.

Therefore the result is useful as an internal engineering signal, but should not be used as a public domain claim until the corpus origin is named and reproducible.

## Decision

Do not pivot GLYPH to deduplication.

Use this as a gate for the narrower research direction:

    replayable exact-byte evidence over repetitive fixed corpora

Next useful step: measure `r/n` on named real corpora such as HDFS logs, versioned backups, source snapshots, and public log datasets.
