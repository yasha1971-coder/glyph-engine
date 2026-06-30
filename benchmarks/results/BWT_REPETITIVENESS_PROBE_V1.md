# BWT_REPETITIVENESS_PROBE_V1

Status: measured
Date: 2026-06-28

## Purpose

Measure BWT run count `r` and `r/n` on available GLYPH BWT artifacts.

This is the first gate for deciding whether RLBWT/r-index direction is worth engineering.

## Interpretation

- `n` = BWT length in bytes
- `r` = number of equal-byte runs in BWT
- `r/n` = repetitiveness signal
- lower `r/n` means stronger RLBWT potential

Thresholds used:

- `r/n <= 0.001`: extremely repetitive
- `r/n <= 0.01`: highly repetitive
- `r/n <= 0.05`: repetitive
- `r/n <= 0.15`: moderately repetitive
- `r/n > 0.15`: poor run-compression signal

## Results

| BWT | bytes n | runs r | r/n | avg run len | classification | verdict |
|---|---:|---:|---:|---:|---|---|
| `/home/glyph/GLYPH_CPP_BACKEND/examples/mini/out/bwt.bin` | 57 | 41 | 0.71929825 | 1.390 | not_run_compressible | poor RLBWT signal |
| `/home/glyph/GLYPH_CPP_BACKEND/examples/public-evidence-demo/work/pizza_english_50mb/bwt.bin` | 50000001 | 15954147 | 0.31908293 | 3.134 | not_run_compressible | poor RLBWT signal |
| `/home/glyph/GLYPH_CPP_BACKEND/examples/rlbwt-bounded-evidence-tiny/out/index/bwt.bin` | 71 | 58 | 0.81690141 | 1.224 | not_run_compressible | poor RLBWT signal |
| `/home/glyph/GLYPH_CPP_BACKEND/examples/xz-cve-2024-3094-demo/work/phase1_corpus/bwt.bin` | 38928 | 6355 | 0.16325010 | 6.126 | not_run_compressible | poor RLBWT signal |
| `/home/glyph/GLYPH_CPP_BACKEND/out/bwt.bin` | 100000001 | 3943474 | 0.03943474 | 25.358 | repetitive | possible RLBWT signal |

## Decision rule

If real log / backup / versioned / medical / legal corpora show low `r/n`, GLYPH should continue toward compressed RLBWT runtime.

If `r/n` is high on real corpora, do not sell RLBWT as a memory solution yet.

## Non-claims

- This does not prove a compact production RLBWT index.
- This does not prove deduplication usefulness.
- This only measures whether the BWT has enough runs to justify further RLBWT work.
- Simple exact block deduplication may still be better solved by hashing.

## Next external question

Ask a person working with repetitive corpora:

> If a fixed corpus has low BWT run count and supports replayable exact-byte evidence, is that useful in your logs/backups/research workflow?
