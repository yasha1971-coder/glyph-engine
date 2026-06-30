# NAMED_EXISTING_BWT_RN_MATRIX_V1

Status: measured
Date: 2026-06-28

## Purpose

Measure BWT run ratio `r/n` on existing named or semi-named GLYPH BWT artifacts after the FASTQ NA12878 probe showed poor RLBWT signal.

## Results

| corpus | BWT path | bytes n | runs r | r/n | avg run | verdict |
|---|---|---:|---:|---:|---:|---|
| HDFS full available | `/home/glyph/GLYPH_CPP_BACKEND/hdfs_out/hdfs.bwt.bin` | 1577982906 | 169747064 | 0.10757218 | 9.296 | weak/moderate |
| HDFS 1GB | `/home/glyph/GLYPH_CPP_BACKEND/bench_1gb/out/hdfs_1gb.bwt.bin` | 1073741824 | 114137344 | 0.10629869 | 9.407 | weak/moderate |
| HDFS sanity 1GB | `/home/glyph/GLYPH_CPP_BACKEND/sanity_hdfs/bwt.bin` | 1073741824 | 114137344 | 0.10629869 | 9.407 | weak/moderate |
| Pizza English 2GB | `/home/glyph/GLYPH_CPP_BACKEND/REFERENCE_BENCH/OUT/pizza_chili_english_2gb/bwt.bin` | 2000000000 | 582473952 | 0.29123698 | 3.434 | poor |
| Pizza sentinel 2GB | `/home/glyph/GLYPH_CPP_BACKEND/REFERENCE_BENCH/OUT/pizza_sentinel_2gb/bwt.bin` | 2000000001 | 582473953 | 0.29123698 | 3.434 | poor |
| internal 1GB | `/home/glyph/GLYPH_CPP_BACKEND/out_1gb/bwt.bin` | 1000000000 | 294636317 | 0.29463632 | 3.394 | poor |
| internal 2GB | `/home/glyph/GLYPH_CPP_BACKEND/out_2gb/bwt.bin` | 2000000001 | 294636323 | 0.14731816 | 6.788 | weak/moderate |

## Interpretation

This matrix decides which corpus family, if any, deserves further RLBWT/r-index engineering for GLYPH.

The target is not deduplication as a product. The target is compact replayable exact-byte evidence over repetitive fixed corpora.

## Decision rule

- `r/n <= 0.05`: continue RLBWT investigation.
- `r/n > 0.15`: do not use as compact-index argument.
