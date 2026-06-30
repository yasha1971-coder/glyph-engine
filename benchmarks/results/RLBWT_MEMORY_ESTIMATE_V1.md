# RLBWT_MEMORY_ESTIMATE_V1

Status: measured estimate
Date: 2026-06-28

## Purpose

Convert measured BWT run counts into rough RLBWT storage estimates.

This is not a production RLBWT index size. It is only a first-order estimate for run storage.

## Estimate model

- `5 bytes/run`: 1 byte symbol + 4 byte run length
- `9 bytes/run`: 1 byte symbol + 8 byte run length
- Does not include full rank/select/locate overhead
- Does not claim production memory usage

## Results

| BWT path | BWT bytes | runs | r/n | avg run | 5B/run | 5B/run vs BWT | 9B/run | 9B/run vs BWT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `/home/glyph/GLYPH_CPP_BACKEND/hdfs_out/hdfs.bwt.bin` | 1577982906 | 169747064 | 0.10757218 | 9.296 | 848735320 | 0.538x | 1527723576 | 0.968x |
| `/home/glyph/GLYPH_CPP_BACKEND/bench_1gb/out/hdfs_1gb.bwt.bin` | 1073741824 | 114137344 | 0.10629869 | 9.407 | 570686720 | 0.531x | 1027236096 | 0.957x |
| `/home/glyph/GLYPH_CPP_BACKEND/sanity_hdfs/bwt.bin` | 1073741824 | 114137344 | 0.10629869 | 9.407 | 570686720 | 0.531x | 1027236096 | 0.957x |
| `/home/glyph/GLYPH_CPP_BACKEND/REFERENCE_BENCH/OUT/pizza_chili_english_2gb/bwt.bin` | 2000000000 | 582473952 | 0.29123698 | 3.434 | 2912369760 | 1.456x | 5242265568 | 2.621x |
| `/home/glyph/GLYPH_CPP_BACKEND/REFERENCE_BENCH/OUT/pizza_sentinel_2gb/bwt.bin` | 2000000001 | 582473953 | 0.29123698 | 3.434 | 2912369765 | 1.456x | 5242265577 | 2.621x |
| `/home/glyph/GLYPH_CPP_BACKEND/out_1gb/bwt.bin` | 1000000000 | 294636317 | 0.29463632 | 3.394 | 1473181585 | 1.473x | 2651726853 | 2.652x |
| `/home/glyph/GLYPH_CPP_BACKEND/out_2gb/bwt.bin` | 2000000001 | 294636323 | 0.14731816 | 6.788 | 1473181615 | 0.737x | 2651726907 | 1.326x |

## Interpretation

If `5B/run vs BWT` is not far below 1.0, RLBWT is unlikely to be a major memory breakthrough for that corpus.

For GLYPH, RLBWT should only become a priority when named real corpora show both:

- low `r/n`, ideally `<= 0.05`
- estimated run storage clearly below raw BWT size

## Current decision

Do not build production RLBWT runtime yet. Continue using RLBWT as a measured viability branch, not as the main GLYPH claim.
