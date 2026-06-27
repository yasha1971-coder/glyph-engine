# RLBWT_LOCATE_OFFSETS_V1_VERIFY

Status: measured local correctness test  
Date: 2026-06-26

## Purpose

Validate RLBWT-aware locate.

Goal:

    FM interval
    -> LF over RLBWT rank
    -> sampled SA hit in LOC1 locate_core
    -> exact corpus offsets

This verifies offset recovery without raw bwt.bin.

Tool:

    tools/rlbwt_locate_offsets_v1.py

## Runtime components

Required runtime files:

- bwt.rlbwt
- bwt.rlbwt.rank
- locate_core_s128.bin

Raw bwt.bin is not used by this locate path.

## Results

| corpus | interval | count | expected offset | recovered offsets | total_steps | max_steps | status |
|---|---:|---:|---:|---|---:|---:|---|
| pizza50 | [12587658, 12587659] | 1 | 53 | [53] | 54 | 54 | PASS |
| xz_cve | [11030, 11068] | 38 | includes 274 | 38 offsets recovered | 5531 | 473 | PASS |
| synthetic_logs50 | [20912727, 20912728] | 1 | 25000227 | [25000227] | 37 | 37 | PASS |

## XZ recovered offsets

274  
328  
1629  
4191  
6254  
6331  
6401  
6923  
8416  
13106  
13176  
13239  
13723  
16472  
16575  
16671  
17289  
18908  
20740  
20808  
20869  
21339  
25239  
25310  
28772  
28848  
28944  
29014  
29103  
29166  
30091  
30151  
33485  
34839  
34941  
35036  
37734  
38141  

## Approximate RLBWT query plus locate runtime size

Using:

- bwt.rlbwt
- bwt.rlbwt.rank
- locate_core_s128.bin
- manifests

### Pizza 50MB

- bwt.rlbwt: 31,922,909
- bwt.rlbwt.rank: 12,649,620
- locate_core_s128.bin: 6,250,040
- manifests: 1,667
- total: 50,824,236
- ratio_vs_corpus: about 1.016x

### XZ CVE corpus

- bwt.rlbwt: 12,790
- bwt.rlbwt.rank: 12,492
- locate_core_s128.bin: 4,904
- manifests: 1,627
- total: 31,813
- ratio_vs_corpus: about 0.817x

### Synthetic logs 50MB

- bwt.rlbwt: 7,404,269
- bwt.rlbwt.rank: 12,649,620
- locate_core_s128.bin: 6,250,040
- manifests: 1,555
- total: 26,305,484
- ratio_vs_corpus: about 0.526x

## Interpretation

RLBWT is now a working compressed query/count/locate path in Python prototype form.

The system can now recover exact offsets using compressed BWT rank/LF plus sampled locate core, without raw bwt.bin.

This moves GLYPH from:

    compact runtime with raw BWT

to:

    compressed BWT runtime with query/count/locate correctness

## Boundary

This is still a prototype.

Missing before production claim:

- C++ implementation
- latency benchmark
- integrated runtime exporter
- integrated runtime verifier
- audit/evidence integration
- multi-query validation
- larger-corpus validation

## Strategic meaning

The old runtime-size objection is now substantially weakened.

Measured direction:

- Compact Runtime V1: about 1.375x on 50MB corpora
- RLBWT Query Runtime V1: 0.891x / 0.401x query-only
- RLBWT query+locate runtime: about 1.016x on Pizza, about 0.526x on synthetic logs

The next technical goal is to package this as:

    RLBWT_FULL_RUNTIME_PROFILE_V1
