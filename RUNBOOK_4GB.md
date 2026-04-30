# RUNBOOK — GLYPH 4GB Exact Layer v1.2

Goal:
Reproduce 4GB exact retrieval pipeline without chunk_map.

Core:
- SA32u
- BWT
- FM
- no chunk_map in retrieval
- chunk_id = SA[i] >> 14

1. Corpus:
data_4gb/corpus_4gb_s0.bin
size: 4,000,000,001 bytes
sentinel: 0x00 appended, unique

2. Build metadata:

python3 tools/build_meta_for_bin_v1.py --bin data_4gb/corpus_4gb_s0.bin --out-meta data_4gb/corpus_4gb_s0.meta.pkl

python3 tools/export_chunk_starts_v1.py --meta data_4gb/corpus_4gb_s0.meta.pkl --out-csv out_4gb/chunk_starts.csv

3. Build SA32u:

./build/build_sa_u32 data_4gb/corpus_4gb_s0.bin out_4gb/sa.bin

Expected:
libsais64_ok
basic_validation_ok
SA size: 16,000,000,004 bytes

4. Build BWT:

./build/build_bwt data_4gb/corpus_4gb_s0.bin out_4gb/sa.bin out_4gb/bwt.bin 0

Expected:
bwt_ok
bwt_size == corpus_size
bwt_zero_count = 1

5. Build FM:

./build/build_fm out_4gb/bwt.bin out_4gb/fm.bin 128

Expected:
hist_ok
C_ok

6. Retrieval v5, no chunk_map:

./glyph_live_retrieve_v5.py --fm out_4gb/fm.bin --bwt out_4gb/bwt.bin --sa out_4gb/sa.bin --server-bin build/query_fm_server_v1 --query-file /tmp/query_41905.bin

Expected:
outcome: EXACT_MULTI
shortlist_size: 4

7. Benchmark v5:

./glyph_live_benchmark_v5.py --corpus data_4gb/corpus_4gb_s0.bin --fm out_4gb/fm.bin --bwt out_4gb/bwt.bin --sa out_4gb/sa.bin --server-bin build/query_fm_server_v1 --limit-queries 100

Expected:
EXACT_MULTI 100/100
startup ~13.2s
total p50 ~1.7 ms
total p95 ~2.8 ms
total p99 ~4.0 ms

Deprecated:
chunk_map.bin is no longer required for production retrieval.

Old:
chunk_map.bin = 16,000,000,004 bytes

New:
chunk_id = SA[i] >> 14

Limits:
Current validated ceiling: 4GB with SA32u.
Above 4GB requires SA64 or segmented index.
