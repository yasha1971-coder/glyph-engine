# HDFS 1GB Benchmark — GLYPH v0.1

Dataset:

- HDFS.log truncated to 1GB
- source: Loghub / HDFS dataset

Query set:

- 100 unique `blk_*` IDs extracted from the 1GB corpus

Baseline:

    time while read -r q; do
      grep -F "$q" bench_1gb/HDFS_1GB.log > /dev/null
    done < bench_1gb/queries.txt

Result:

- grep -F repeated scan: 11.516 sec

GLYPH index:

- SA32u built
- BWT built
- FM-index built
- persistent backend: query_fm_server_v1

Index artifacts:

- corpus: 1.0GB
- SA32u: 4.0GB
- BWT: 1.0GB
- FM: 8.1GB

Persistent FM benchmark:

- queries: 100
- total_time_sec: 0.001673
- avg_ms_per_query: 0.016727

Interpretation:

GLYPH is not a replacement for grep on one-off ad-hoc scans.

GLYPH is designed for repeated exact queries over pre-indexed static corpora.
In that mode, it avoids rescanning the corpus and uses a persistent FM-index backend.
