# HDFS Benchmark — GLYPH v0.1

Dataset:

- HDFS.log
- size: 1.5GB
- source: Loghub / HDFS dataset

Query set:

- 100 unique `blk_*` IDs extracted from the dataset

Baseline:

    time while read -r q; do
      grep -F "$q" HDFS.log > /dev/null
    done < hdfs_queries.txt

Result:

- grep -F loop: 14.385 sec

GLYPH index:

- SA32u built
- BWT built
- FM-index built
- persistent backend: query_fm_server_v1

Persistent FM benchmark:

- queries: 100
- total_time_sec: 0.001526
- avg_ms_per_query: 0.015261

Interpretation:

GLYPH is not faster than grep for a single ad-hoc scan.

GLYPH is designed for repeated exact queries over pre-indexed static data.
In that mode, it avoids rescanning the corpus and uses the FM-index backend directly.
