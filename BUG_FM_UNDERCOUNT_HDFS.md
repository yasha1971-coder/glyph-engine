# BUG — FM count undercounts HDFS block-id pattern

Branch:

- feature/segmented-v0.2

Status:

- BLOCKER for segmented correctness validation

Dataset:

- sanity_hdfs/corpus.bin
- source: HDFS_1GB.log copied from bench_1gb/HDFS_1GB.log
- size: 1GB

Pattern:

- blk_-1000095285706020638

Ground truth:

- Python bytes.count: 27

FM result after clean rebuild:

- FM interval: [715386539, 715386552)
- FM count: 13

Commands:

    rm -rf sanity_hdfs
    mkdir -p sanity_hdfs
    cp bench_1gb/HDFS_1GB.log sanity_hdfs/corpus.bin

    ./build/build_sa_u32 sanity_hdfs/corpus.bin sanity_hdfs/sa.bin
    ./build/build_bwt sanity_hdfs/corpus.bin sanity_hdfs/sa.bin sanity_hdfs/bwt.bin 0
    ./build/build_fm sanity_hdfs/bwt.bin sanity_hdfs/fm.bin 128

    python3 - << 'PY'
from pathlib import Path
q = b"blk_-1000095285706020638"
data = Path("sanity_hdfs/corpus.bin").read_bytes()
print(data.count(q))
PY

    ./build/query_fm_v1 sanity_hdfs/fm.bin sanity_hdfs/bwt.bin 626c6b5f2d31303030303935323835373036303230363338

Control:

- synthetic sanity_fm corpus returns correct count:
  - Python bytes.count: 1000
  - FM count: 1000

Conclusion:

The FM path works on small synthetic data but undercounts this real HDFS pattern on a 1GB corpus.

Next investigation:

- validate BWT output against SA for sampled positions
- verify FM rank/checkpoint logic on large BWT
- test same pattern on smaller prefixes of HDFS corpus
- bisect corpus size threshold
