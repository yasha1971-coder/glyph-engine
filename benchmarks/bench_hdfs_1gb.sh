#!/usr/bin/env bash
set -euo pipefail

mkdir -p bench_1gb/out

echo "[1] Prepare 1GB corpus"
head -c 1073741824 HDFS.log > bench_1gb/HDFS_1GB.log
ls -lh bench_1gb/HDFS_1GB.log

echo "[2] Prepare 100 queries"
grep -o 'blk_[-0-9]*' bench_1gb/HDFS_1GB.log | sort -u | head -100 > bench_1gb/queries.txt
wc -l bench_1gb/queries.txt

echo "[3] grep benchmark"
time while read -r q; do
  grep -F "$q" bench_1gb/HDFS_1GB.log > /dev/null
done < bench_1gb/queries.txt

echo "[4] Build SA/BWT/FM"
./build/build_sa_u32 bench_1gb/HDFS_1GB.log bench_1gb/out/hdfs_1gb.sa.u32.bin
./build/build_bwt bench_1gb/HDFS_1GB.log bench_1gb/out/hdfs_1gb.sa.u32.bin bench_1gb/out/hdfs_1gb.bwt.bin 0
./build/build_fm bench_1gb/out/hdfs_1gb.bwt.bin bench_1gb/out/hdfs_1gb.fm.bin 128

echo "[5] artifact sizes"
du -h bench_1gb/HDFS_1GB.log bench_1gb/out/* | sort -h
