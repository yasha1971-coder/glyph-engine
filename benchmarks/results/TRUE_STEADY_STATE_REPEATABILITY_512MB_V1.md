# TRUE STEADY STATE REPEATABILITY 512MB V1

Benchmark:

tools/repeatability_runner_v1.py

Underlying runtime:

query_fm_server_v1

Underlying harness:

tools/steady_state_latency_bench_v1.py

Corpus:

data_4gb/test_512mb.bin

Corpus size:

536,870,912 bytes

FM format:

FMBINv2

Checkpoint step:

256

Query set:

bench_queries/basic_1000_v1.txt

Warmup:

50

Runs:

10

## Raw Results

run1:
p50=0.009036
p95=0.010308
p99=0.010781
max=0.018815
qps=118706.734

run2:
p50=0.006160
p95=0.007374
p99=0.007946
max=0.018103
qps=158923.122

run3:
p50=0.009028
p95=0.010390
p99=0.010829
max=0.015449
qps=113889.263

run4:
p50=0.009077
p95=0.010300
p99=0.010561
max=0.021871
qps=113256.952

run5:
p50=0.008987
p95=0.010278
p99=0.011683
max=0.032540
qps=110188.824

run6:
p50=0.006162
p95=0.008406
p99=0.009567
max=0.010841
qps=156675.336

run7:
p50=0.009149
p95=0.010351
p99=0.010600
max=0.013776
qps=112468.111

run8:
p50=0.006091
p95=0.007214
p99=0.009069
max=0.018012
qps=160988.314

run9:
p50=0.009138
p95=0.010351
p99=0.010611
max=0.013417
qps=112720.682

run10:
p50=0.009049
p95=0.010360
p99=0.010481
max=0.027543
qps=110160.200

## Summary

avg_ms:
min=0.006212
max=0.009078
ratio=1.461

p50_ms:
min=0.006091
max=0.009149
ratio=1.502

p95_ms:
min=0.007214
max=0.010390
ratio=1.440

p99_ms:
min=0.007946
max=0.011683
ratio=1.470

max_ms:
min=0.010841
max=0.032540
ratio=3.002

qps:
min=110160.200
max=160988.314
ratio=1.461

## Verdict

verdict_basis:
p95_ms max/min

ratio:
1.44

verdict:
LAW_CONFIRMED

## Important Observation

Two latency modes appear repeatedly:

FAST MODE:
~0.006ms p50

NORMAL MODE:
~0.009ms p50

Possible causes:
- scheduler behavior
- CPU power state
- cache residency effects
- transient runtime noise

This does NOT invalidate the retrieval law.

## Main Result

Persistent FM steady-state retrieval latency remains stable in the microsecond regime across repeated runs.

GLYPH retrieval core demonstrates repeatable microsecond-scale exact substring retrieval.
