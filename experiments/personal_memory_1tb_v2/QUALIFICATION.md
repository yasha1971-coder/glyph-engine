# Qualification measurements V1

This is an engineering measurement runner, not industrial certification.
Run against a completed, unchanged Golden Demo RC1. Stop the demo browser before
running; permanent deletion of benchmark history invalidates that workload.
No source file changes or writes to the existing demo are performed by this runner.
Synthetic regressions operate in their own temporary directories.

    python3 experiments/personal_memory_1tb_v2/qualification.py --demo /path/to/GLYPH-GOLDEN-DEMO-RC1 --output /path/to/NEW-QUALIFICATION

Output must be new and separate. Requires the staged input directory for baseline.
The runner performs the full regression suite, five fresh-process batches each of
12 corpus reads, 101 history reads, and independent deflate6 encode/decode, then
two concurrent whole-corpus readers. Every decoded output is verified. Each
worker has a 600 second deadline. Repeats configurable from 3 to 10.

Raw per-read/per-file samples, batch durations, Linux process peak RSS, nearest-rank
p95, medians, source hashes, environment, test log and checksummed JSON are retained.
Failures retain partial evidence without producing a successful final report.
No automatic uploads. No administrator privileges or filesystem cache flushes.
Fresh process does NOT mean cold storage. Deflate baseline excludes catalogue
metadata; its decode timer excludes external validation, unlike GLYPH internal
checks. These differences must accompany comparisons. No p99/SLA claim from five
samples. CPU count is not a hardware identification or frequency measurement.

Not covered: real power loss, filesystem ENOSPC at each write boundary, 1TB scale,
real PDF/editor workload distributions, encryption, native browser timing,
iPhone performance, or thermal steady state. Existing journal recovery tests are
simulated failures, not physical power-cut qualification. Original Golden report
contains the synthetic insert/delete/revert space experiment.
