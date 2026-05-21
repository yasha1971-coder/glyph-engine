GLYPH BENCHMARK RULES

Rule 1

Never compare different scopes.

Wrong:

cold_start vs persistent

Correct:

persistent vs persistent

Rule 2

Machine spec required.

Benchmark without machine spec is incomplete.

Rule 3

Commit required.

Every benchmark result belongs to a commit.

Rule 4

Query set required.

Benchmark query distribution affects latency.

Rule 5

Warm and cold results separated.

Never mix.

Rule 6

Single shard and segmented compared separately.

Rule 7

Correctness first.

Incorrect fast benchmark loses value.

Rule 8

Determinism before optimization.

Optimization that changes retrieval behavior is invalid.

Rule 9

Protocol version required.

Query Protocol / Runtime Contract / Capability Contract
must be recorded.

Rule 10

Silent benchmark drift forbidden.

Benchmark methodology changes
must be documented.
