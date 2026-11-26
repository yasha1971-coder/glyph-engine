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

Rule 11 — acceptance of all 23 GLYPH tools

Every tool must have an acceptance card before any release claim. This rule
applies to all 23 tools, including tools not yet individually evaluated; the
existence of the catalog is not evidence that those tools passed. The card
names a professional, one concrete job, the strongest actually verified
alternative for that job, and a measurable reason to choose GLYPH. An untested
competitor is marked untested, never counted as a loss.

Test every tool independently and every advertised combination. For a chain,
measure both the first complete handoff and the incremental bytes, time and
dependencies for each following result with already delivered state. Define
what the recipient trusts, what is sent and what is regenerated. Count all
required files and name preinstalled OS/runtime dependencies explicitly.

Look for an existing GLYPH mechanism first, then a compatible library. Before
reusing code, check byte format, invariants, license and dependency cost. Do
not replace a format merely because its records look similar (LOC1 != LOC2).

Before measuring, commit CALIBRATION: consumer/job, decision, corpus identity,
queries or private query recipe, versions/configuration, forecast, failure
condition, measurement boundaries, cache/thread settings and rejection cases.
Retain every attempt, including failures, signals and harness corrections.
Changing a forecast or silently discarding an inconvenient run is forbidden.

A first-place claim must name its metric, corpus, compared configurations and
actual observations. It is not a universal ranking. A separate controller
checks public numbers; state explicitly when only the same executor reran or
recalculated them. Release and publication decisions belong to Yakov.

Initial cards for the currently tested pair:

- Tool 1, glyph-evidence: an analyst verifies an exact result over a committed
  file. Practical scan baseline: ripgrep + SHA-256; indexed workflow baseline:
  measured SDSL with recovery. GLYPH's stock independent corpus pin, full
  count/byte checks and canonical-runtime reconstruction are explicit features;
  their time/RAM/wire cost must be measured, not called a free advantage.
- Tool 2, glyph-search: an engineer asks new exact queries of an immutable
  transferred index. Measured alternatives: SDSL csa_wt<> for build/RAM and
  Move-r locate-move for loaded queries. The prior 100 MB corpus gave a
  101787726-byte GLYPH index versus 110670771-byte SDSL, and gzip34793113 versus
  58395912 bytes. This is a corpus/configuration-specific reason to choose it,
  not a speed or world-ranking claim. New code must recheck the claimed scope.
- Chain 2 -> 1: another engineer receives and verifies the chosen result.
  Measure fresh-recipient A and already-verified-index B separately; prior
  SDSL recovery kit58626159 bytes is the strongest measured size reference.

Other tools need their own concrete cards and evidence; these two cards do
not accept tools3..23 or imply that their strongest alternatives are identical.
