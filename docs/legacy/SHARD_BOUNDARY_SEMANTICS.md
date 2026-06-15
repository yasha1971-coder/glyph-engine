# SHARD BOUNDARY SEMANTICS

GLYPH v0.x currently supports segmented retrieval by splitting corpora into independently indexed shards.

This improves:
- operational scalability
- memory management
- partial index reuse
- distributed retrieval experiments

However, segmented retrieval introduces important semantic constraints.

---

# Core invariant

Each shard is indexed independently.

FM retrieval operates only within a single shard boundary.

GLYPH v0.x does NOT currently perform:
- cross-shard stitching
- overlap-aware reconstruction
- boundary-spanning verification
- multi-shard suffix continuation

---

# Consequence

Patterns spanning shard boundaries may be missed.

Example:

shard0 ends with:

    blk_000

shard1 begins with:

    123\n

Query:

    blk_000123

Expected global-corpus count:

    1

Current segmented result:

    0

because the pattern crosses a shard boundary.

---

# Current status

This behavior is currently:
- known
- expected
- architectural

It is NOT currently treated as a bug.

---

# Why this matters

Segmented retrieval correctness depends on whether retrieval semantics are defined as:

A:
    exact retrieval within independent shards

or:

B:
    exact retrieval over the logical global corpus

GLYPH v0.x currently implements A.

It does not yet implement B.

---

# Future possible approaches

Future versions may support boundary-safe retrieval via:

- overlap regions
- shard stitching
- rolling suffix carry-over
- hierarchical verification layers
- cross-shard continuation indexes

None are currently implemented.

---

# Current recommendation

Segmented retrieval should currently be treated as:

    exact retrieval within independently indexed shard regions

not as globally complete substring retrieval.

---

# Testing implications

Future regression tests should explicitly include:
- boundary-crossing patterns
- overlap-region semantics
- duplicate-offset handling
- shard-local vs global retrieval expectations

This prevents accidental semantic drift.

---

# Core principle

Segmented retrieval correctness must be defined explicitly.

Silent incompleteness is more dangerous than explicit constraints.
