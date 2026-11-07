# Research to implementation, 2026-11-07

Candidate only, not full integrated Vault acceptance.

Selected primary sources:
- https://github.com/restic/restic/blob/master/doc/design.rst — immutable
  content objects and snapshots. Adapt separation of content identity/logical ID;
  no encryption, chunking or restic compatibility claim.
- https://www.sqlite.org/atomiccommit.html — publication order, flush assumptions,
  and failure testing. Adapt serialized writer and durable prerequisites before
  CURRENT replace; this is not SQLite's journal protocol.
- https://github.com/openai/math — public mathematical manuscripts and Lean proof
  artifacts. Lean files have .lean extension. No special access to internal workspaces
  is available or required. For GLYPH the transferable method is an explicit
  statement plus separately checked evidence, not importing unrelated theorems.

Predictions to test: content sharing should prevent a second physical index for
identical file contents while retaining two logical IDs; new names sorting before
old names should not renumber docs; a crash before CURRENT replacement should
leave old complete state; after replacement, new complete state should be readable.

Local experiment: synthetic-only files <=256 bytes, 3 snapshots, duplicate-content
new doc, explicit rename/revision, binary and empty input, exact indexed search,
independent native selected-source restoration, corrupted EF even on absent query,
SIGKILL at object-ready, snapshot-ready, before-current, after-current, published.
First test FAILED: deterministic retry encountered an already-written unpublished
snapshot (FileExistsError). Fixed by byte-comparing/reusing the immutable snapshot
and flushing it again before publication. Subsequent local smoke PASS.

This is preliminary evidence, not full ACCEPTANCE.md closure: independent logical
snapshot/receipt verifier, stronger schema/path/lineage hostile checks, all planned
negative cases, concurrent writers, CI/cross-host replay, cost accounting and 1GB
remain open. Hard-coded local smoke decoder/reference locations must be made
explicit reproducible inputs before CI acceptance. Local SIGKILL is not power-loss
or filesystem write-order simulation. No new global compression claim.
