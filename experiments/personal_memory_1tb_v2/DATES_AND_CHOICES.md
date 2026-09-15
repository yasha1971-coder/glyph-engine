# Source dates and explicit save choice — 2026-09-13

Supersedes automatic duplicate redirection and conflict-only responses described
in earlier UI notes. Extends the existing browser, worker, immutable snapshot,
whole-file identity and hybrid chunk storage; no new storage engine or migration.

## Reality check / primary references read

https://www.w3.org/TR/FileAPI/#file-section
The browser File interface exposes lastModified; it does not supply filesystem
creation time. A nonce-authorized local change listener copies lastModified into
a hidden upload field. Without JS or metadata, the date remains unknown. The
reported timestamp is not an authenticated historical fact and may be affected
by copying, filesystem behavior, browser behavior or inaccurate clocks.

https://restic.readthedocs.io/en/stable/100_references.html
Metadata and content references are separate in snapshot trees. GLYPH likewise
keeps per-path version metadata distinct from shared content identities.
No claim of feature/performance equivalence to restic is made.

## Implemented semantics

For a new or changed path, store saved_ns (host clock), source_modified_ms
(browser-reported or null), source_created_ms=null and date_source. Never use the
upload temporary file mtime as the source date. Dates are shown separately in the
card/history, UTC, with explicit unknown values. Equal bytes at the same path
remain a no-op and do not replace previously captured dates. A separate path gets
its own metadata while reusing the content. Dates are not content identity.

Default uploads matching basename (casefold) or content digest show a choice:
continue history of a listed file, retain separately, cancel. Matches are not
proof of document identity. Up to 20 candidate paths are offered. Explicit Update
this file from the card already supplies the target and needs no second choice.
A separate upload uses its own name, or appends (2), (3), ... if occupied.
No semantic merge of document contents is attempted. Storage routing and dedup
remain automatic; choosing separate does not force a second payload copy.

Choice tickets hold at most two pending uploads in server memory, each <=8 MiB,
expire after 10 minutes and are consumed on acceptance/cancel. They disappear on
server restart; select the file again. CURRENT is checked again before commit.
Cancellation performs no storage write. A corrupted shared object cannot be
accepted through a duplicate choice. Existing source/restore/worker limits apply.
There is still no encryption, native creation-time collector, full metadata
backup or graphical-browser automation. Laptop acceptance remains pending.

## Evidence

DATES_CHOICE_TEST_OUTPUT.txt: 14 HTTP integration tests, real worker processes.
Includes distinct source/save dates, date survival through selection, preservation
of older metadata, invalid-date rejection, separate-copy dedup, cancellation and
stale choice. The JS lastModified event itself is not run by HTTP tests.
DATES_CHOICE_FULL_TEST_OUTPUT.txt records the entire synthetic suite after the
metadata addition. Previous VERSION_GROWTH_RESULT.json was measured at the earlier
chunk-history implementation; this change has additional metadata overhead and
that size measurement has not been reissued for the new schema.

Run:

    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -v

Next acceptance on laptop: add an already-known filename and choose its existing
record; confirm the resulting card separately labels source modification and
GLYPH save time. Keep original archive and all earlier snapshots intact.
