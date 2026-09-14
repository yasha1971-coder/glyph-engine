# Adaptive base archive integration

## Laptop acceptance from the supplied terminal output

The user ran the frozen policy at commit 3d846a7 on both Silesia and current VIKA.
Each policy ran three times with two workers. Full per-file JSON reports were
not supplied in this turn; CODEC_FRONTIER_LAPTOP_TRANSCRIPT.json records only
the terminal values and calculations, with that provenance explicit.

| Corpus | Legacy median | Adaptive median | Speedup | Extra payload |
|---|---:|---:|---:|---:|
| Silesia | 66.604 s | 17.355 s | 3.84x | 4.61% |
| VIKA | 55.214 s | 20.769 s | 2.66x | 1.45% |

Both pass the proposed <=5% payload cost criterion. The two-profile scheduler
ran legacy first in every repeat; these results are not counterbalanced or
cold-cache measurements. They do not compare against the Precomp archive that
saved 21.25% on an older VIKA source. No average personal-corpus claim is made.

## Implemented

* Optional `--codec-policy sample-bz-xz6 --workers 2` in verified_hybrid_archive.
  The frozen sampling rule is reproduced in adaptive_codec.py and checked for
  identical output against the independently retained codec_frontier reference.
* Hash before dispatch. Whole-file duplicates are compressed once even if their
  first occurrence is still queued. At most two rows/futures are queued; large
  completed payloads cannot accumulate for the entire corpus. Publication is
  ordered, so a slow leading file can temporarily reduce worker utilization.
* Encoder workers verify exact restoration before the archive writer receives
  a payload. Threads only encode; all filesystem writes remain with the builder.
* Explicit xz-6 support in archive restore and the bounded ArchiveView reader,
  enabling the existing CompressedPreservation and Memory paths to read it.
  Old codecs remain supported. Older readers reject xz-6; xz-6 is never mislabeled
  as xz-9. The new encoding policy and worker count are included in the manifest.
* Golden demo accepts its own previously staged, size/MD5-verified aliases, so
  the next run uses the known twelve files without searching the full D drive.
  Policy/worker options and native CDC library identity are recorded in reports.
* A test exposed a manifest-size fixed-point cycle caused by JSON float lengths.
  When a cycle is detected, small explicitly counted padding breaks the cycle.
  Logical bytes still exactly equal payload + manifest + checksum sidecar.

Legacy defaults stay legacy-best/one worker. Existing archives and the current
personal memory are not migrated. Daily versions still use their existing
raw/deflate/CDC path; replacing that already-cheaper encoder with per-chunk bzip2
would not follow from whole-file benchmark results. Background repacking and
zstd ingestion are not implemented by this change.

The parallel/adaptive pilot accepts 1..500 files, each at most 64 MiB, and refuses
the excluded password-file basename. This is not the 1 TB path. As before, this
base builder does not provide a tested power-loss durability guarantee; its
temporary directory/final rename are not a substitute for a durable commit protocol.

## Verification

135 regression tests passed; the final Golden-demo reporting/preflight adjustment
also passed its five targeted tests. Coverage includes duplicate handling, exact
byte accounting, bad encoder output, missing publication after worker failure,
bad options, password-file exclusion, old codec restore, explicit xz-6 decoding,
cold memory reopen, shared version dependencies, corruption rejection and deletion.

A real full-Silesia integration run then used two archive workers and native CDC,
created the archive, restored and compared all twelve originals, generated 100
XML edits (101 states including the original), reopened and verified every state,
and exercised corruption/deletion on a disposable overlay copy. All gates passed.
Public data only; neither VIKA contents nor private filenames are published.

Payload: 50,095,977 bytes. Complete logical container: 50,103,847 bytes, from
211,938,580 source bytes. Base build: 48.117 s; complete demonstration: 190.980 s.
These are single-run integration timings in the assistant environment. They
must not be substituted for the laptop medians or presented as an application
speedup against a baseline that was not measured through this same path.
The checked-in report truthfully records an uncommitted local worktree at run
time; exact runtime source hashes identify the tested implementation.

Evidence: ADAPTIVE_ARCHIVE_BUILD_RESULT.json, ADAPTIVE_ARCHIVE_ENVIRONMENT.json,
ADAPTIVE_ARCHIVE_TEST_OUTPUT.txt. Graphical PDF rendering, LLM support, recovery
from real power loss, 1 TB operation and production readiness remain unclaimed.

## Run the isolated laptop demonstration

Use the already staged `GLYPH-GOLDEN-DEMO-RC1/input` as `--golden`. Choose a new
output directory on the laptop Linux filesystem. Build with `--codec-policy
sample-bz-xz6 --workers 2 --revisions 100`, then use `golden_demo.py serve` on the
same output. Serving uses port 8767; stop an older Golden demo on that port first.
The working VIKA/Precomp archive is not the source of this demonstration.
