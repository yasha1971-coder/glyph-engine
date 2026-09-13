# Personal Memory browser pilot

Supersedes the CLI-only interface status in INCREMENTAL_MEMORY_PILOT.md.
This is still a private pilot, not a replacement for the installed V1 release.

`memory_browser.py` exposes a loopback UI on port 8766 with a random capability
URL, strict Host/Origin checks, same-origin referrer policy and no external JS.
The human chooses one local file (<=8 MiB) using the browser picker. An optional
relative path names its location in memory. Reusing that path adds a new version.
An identical addition is a no-op. The version menu exposes immutable snapshots;
name/path search and selected, hash-verified downloads work within each snapshot.

The UI persists a local CURRENT pointer using fsync and atomic replace. A lifetime
browser lock prevents two UI processes from writing it. A stale upload parent is
rejected with HTTP 409. Explicit CLI branches do not move CURRENT. The pointer is
local state, not an externally authenticated root or rollback protection. If
publication finishes but CURRENT update fails, a snapshot can remain unreachable;
previous data is preserved, automatic orphan recovery is not implemented.

Upload and restore run in subprocesses with the existing Linux resource limits
(1 GiB address space, 128 MiB per output file, 300 CPU seconds, 330-second wait).
This is not a hostile-code sandbox. Trusted local archive/binary assumptions and
all incremental-storage limitations still apply. Original base files up to the
existing 64 MiB read budget may be restored; NEW uploads are limited to 8 MiB.
Empty directories, filesystem permissions, encryption, content search and LLM
integration are not provided. The original archive must remain available.

`bash experiments/personal_memory_1tb_v2/start_personal_memory.sh` launches the
user's pilot. It checks only completed receipt candidates one directory below
`$HOME/GlyphPilot/GLYPH-V2-runs` against the known archive hash, and pins the known
Precomp binary. It does not inventory original data. `GLYPH_PILOT_HOME` overrides
the default base. State is created in `GLYPH-V2-runs/personal-memory-ui-v1`.
An existing state is reopened and validated; it is not reinitialized. Ctrl+C stops
the server; a new URL is printed after each restart. Initial archive saving is not
a measurement of combined storage after additions.

Validation: 6 HTTP integration tests passed (MEMORY_BROWSER_TEST_OUTPUT.txt),
using real worker processes and synthetic archive bytes: upload/change/download
both versions/repeat, reopened storage, stale tab, foreign/null origin, traversal,
corruption, size budget, and original Unicode filename. No graphical browser was
available for automated clicks. User laptop acceptance remains pending.

Next human check: launch, upload a small disposable text file, download it, change
its text, upload under the same path, and select the earlier version to restore
its original bytes. Keep the existing V1 installation and base archive in place.
