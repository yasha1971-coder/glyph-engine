# Enrich legacy source dates from Windows — 2026-09-13

The legacy archive omitted original filesystem timestamps. Browser upload metadata
cannot retroactively repair those records. This opt-in helper reads the original
files at explicitly supplied paths, verifies bytes, and adds missing dates.

Use start_personal_memory.sh --enrich-source-dates /mnt/d/EXPLICIT_SOURCE_FOLDER.
The flag is optional; normal startup performs no native source read. Only current
known paths with missing dates are requested. No recursive directory enumeration,
D-drive inventory, archive rebuild, decoding or source writes are performed.
Each candidate's source bytes are read once to compute SHA-256: a necessary read
to avoid attaching current filesystem dates to different archived bytes. A missing
or changed file is counted and skipped. Current root-level aliases with no source
at that path remain unknown; dates are not guessed from another same-named file.

Windows PowerShell uses .NET CreationTimeUtc and LastWriteTimeUtc; FileInfo is
refreshed before/after hashing. Reads use FileShare.Read, and reparse points are
rejected. The helper passes requests via JSON stdin to constant encoded script
text; filenames never become executable PowerShell source. Response records are
matched by exact relative path, size and SHA-256. SHA binds content, not historical
authenticity of a filesystem clock value.

Reference read:
https://learn.microsoft.com/en-us/dotnet/api/system.io.filesysteminfo.creationtimeutc
CreationTimeUtc is filesystem metadata and can be inaccurate/cached. Even after
refresh, it is not proof of the first-ever creation of a document (copies and
filesystem behavior matter). UI labels provenance explicitly.

Only unknown fields are populated. Observation time is recorded separately;
saved_ns is NOT set to now for an old version. Existing snapshots stay immutable,
metadata enrichment creates a successor snapshot without another content version.
The UI retains one history row per content transition and displays supplemented
dates without inventing an original save date. Original archive/payloads stay
unchanged. The browser lifetime lock and writer lock protect CURRENT publication.

Validation: SOURCE_DATES_TEST_OUTPUT.txt records the synthetic Python suite,
including bound dates, changed/missing source rejection, malformed record rollback,
idempotence, old-snapshot and payload preservation, bounded known-path request and
UI history/card display. The collector call is mocked in the known-path test.
Windows PowerShell/NTFS execution is NOT available in this environment and remains
a required laptop acceptance check. No claim of a completed real-source backfill.
PowerShell failure/timeout stops startup before CURRENT is changed. No private
metadata result or laptop document is included in the repository.
