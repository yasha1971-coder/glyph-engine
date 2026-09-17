# Personal application: delivery gate, 2026-09-17

This is an incremental improvement to the existing personal-memory UI, not a
new archive format or a claim of industrial certification. No LLM is required
for saving, finding literal text, choosing a version, preview or downloading.
The model experiments are separate from the application and are not admission
tests the user needs to run before using it.

## Completed user journey

Save a UTF-8 text file, update it, search a phrase without case sensitivity across
history, open the pinned historical version and download exact original bytes.
This journey has an automatic HTTP test using real archive and memory fixtures,
not a mocked search worker or a live user's documents.

Main content-search form defaults to available history and Unicode casefold.
Legacy direct URLs retain current-only, case-sensitive semantics. Search results
carry their own snapshot pin, saved date (if recorded) and version note. Opening
does not silently substitute CURRENT. Unchanged path/content/save-time entries
are collapsed across snapshots. Missing dates remain unknown, not fabricated.

The search adapter reads only reachable snapshot ancestry, never filesystem
history globbing. Historical search has limits: 100 snapshots, 10,000 examined
file entries (including repeated entries), 10 seconds ancestry enumeration,
32 MiB source budget, 20 seconds index construction, 20 returned matches.
The worker is limited to 45 seconds. Historical enumeration cutoffs and partial
index coverage are explicitly displayed. Corruption or worker timeout is an
error, never evidence that a file is absent. Index state is temporary and rebuilt.
The original archive and CURRENT are not modified by search.

## Existing capabilities and limits

Existing UI supplies file cards, version upload, version notes, preview, download
and separately confirmed deletion. One upload remains limited to 8 MiB.
Deletion semantics and storage format are unchanged in this revision.
Browser previews depend on browser-supported formats; no general document viewer
or OCR is added. No PDF/photo/audio content recognition, semantic retrieval,
automatic import, unattended backups or authenticated remote access is added.
This revision therefore does NOT yet satisfy every personal-product acceptance
criterion. It must not be marketed as production-ready or 20-year proven storage.

## Release gates before a wider personal release

- Repeatable install/update/rollback with a stable entry point, no experimental
  commands required in routine use.
- Persistent bounded search index, explicit coverage, recovery after index loss.
- Realistic mixed-format intake, extraction provenance and visible unsupported
  formats; OCR and speech engines evaluated independently.
- Backup/restore of the entire application state, disk-full/interruption tests
  through the actual UI, not merely a separate storage prototype.
- Independent human retrieval cases before enabling LLM-mediated answers.
- Clear separation of document date, filesystem date and archive save date.

No user files, model outputs, machine identifiers or credentials are published
with this change. Test fixtures are generated locally and synthetic.

## Verification for this revision

`python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -v`
completed in the development workspace: 174 tests, 168 passed, 6 skipped,
0 failures/errors, 53.425 seconds. Skips are not successful validations.
The experimental packed backend is not installed; its dependent integration
checks remain unverified here. The five new history-search tests all passed,
including real HTTP upload/update/search/pinned-download byte comparison.
No laptop installation or live-data evaluation was performed by this workspace.
