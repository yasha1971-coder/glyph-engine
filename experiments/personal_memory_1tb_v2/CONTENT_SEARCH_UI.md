# Content search UI integration

The personal-memory home screen now offers exact, case-sensitive text search
beside filename search. Search covers the currently pinned incremental snapshot,
including uploaded versions; it does not search the complete historical timeline.
Results link to a separate version card with a pinned preview and download.
Later edits do not silently redirect that download to the latest content.
Removed/unreachable snapshots are rejected. Missing dates remain unknown.

Each request builds a disposable in-memory FTS5 index in a child process using
`memory_content_search.search`. No persistent plaintext index is written by this
component. The existing `limited_child` resource limits and a parent-enforced
30-second timeout apply; timeout kills the process group. At most two read jobs
run concurrently. Search does not hold the mutation lock during decoding.
After completion, CURRENT is checked under lock: changed state requires retry.
All quotes and filenames are HTML-escaped and the existing token/origin checks
remain. This is a single-user local UI, not a multi-tenant permission system.

Limits: 32 MiB source-byte indexing budget (not 32 MiB extracted text), 8 MiB
per file, 10,000 files and 20 results. No OCR or PDF extraction. Existing text
decoder/FTS restrictions apply. Unsupported or skipped content is reported as
incomplete. Corruption/resource failures give an explicit unsuccessful search,
never a successful no-match. The index is rebuilt for every query; repeated-query
performance and a persistent incrementally updated index remain future work.

This adapter can invoke the already-configured verified Precomp backend in its
bounded worker, unlike ArchiveView-only search. No LLM is invoked, and no cloud
endpoint is added. No laptop deployment or industrial performance is claimed.

Validation uses synthetic real archives and localhost HTTP, including upload,
search, pinned version card after another edit, exact-byte download, HTML escaping,
short-query rejection, unknown-version rejection and timeout error handling.
Run from repository root:

```sh
python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -p test_memory_content_ui.py -v
python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -p test_memory_browser.py -v
```

The running laptop scale experiment is independent. Deploy this UI revision only
after that run has finished, using the existing personal-memory startup script.
