# Preview and explicit deletion from Personal Memory

## Delivered scope

The UI opens verified PDF, PNG, JPEG, GIF, WebP and UTF-8 text in a new browser tab.
The current list filename, current card and each historical version have preview
links. Download remains separate. Restoration runs in the existing bounded child
process; the entire selected file (maximum 64 MiB) is verified before a response.
This is not random-range progressive decoding. Temporary decoded files are removed.
No external document conversion service or network upload is used.

Inline response type requires both an allowed extension and signature for binary
formats. Text must decode as UTF-8 with no NUL bytes. HTML/SVG and office documents
are not served as active pages. Responses are no-store, nosniff and no-referrer.
Text/images have CSP sandbox; PDF uses the browser native viewer (CSP sandbox
breaks Chromium PDF plugins), with restrictive CSP and no embedding in other pages.
PDF rendering and settings remain browser-dependent; no claim that all PDF active
features are disabled or that a PDF engine vulnerability is impossible.

## Permanent deletion boundary

There is no trash. A separate confirmation names either the entire document and
history or the chosen version (number, date if known, note and size). The ticket is
bound to CURRENT, path and selected snapshot, expires in ten minutes, is one-use
and requires an explicit POST. GET never deletes. Stale confirmation is rejected.
Deleting current promotes the preceding remaining version; deleting the last
remaining version removes the document. Deleting a middle version preserves the
other content versions, including same-content reverts with distinct saved times.

Deletion rewrites the active snapshot chain without the selected entry/run,
preserving other documents' content and metadata. Old global snapshot URLs become
invalid when their snapshot is replaced; reload other open cards after deletion.
The previous snapshot files are unlinked, not merely hidden by UI filters. Payloads
are unlinked only when no surviving snapshot/recipe refers to them. Legacy range
sources must be retained in full while any surviving version uses their ranges.
The original external base archive is NEVER changed: it is an independent source
archive still required by this pilot. Source files, downloads, portable copies and
backups outside the active overlay remain. This is permanent removal from this
active catalog and eligible overlay payloads, NOT secure erasure from the device,
SSD free space, base archive or backups. UI confirmation states the boundary.

All mutation is under writer.lock; UI lifetime browser.lock remains in use.
DELETE_PENDING.json is fsynced before publishing replacement snapshots, then CURRENT
is atomically replaced and fsynced before old metadata/payload cleanup. Startup and
requests recover a pending operation: old CURRENT aborts new snapshots; new CURRENT
finishes cleanup idempotently after rechecking surviving references. Failure does
not claim success. Cleanup targets are restricted hashed objects in overlay dirs;
symlink files/directories are rejected. Unrelated detached snapshots cause rejection
rather than accidental branch destruction. Max 1000 snapshots, 64 MiB aggregate
canonical metadata and 16 MiB journal. This is a bounded local pilot, not multi-host GC.

## Evidence and remaining acceptance

PREVIEW_DELETE_TEST_OUTPUT.txt: 108 synthetic tests, 29.763 s, OK.
New cases cover exact preview payload/header contracts; unsupported active formats;
corruption and foreign-origin rejection; confirmation, stale/replayed requests;
old-link rejection; middle/current/full deletion; shared payloads and legacy ranges;
source archive preservation; reverts; crash before CURRENT and during cleanup.
No personal laptop files or metadata are included in tests or publication.

Real browser visual acceptance was attempted with a synthetic PDF/image/text.
Playwright was present but its browser executable was absent; browser download from
cdn.playwright.dev timed out. Consequently native PDF/image graphical rendering is
NOT claimed tested here. Laptop acceptance: open a PDF and an image; on a disposable
user-created document add two versions, delete one after confirmation, verify the
other opens, then delete the document. No existing personal document should be
selected just to test deletion.

## Primary references read

- https://restic.readthedocs.io/en/stable/060_forget.html
  Separating snapshot removal from reclaiming unreferenced shared data.
- https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Content-Disposition
  inline versus attachment disposition.
- https://issues.chromium.org/issues/40754148?pli=1
  Search result documents CSP sandbox blocking Chrome PDF viewing.

These inform design; no comparative performance claim is made.
