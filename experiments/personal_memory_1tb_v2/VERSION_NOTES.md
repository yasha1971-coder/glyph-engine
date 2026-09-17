# Version notes

New uploads accept an optional version_note (maximum 500 Unicode characters).
The note is stored inside the authenticated snapshot alongside the version save
time, and HTML-escaped in the current card and historical rows. The upload choice
retains the note until the user selects a document or cancels. Existing snapshots
are unchanged. Byte-identical uploads remain no-ops, including their notes; editing
notes of already saved versions is not implemented by this change.

This adds no new payload duplication policy: the existing chunk/whole compression
selector remains in use. No guarantee of storing only individual changed bytes.

Validation: 96 synthetic tests pass (VERSION_NOTES_TEST_OUTPUT.txt), including
HTTP upload, real worker, two named versions, exact restoration, HTML escaping,
and oversized note rejection without changing CURRENT. Laptop UI acceptance pending.
No personal source files or laptop documents are included.
