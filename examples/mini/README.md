# Mini example

This is a minimal end-to-end GLYPH pipeline.

It builds a tiny corpus, constructs:

- SA32u
- BWT
- FM-index

Then it runs a direct FM query for `error`.

Run:

    ./examples/mini/build_mini.sh

Expected output includes:

    count:    2

This example is intentionally small and does not use the HTTP layer.

For the larger local benchmark, see:

- RUNBOOK_4GB.md
