# glyph-search 0.1.0-preview2

Build an index of one static file, send the index directory to a reader, and answer new exact substring queries without opening the original. Runs offline on Linux x86-64 with Python 3.10+, glibc with renameat2, and Bash for index construction. The compiled GLYPH programs are included.

## Try it

From the extracted release directory:

```bash
printf 'banana banana' > notes.txt
./glyph-search build notes.txt --out notes.gidx
./glyph-search query notes.gidx --text ana --limit 2
printf '\x62\x61\x6e' > query.bin
./glyph-search query notes.gidx --query-file query.bin
printf '616e61\n62616e\n78797a\n' > queries.hex
./glyph-search query notes.gidx --queries-hex queries.hex --limit 2
```

One JSON line per query. For the three batch requests the expected (count, offsets) pairs are (4, [3, 10]), (2, [0, 7]), (0, []). Counts include overlapping occurrences. Positions are zero-based byte offsets. A positive limit returns the sorted positions of a deterministic subset selected in suffix-array order. A limit of zero counts only. When the cap truncates output, bounded is true.

## Transfer and limits

Transfer the release plus the complete .gidx directory. The directory contains three runtime files and a manifest with corpus and file hashes. The original is absent; the compressed index can reveal its contents and should receive the same access controls. Every query invocation verifies the runtime hashes before opening the engine. --queries-hex accepts one hex-encoded query per line and shares verification and one loaded engine across the whole batch. For accidental damage the manifest suffices. To authenticate a received index, pass the build's independently trusted manifest digest via --expect-manifest-sha256 HEX. Use a trusted builder and immutable directory during querying. The result identifies positions in that builder's index; use glyph-evidence to replay a result against an independent corpus digest.

Supported inputs are regular nonempty files up to 128 MiB containing no NUL bytes. Each query is 1–1024 nonzero bytes. --text encodes UTF-8, --query-file preserves literal bytes, and --limit accepts 0–65536. Build uses temporary disk and RAM and publishes a new directory atomically after success. Choose a fresh output name. Reader workloads needing many queries should use a batch that fits memory. See stderr for build progress and error details.

## Index compatibility

Preview2 reads preview1 indexes and creates v2 indexes for the preview2 reader.
The locate component has a 24-byte little-endian header (LOC2 magic, u64 suffix
array length including sentinel, u32 sampling step, u64 sample count), followed
by one little-endian u32 corpus position per sampled row. Row i in this table
corresponds to suffix-array row i × step. Sampling remains every 128th row.
The BWT and rank formats and query syntax are shared with preview1.
