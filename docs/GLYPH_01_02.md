[written by ChatGPT (AI) at the author's request; checked by the author]

This joint pre-release lets an engineer receive a searchable index, choose a new exact request, and send the selected result to a colleague for independent offline checking. glyph-evidence (#1) and glyph-search (#2) also work separately. The full autonomous #1 verification mode is preserved.

The user download contains ready Linux x86-64 binaries, licences, checksums, a public Shakespeare index/packet, and the three README examples below. Normal recipient use needs Python and bash, with the supplied native helpers; no GLYPH compilation or network access is required. Tested on Python 3.12.3. The profile accepts nonempty files up to 128 MiB without NUL; full #1 request files are UTF-8, 1–1024 bytes, cap1–128.

## Three checked examples

Download the user .tar.gz and its .sha256 into the same folder. On Linux x86-64 / Ubuntu WSL:

```sh
sha256sum -c glyph-01-02-user-linux-x86_64.tar.gz.sha256
tar -xzf glyph-01-02-user-linux-x86_64.tar.gz
cd glyph-01-02-candidate
sha256sum -c SHA256SUMS
```

1. **#1 independently verify a full result:**

   ```sh
   bash examples/01_evidence.sh
   ```

   Expected full count 1, offsets [1950801], byte_check=true. This command uses the independent corpus PIN and reconstructs canonical runtime.

2. **#2 ask a new exact request of a transferred index:**

   ```sh
   bash examples/02_search.sh
   ```

   The public request `the` gives full count 47884 and 10 returned positions. Every emitted coordinate was checked against the independent overlapping byte oracle.

3. **#2 -> #1 send a later result to a second receiver:**

   ```sh
   bash examples/03_chain.sh
   ```

   The script transfers software/index first, then introduces a different 40-byte request, creates a portable JSON result, and verifies it in a separate clean receiver. Expected full count 1, offsets [65516], byte_check=true. The original file is not supplied; the receiver recovers and pins it from the index. For your own request, substitute your query file after receiving the index and pass the independently expected corpus/index hashes; the script exposes the exact commands.

The README gives full `verify` and pinned `verify-index` as separate commands. The corpus SHA-256 was fixed before any run; the index manifest SHA-256 is a separately trusted builder-supplied identity. PINS.env is a convenience copy, not an independent trust channel.
