# glyph-evidence 0.1.0-rc1

One person records an exact query on a fixed file. Another receives the evidence archive, checks the expected file hash, independently counts overlapping matches, checks the returned byte offsets, and replays the existing GLYPH bounded verifier. The archive contains the **entire original file**, exact query and index. Anyone receiving it can read them.

## Try it on Linux x86-64 or Ubuntu x86-64 in WSL

Extract the release tarball and enter its directory. The trusted release includes four statically linked native executables and the existing Python producer/verifiers. Install Bash and Python 3.10+ with the standard library. Compilation, a repository checkout and network access are unnecessary for using the release.

    printf 'banana banana' > notes.txt
    printf 'ana' > phrase.bin
    ./glyph-evidence create --file notes.txt --query-file phrase.bin --max-offsets 2 --out result.glyph
    ./glyph-evidence verify result.glyph --query-file phrase.bin --expect-corpus-sha256 "$(sha256sum notes.txt | cut -d' ' -f1)"

Expect count 4, two distinct zero-based offsets and bounded=true. Bounded offsets follow the engine's selection order and may differ from the first matches in file order. Exit 0 means success. A rejection returns a nonzero code and an error. Success output is JSON. Existing output paths are protected against replacement.

For an actual transfer, send result.glyph to a second machine with the trusted CLI. Obtain the expected corpus SHA-256 independently (for example from the recipient's prior record). Only the demo calculates it from notes.txt. Pass --query-file if the recipient wants to check a particular query. Otherwise read the exact query_hex from the result.

## Inputs and verification cost

This first release reuses the RLBWT V1 sentinel profile. It accepts regular nonempty NUL-free corpora through 128 MiB, and 1–1024 NUL-free UTF-8 query bytes. Query files are consumed literally, including trailing newlines. The offset cap is 1–128 (default 10). This is a single-file tool. A .glyph is a ZIP transport around the existing V1 bundle. The CLI limits both wire size and declared total expanded size to 768 MiB.

In corpus-pin mode the recipient recreates three native index files from the pinned corpus before invoking native replay on the received index. Allow temporary disk and build RAM comparable to the producer. For direct replay of an archive from a **trusted producer** the recipient may additionally supply --expect-bundle-sha256 from an independent trusted record. This skips index reconstruction. The independent corpus scan, query comparison if requested, manifest verification and native replay still run. A checksum sent alongside untrusted data does not establish trust.

Input preparation may be costly for large files. Previously published lookup and verification times for other profiles do not measure this CLI. This release candidate makes no production-readiness or general hostile-input certification claim.

## Build the same package from source

From the repository root on GNU/Linux x86-64, run:

    tools/glyph-evidence/build-release.sh dist

Build dependencies are Bash, GCC/G++ supporting C++17 with static C/C++ runtime archives, GNU tar, gzip and coreutils. Vendored libsais sources and their license ship in the repository. The builder writes the tarball and its SHA-256 checksum to the chosen output directory.
