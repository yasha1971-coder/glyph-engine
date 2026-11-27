[written by ChatGPT (AI) at the author's request; checked by the author]

# GLYPH #3: fixed source-tree index

GLYPH #3 searches an index of an included source-tree sample and returns an exact full count with file paths and zero-based byte offsets. Download `glyph03-candidate-linux-x86_64.tar.gz` and its `.sha256` from the #3 pre-release for the accepted Linux x86-64 demo.

The recipient can choose a new literal query after receiving the index. A second recipient can verify the resulting data capsule offline, recover the included tree from the index, and check every file SHA-256 against independently trusted corpus and index identities.

The release keeps the existing RLB1/RLR1/LOC2 format and the previously released native reader and evidence verifier. The original #1 and joint #1/#2 releases remain separate.

## Run the accepted demo

Use Ubuntu 24.04 / Ubuntu WSL with Python 3.12 and `bubblewrap` (`bwrap`) installed. Native Linux x86-64 helpers and licences are in the archive; recipient use requires no GLYPH compilation, network access or original source checkout.

In an empty directory, with the two downloaded files present:

```sh
sha256sum -c glyph03-candidate-linux-x86_64.tar.gz.sha256
tar -xzf glyph03-candidate-linux-x86_64.tar.gz
cd glyph03-candidate-linux-x86_64
sha256sum -c CHECKSUMS.sha256
```

The independently expected archive SHA-256 is:

```text
c29dcb2096e5b9803acfa705ced3bd95b1703a16d72e14340c8ab1f21aa8c48f
```

Each work directory below must not already exist. Run these commands inside Ubuntu, without embedding the shell variables in a PowerShell one-liner.

Standalone #3:

```sh
python3 demo.py standalone --query ALLOC_GROW --work /tmp/glyph03-alone-1
```

Expected: exit 0, `ok=true`, full `count=252`, `returned=10`, `byte_check=true`, `files_verified=2255`. Results contain `(path, offset)` pairs; the limit does not reduce the full count. Any ten valid occurrences are allowed by this #3 profile; it does not promise the first ten text positions. Replace `ALLOC_GROW` with your own supported literal query, or use `--query-file` as described in the bundled README.

The #3 -> #2 -> #1 handoff:

```sh
python3 demo.py chain --query-file acceptance/chain.query.bin --work /tmp/glyph03-chain-1
```

Expected: exit 0, `count=1`, `returned=1`, `t/t1003-read-tree-prefix.sh` at byte offset 234, `byte_check=true`, `native_replay=true`, `index_unchanged=true`, `files_verified=2255`, `sender_code_executed=false`. The demo also requires a separate changed-count capsule to be rejected with child exit 1 and `count or coverage mismatch`; the top-level demo returns 0 only when the success and expected refusal both pass.

The second recipient obtains its trusted verifier/runtime and corpus/index anchors from the authenticated publisher before receiving the sender's result. A checksum or new trust anchor supplied only inside an untrusted result is not independent authentication. The capsule contains data; the recipient does not execute the sender's scripts. The bundled README gives the exact trust hashes and the expanded verifier command.

## Scope and measured kits

This uses the pinned-index semantic verification profile, `canonical_runtime_rebuilt=false`. The included tree is recovered from the index; full autonomous #1 canonical rebuilding remains a separate mode of the earlier evidence tool. Verification timings start with the archive already on local disk and exclude downloading and index preparation.

Full first-transfer sizes include all files absent at a fresh receiver: compressed kit, checksum, receiver/bootstrap command files and independently expected pins. Paths/boundaries, native helpers, licences, index and result are included. These measurements are separate from the **7,572,753-byte downloadable Git demo**, which has its own README, acceptance examples and packaging.

| Verification kit | Git, 2,255 files: full bytes | CPython, 2,832 files: full bytes |
|---|---:|---:|
| GLYPH release | 7,565,620 | 16,064,796 |
| SDSL csa_wt | 12,782,571 | 30,516,582 |
| Move-r | 70,592,233 | 162,050,587 |
| Zoekt + wrapper | 46,092,173 | 97,788,332 |

In the recorded runs, Move-r had the lowest warm-search p50/p95 on both samples; Zoekt with the verification wrapper had the lowest median time to full receiver PASS. These are different metrics. Complete verification includes recovery of the included tree, full count, checked returned positions, native replay and per-file hashes. The competitor evidence workflow is supplied by our adapters and verification wrappers, rather than being advertised as an upstream feature.

`PUBLIC_MEASUREMENTS.json` gives corpus commits and hashes, aggregate timings and measurement boundaries. The pinned samples are Git commit `683c54c999c301c2cd6f715c411407c413b1d84e` and CPython commit `0fb18b02c8ad56299d6a2910be0bab8ad601ef24`. The figures reuse the accepted same-laptop/configuration campaign; some query samples are earlier accepted runs, not a fully paired simultaneous campaign. Raw controller logs and comparative query files remain local.

## Source and immutable asset identity

`release-demo/` contains byte-identical text/source members from the manually accepted user archive. Five generated-index or native-binary members remain in the release asset, with their hashes in the shipped manifests; a Git source checkout alone is not the complete ready-to-run demo.

`SOURCE_ASSET_MAP.json` records every copied member and the unchanged stock runtime scripts. The asset retains the original tested source snapshot `10ab5f9002792a7c241a700e6086cccf092194e8` in `BUILD_IDENTITY.json`. The public source branch is based on the already released `5fb6fb0c7d825a1070c7d48f70d6533f7e10cd28`, preserving identical delivered code while keeping private experimental history and controller logs local. The asset is copied unchanged, not rebuilt for publication.

Both examples were checked from the final archive by a separate AI controller, and Yakov personally accepted them in Ubuntu WSL. This release-preparation pass rechecked the archive/source hashes and recalculated the public figures from preserved raw records; it did not rerun the benchmark or constitute a new independent external laboratory test.
