[written by ChatGPT (AI) at the author's request; checked by the author]

## Measured recipient trade-off

Corpus 1: **enwik8, 100,000,000 bytes**, exact range [0,100000000). Identity was confirmed locally against the published MD5/SHA-1 at [Matt Mahoney's data page](https://www.mattmahoney.net/dc/textdata.html). A is one completed working-kit run per implementation; B is the median of five fresh-process checks of the **same single later request**, not five distinct queries. Private query files and local raw logs are not in the download.

| Verification kit | First transfer A, bytes | First full PASS, s | Additional transfer B, bytes | Repeat full PASS B, s |
|---|---:|---:|---:|---:|
| GLYPH | 38,814,541 | 20.357 | 1,017 | 0.677 |
| SDSL FM-index | 59,668,395 | 58.193 | 1,021 | 0.490 |
| Move-r + order adapter | 457,173,473 | 14.389 | 1,026 | 1.828 |
| r-index + order adapter | 306,786,030 | 205.721 | 1,020 | 1.663 |

GLYPH's first transfer is 20,853,854 bytes smaller than the nearest alternative, SDSL, on this corpus/configuration. Move-r finishes the ready-kit first verification faster; SDSL has the fastest repeat verification. Tiny B byte differences are metadata/packaging effects, not a product advantage.

Corpus 2: **The Complete Works of William Shakespeare, PG100, 5,852,404 bytes**, full unmodified 100-0.txt from [pinned GITenberg commit c6728ea4a6edd5f1bfabd31324dbd509b2c6f29a](https://raw.githubusercontent.com/GITenberg/The-Complete-Works-of-William-Shakespeare_100/c6728ea4a6edd5f1bfabd31324dbd509b2c6f29a/100-0.txt). Five alternating one-thread rounds; A is median of five, and B is median time for the **whole five-case later-request series** in each round (absent, frequent cap10, boundary, EOF, frequent cap1). It is not a per-query latency.

| Verification kit | First transfer A, bytes | First full PASS, s | Additional transfer B, bytes | Repeat full PASS B, s |
|---|---:|---:|---:|---:|
| GLYPH | 6,299,039 | 1.572 | 2,566 | 0.463 |
| SDSL FM-index | 4,744,879 | 1.356 | 2,565 | 0.381 |
| Move-r + order adapter | 35,307,890 | 1.105 | 2,567 | 0.883 |
| r-index + order adapter | 19,984,484 | 8.148 | 2,568 | 0.687 |

SDSL is smaller on corpus 2: GLYPH needs 1,554,160 more first-transfer bytes and has no overall first place among these four on this corpus. The predeclared correctness prediction HIT, with zero independent KMP disagreements; the prediction that GLYPH would beat all three on first-transfer size MISSED. No corpus was changed after observing the outcome. Repeated B times across the two tables are not directly comparable because their series differ.

These are complete pinned-index verification kits, including their verification adapters, rather than bare compressed-index sizes or pure query latency. A counts all files absent at a fresh receiver: runtime/software, required libraries and licences, index, request/result, bootstrap and command files. B counts every additional request/result/command file; the receiver already has the verified software, index and corpus cache. Timers start with the archive on local disk; network transfer time and kit preparation are outside these PASS timings. A includes archive hashing/unpacking, index snapshot/hashes, corpus recovery and independently expected corpus SHA-256, full overlapping count, every returned position's bytes, and native replay. B rehashes the index/cache and repeats count/byte/native checks.

The GLYPH pinned-index profile authenticates the separately expected corpus/index identities and verifies this particular result. Full autonomous glyph-evidence separately rebuilds canonical runtime from the independently pinned corpus. SDSL/Move-r/r-index provide search/recovery; the corpus/index/query/result binding and full verification in these tables are supplied by our shared experimental verifier. The stock upstream tools are not being claimed to ship that evidence workflow. Recoverable index/corpus content is available to the recipient.

Both datasets were tested on the same Intel Core i7-1355U laptop, one algorithm thread on CPU0, process-cold and filesystem-warm, with GNU tar + gzip -n -6. Ubuntu 24.04 userspace, Python 3.12.3/glibc 2.39, WSL2 kernel 6.6.87.2; the existing private Ubuntu-userspace bridge was used because the ordinary Ubuntu launcher reported E_UNEXPECTED. Docker remained running. Configurations: SDSL 2.1.1+dfsg4 csa_wt<>; Move-r f48c9e1726a30861afab9b681b89616b2e45eda3, locate-move/suffix-array/threads1; r-index 7009b5374282aa1d8e9b594d0a06c49a0bc1330f, small/divsufsort/fast=false, with its existing locate_all-then-cap adapter. Move-r/r-index adapters sort only the already returned bounded positions. Each table uses its recorded kit revision; minor wrapper/licence/layout changes and the different B series prevent treating the two tables as a pure corpus-only experiment.
