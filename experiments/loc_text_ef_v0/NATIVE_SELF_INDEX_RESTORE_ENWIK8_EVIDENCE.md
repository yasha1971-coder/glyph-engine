# CORE-L4 — Actual self-index source restore and bounded range extraction

Date: 2026-11-05
Status: **EXECUTED RESEARCH PASS, NOT P/R/O PORTABLE TRUST RELEASE**
Historical original source: \`c33cb493bc75f13ee4a904fabb190a29a196c94b\`.

## A. First native full source reconstruction from one frozen RLB2 (100MB enwik8)

[Exact GitHub Actions run 42116957740 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42116957740), experimental code HEAD \`e7b7c539bc00fc70eff7f8f82fbbe9369f3dd8eb\`.

The clean GitHub runner downloaded canonical Matt Mahoney enwik8, checked the fixed 100,000,000-byte source SHA256 \`2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8\`, built historical binary-safe SA/BWT and original RLB2/RLR2, checked **ALL** original pinned file sizes/SHA256. It then **deleted the original source file, 400MB construction SA and 200MB canonical BWT files** before invoking a new independent native C++ decoder with **RLB2 as the sole input**.

The native code \`native_inverse_rlb2.cpp\` scans the original RLB2 canonical runs, reconstructs an O(n) temporary 32-bit LF permutation, and follows it to recover the original byte stream. It does not read original source, SA, BWT, RLR2, LOC1/LOC2/EF or secondary compressed source payload. The decoder checks RLB2 format, run encoding, sentinel cardinality and FNV, while the outer independent test confirms the *cryptographic* frozen RLB2 SHA256 and recovered full-source SHA256.

**Measured results:**

- 100,000,000 / 100,000,000 recovered source bytes; output SHA256 **exactly identical** to the original external enwik8 SHA256.
- **Native decoder wall 31.114926 s**, peak RSS **576,020 kB (~562.5 MiB)** on the GitHub runner.
- Frozen RLB2 **73,365,105 B**; RLR2 **12,844,028 B**. Together **86,209,133 B (0.86209133x)** as count/search rank substrate (LOC and new proof excluded). No second source payload is needed for *full sequential restoration*.
- Corrupt one byte of RLB2 payload with original header/rank unchanged: native recovery **REJECTED**, no trusted output created.
- The earlier [small native gate 42116483550](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42116483550) also rejected bad header and truncated payload. Source recovered 24/24 binary bytes including NUL/0xFF, original source SHA matched.

**Performance limit:** 31 seconds for 100MB is ~3.21MB/s end-to-end on this runner. This is **not** an ACEAPEX-class decode speed, not zero-RAM and not 1TB capable yet. The method is classical inverse BWT with a full LF array; engineering proof of retaining one compressed source, not a new BWT mathematical invention. Relevant prior art includes Crochemore et al., *Computing the Burrows-Wheeler transform in place and in small space* (2015) and Kärkkäinen et al., *Slashing the Time for BWT Inversion* (2012). Bounded-memory and parallel BWT inversion are future competitive comparators.

## B. Exact byte windows from RankV2 + EF without a second archive

[GitHub Actions small-range run 42118316244 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42118316244), code head \`722e1b502850bfea20feef2c58be6134cc0b539e\`.

Using the original 24-byte binary-safe source, original RLB2/RLR2, physical EF sample file and an in-memory **u32 q-to-SA-row mapping reconstructed from EF data** (not persisted independently), the \`EFByteExtractor\` recovered all **325/325** possible \`(start,length)\` byte ranges and 13 exact query-match byte excerpts; 4 invalid ranges rejected. No extra persistent compressed source payload. The strict work bound for each query is **number of LF steps <= requested byte length + sample_step - 1**; this is NOT a bound on wall latency because one LF call may scan 8192 RLR2 symbols in the current Python implementation.

The sample inversion table has counted ephemeral RAM \`4*m\` bytes; the later O(m) construction walks physically encoded high bits once rather than performing an EF select for each of m samples. Important: the 325-case test is binary-small; no 100MB query/range claim is inherited from it.

## C. Complete retained space and proof gap

A **candidate arithmetic** subtotal for the enwik8 search-and-recover substrate is:
- Frozen RLB2 73,365,105 B
- Frozen RLR2 12,844,028 B
- Experimentally built EF text-order locator 2,844,324 B
- **Total 89,053,457 B = 0.89053457×**, **before** any new canonical manifest, proof bundle, external trust pins, verifier/code package or provenance layers.

The independent RLB2 full-restore gate and the EF physical/search gates each PASS, but until a **same-run integrated test** passes, the entire RLB2+RLR2+EF \`count/locate + arbitrary bytes + full restore\` must be called a **research hypothesis**, not closed. The new [integrated 100MB run 42118746791](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42118746791) is separate and its final status must be checked before promotion.

**NOT proven:** cryptographic global nonmembership, O3/O4 immutable evidence and canonical document coordinate ordering, independent trusted freshness/currentness, source-byte extract speed under strict SLA, 10TB admission, or <=0.70× fully authenticated storage. Existing 73.365MB RLB2 prevents that ratio on enwik8 without a different entropy representation.

## D. Scientific next direction

Use the existing text-position EF samples as independent block boundaries for **parallel** inverse-BWT of disjoint source windows of up to 128 bytes, rather than a 100M-step serial LF dependency chain. This is the falsifiable [CORE-L5 Issue #22](https://github.com/yasha1971-coder/glyph-engine/issues/22): same CPU/hardware 1/2/4 thread tests, full source SHA, normalized LF end-row checks, peak RSS, meaningful baselines. Parallelism itself does not solve O(n) temporary LF-memory cost. A later run-based/select solution may be required for TB.

**Owner strategic veto:** Research PR #21 stays DRAFT; no merges, deployed host updates, frozen format redefinitions, C ABI freeze or authoritative root changes.
