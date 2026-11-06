# CORE-L6 — run-interval LF research, tracked results and limits

Date: 2026-11-06 Europe/Berlin. Research branch in [Draft PR #21](https://github.com/yasha1971-coder/glyph-engine/pull/21). Historical frozen source pinned to \`c33cb493bc75f13ee4a904fabb190a29a196c94b\`. **NOT an accepted native runtime, Move-r implementation, standalone P/R/O proof or production speed claim.**

## Algorithm and memory accounting

A direct 32-bit dense LF mapping stores 4*n bytes. New \`native_run_interval_inverse_rlb2_ef.cpp\` instead exploits the classical run lemma, with BWT run start \`l_j\` and interval target \`LF(l_j)\`:

\`\`\`text
For l_j <= row < l_{j+1}: LF(row) = LF(l_j) + row - l_j
\`\`\`

An auxiliary coarse predecessor table contains no more than r entries, with at least 512 input positions per cell for large r and a wider cell where r is small. Its entries are *temporary*, not serialized next to the existing RLB2/RLR2/EF. Queries use binary predecessor inside a cell: **O(log runs-per-bucket), not O(1) worst case**. This is a baseline run-interval structure, not the balanced or length-capped constant-amortized Move-r/Orbit algorithms.

Transient index bytes:
\`\`\`text
4*r run starts + 4*r run targets
+ optional 1*r byte symbols
+ 4*b hint run indices,  b <= r
\`\`\`
The original encoded RLB2, retained EF and full reconstructed output still reside in RAM. Therefore **the process is NOT O(r) total memory**; only the LF navigation representation scales with r in this experiment. A streaming output and/or native bounded-memory LF iteration is a separate gate. Working representation uses u32 and currently rejects n beyond 2^32.

Prior art: [Move-r](https://github.com/LukasNalbach/Move-r), [Orbit](https://github.com/drnatebrown/orbit), [SEA 2026, Brown & Langmead](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SEA.2026.9). The run-interval LF lemma and O(r) representations are known in computer science; no mathematical novelty is claimed.

## Gate A: actual binary-safe 24B frozen fixture — PASS

[CI 42137937725 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42137937725):

- 18 out of 18 exact full-source reconstruction variants, text sampling steps 2/3/5, modes rank/direct and thread counts 1/2/4.
- Three hostile input cases REJECTED: RLB2 payload bitflip; EF header magic corruption; truncated EF.
- Executed output SHA equal to original fixture, including NUL and 0xFF. This is small-model component evidence, not a 100MB speed result.

## Gate B: 8MiB deterministic highly repetitive edited snapshots — PASS

[Initial CI 42138252080 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42138252080).
[Counterbalanced three-repeat CI 42138679311 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42138679311) at head \`af1615013745545554102df2e4dd2c84bafae9d1\`.

Constructed 128 versions of one 65,536-byte byte sequence, each changing 32 positions. Source 8,388,608 bytes; canonical BWT 8,388,609 rows and **24,785 runs** (\`r/n=0.0029546018893001212\`). Original raw/SA/BWT were removed before native recovery. Every one of **21/21** native reconstructions matched the full source SHA-256. RLB2+RLR2+EF retained only **1,350,543 bytes** (about 0.1610× on this deliberately repetitive fixture, **not** inclusive of proof, root, code or full product). This corpus class was designed to favor repetitive structure; it does not represent typical personal files.

| Algorithm | Threads | Median full wall, three repeats | Maximum peak RSS |
|---|---:|---:|---:|
| Original sequential dense LF | 1 | 1.272152 s | 45,880 KB |
| Original dense LF, rank | 1 | 1.376100 s | 45,304 KB |
| Original dense LF, rank | 4 | 0.555334 s | 45,428 KB |
| Run-LF, rank | 1 | 1.644425 s | 12,836 KB |
| Run-LF, rank | 2 | 0.849027 s | 12,908 KB |
| Run-LF, rank | 4 | 0.773656 s | 12,964 KB |
| Run-LF, direct per-run symbol | 4 | **0.519392 s** | **12,964 KB** |

- Transient LF run index: **263,820 bytes** (rank) / **288,605 bytes** (direct), versus **33,554,436 bytes** for a 32-bit dense LF over this source.
- Run-based peak process RSS drops from ~45 MB to ~13 MB.
- 4T run-direct median is ~6.47% lower wall time than 4T dense-rank; **only 3 repeats**, no CPU-frequency control, not a p95/p99 or production throughput claim.
- 4T run-rank is **slower** than 4T dense-rank, showing that extra predecessor work matters despite RAM savings.

## Gate C: original canonical enwik8 100MB — PASS, one run per profile

[CORE-L6 enwik8 CI 42138066924 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42138066924), executed code head \`f3b95e07de6668605c16bf315defe40e6e5336f9\`. Same pinned source/SA/BWT/RLB2/RLR2/EF hashes reproduced, construction-only raw/SA/BWT deleted before any native decode. Every **7/7 full 100,000,000-byte restored source SHA-256** matched exactly. Frozen reference decoder, research run-interval index and optional direct run symbol encoding were executed on the same GitHub runner.

| Decoder | Threads | Full process wall | Peak RSS (kB) |
|---|---:|---:|---:|
| Frozen original sequential dense LF | 1 | 30.543427 s | 575,868 |
| Run-index rank | 1 | 51.980290 s | 466,176 |
| Run-index rank | 2 | 31.517583 s | 466,152 |
| Run-index rank | 4 | 25.964716 s | 466,116 |
| Run-index direct (extra symbol byte per run) | 1 | 52.322078 s | 502,052 |
| Run-index direct | 2 | 31.761341 s | 502,044 |
| Run-index direct | 4 | 25.691846 s | 502,044 |

- enwik8 **r=36,661,040** BWT runs over 100,000,001 rows: \`r/n≈0.3666104\`, much higher than deliberately repetitive test data.
- Transient run LF index: **294,069,572 bytes** rank, **330,730,612 bytes** direct (versus **400,000,004 bytes** for a dense u32 LF mapping).
- Run rank 4T reduces process peak RSS by ~19% against the sequential dense reference on that runner, **but** the rank1 variant is substantially slower.
- Coarse hints: **195,313** entries, maximum **504** candidate runs per bucket. Bucketed binary search does **not** match the O(1)-amortized theoretical Move/Orbit structure.
- Only the original **sequential** dense baseline was present in this CI. The already observed CORE-L5 dense 4T time of 17.633s is a *different execution*. Do **not** compare these as a same-run 4T speed proof.
- The prior note that same-run dense 4T had not executed is **superseded** by Gate D below. Preserve the historical run identity and distinguish results from different runs.

## Gate D: same-run 100MB dense 4T versus run-based 4T — PASS

[CI 42139617303 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42139617303), executed source commit `1ec9734755e80a7d98a5152a5f4f29358e76dd5f`. The runner rebuilt and hash-pinned canonical enwik8 and the original RLB2/RLR2/EF, deleted raw/SA/BWT before decode and verified **9/9 complete source SHA-256 matches**.

| Profile | Same-run full wall | Peak RSS kB |
|---|---:|---:|
| Original dense LF 4T | **17.608084 s** | 569,568 |
| Run LF rank 4T | 25.276730 s | **466,152** |
| Run LF direct 4T | 25.094125 s | 502,052 |

**Measured Pareto tradeoff:** rank-run LF reduces process peak RSS by **103,416 kB (~18.2%)** but costs **~43.6% more wall time** than 4T dense on enwik8. The run-direct variant is ~42.5% slower than 4T dense despite an extra 35MiB symbol cache. All results are single-run and no statistically reliable tail latency is claimed.

**Decision:** neither universally replaces the other. Use *corpus-structured admission* driven by actual `r/n`, predicted transient bytes and measured end-to-end wall time under the same machine configuration. No globally hard-coded `r/n` cutoff is qualified yet: the two tested regimes were `0.00295` and `0.36661`. This closes the direct equal-thread comparison and establishes a precise falsifiable cost frontier for the next design.

## CORE-L6 proof bridge

[Lean 4.25.0 / pinned Mathlib CI 42141468659 — SUCCESS](https://github.com/yasha1971-coder/glyph-engine/actions/runs/42141468659) checks `RunIntervalLFProof.lean`: prefix rank of a constant-symbol run increments by its length, so `LF(a+d)=LF(a)+d`. The proof is a theorem about an abstract BWT/rank model, not a verification of C++ implementation, SHA-256 binding or the P/R/O trust graph. A first Lean run failed on an arithmetic rewrite direction and was corrected before PASS.

## Promotion/kill boundaries

- Must return byte-identical source SHA on every declared corpus, including altered/repetitive/large and negative cases.
- On structurally nonrepetitive enwik8, O(r) may consume nearly dense LF memory and be slower: both outcomes are valid and useful results.
- No silent acceptance of a missing, changed or unauthenticated required index. The decoder's internal FNV is not an independently pinned cryptographic proof.
- The global GLYPH claim remains exact count + locate + restored bytes + canonical document coordinates + P/R/O independently authenticated evidence. These experiments cover **recovery and experimental LF navigation only**.
- Frozen formats, prior benchmark evidence, canonical source commits, owner strategic veto and public deployments remain unchanged.
- Next order: finish enwik8 Pareto point; compare genuine independently executed Move-r/Orbit baselines on a repetitive *real* dataset; investigate avoiding O(n) resident output and test 64-bit admission. An LF change by itself does not solve RLB2 enwik8 >0.70× or the proof-layer cost.
