# Field Routing G0 — composition + mask semantics gate (research only)

Date: 2026-11-05  
Tested source commit: \`e91845373ebcfcb3c4b34f0dd0e8c32d64c42d28\`  
GitHub Actions: https://github.com/yasha1971-coder/glyph-engine/actions/runs/42089927296  
Verdict: **35/35 new composition-mask tests PASS; all 64/64 protocol tests PASS**. No production runtime acceptance claim.

## What was exercised

Implementation: \`field_routing_reference.py\` and \`test_field_routing_reference.py\`, both isolated under \`experiments/core_convergence_v1/\`.

A synthetic immutable, five-document logical corpus includes:
- duplicate bytes in different canonical document positions;
- an empty document;
- binary NUL/0xFF bytes;
- distinct metadata labels \`public\` / \`private\`;
- overlapping pattern occurrences and a cross-document false-positive trap.

Two partitions each consist of three contiguous, nonempty blocks:
- A: [0,2), [2,3), [3,5)
- B: [0,1), [1,4), [4,5)

The logical runtime corpus identity and source manifest identity are invariant between A and B; the physical composition-root identity is different. Documents retain global \`(doc_id, doc_offset)\` coordinates. Search is performed **within each document only** using a simple byte scanner, while a separate naive byte-window oracle recomputes the results, even for bounded global locate. The independent oracle does not reuse the search function.

Metadata is bound separately from the corpus byte identity by a versioned hash over the per-document label vector. The field-mask ID commits to the logical corpus ID, source manifest ID, generation, metadata digest, field/value and bit vector. ExternalPin is an explicit required argument; stale pin, mask or version fails closed. Only an explicitly complete set of all required blocks may justify \`PROVEN_EMPTY\` within this **synthetic fixture**.

Negative tests reject missing/duplicated/extra blocks, gaps or overlaps, mutated index/manifest identities, missing root authority input, changed bytes, stale metadata/generation, mutated mask, wrong selected field, falsified count/coordinates/query/max_offsets, and false zero match. Completion scheduling order does not change canonical results.

## What is NOT proved

- This uses mock namespaced block identities; **no compiled C++ FM/RLBWT/Index Forest runtime**, actual operator block artifacts, actual compressed bytes, production bitmap format or persistent field index is used.
- The \`ExternalPin\` is supplied by the test fixture itself; it illustrates a verification API boundary. It is **not an independently distributed, signed, time-stamped, rollback-resistant trust anchor**.
- The separate naive oracle is independent of the byte-search implementation, but runs in the same Python test process. This is **not** O1–O6 independent portable replay or cryptographic non-membership.
- Field metadata is included as a fixture label and does not enforce permissions/ACL; no access-control confidentiality claim.
- No RAM, throughput, index size, scalability, GPU, SIMD or proof-size result.
- This does not close PR #6 Composition Foundations nor authorize the Index Forest runtime. Keep P/R/O/I0 and frozen GLYPH V1 unchanged.

## Execution

\`\`\`bash
python3 -m unittest discover -s experiments/core_convergence_v1 -p 'test_*.py' -v
\`\`\`

Tested first candidate had an **incorrect expected test coordinate** in the synthetic binary document: expected offset 4 where the true and independently returned offset was 3. No engine implementation change was needed. The fixture expectation was corrected and the final commit passed.

## Next gate (only after explicit decision)

Reuse the **real** Composition Reference's 25 EXACT obligations and original independent replay, rather than adding a parallel Forest runtime. Define a versioned field-mask profile over existing operator source-manifest identities; use **real verified blocks** and original independent evidence replay with a mask as an additional bound filtering claim. Any missing-coverage and stale-mask case must remain UNKNOWN/UNTRUSTED; verify existing bounded locate semantics. First make this pass on a tiny fixture, then measure size/latency before adding SIMD.

A separate LOC1→LOC2/enwik8 test may run as an independent, purely additive performance experiment, preserving the existing V2 as reference. Neither gate licenses an Embedded ABI freeze or CUDA rollout.
