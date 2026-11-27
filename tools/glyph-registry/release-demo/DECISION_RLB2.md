Decision №3: preserve release GLYPH as the primary user format.
Accepted evidence: glyph03-five-sh1mnw75, immutable source subsets/queries/configs.
RLB2 has a smaller working index and faster warm exact search, but today's №1
requires compatibility export to RLB1/RLR1 plus an extra release reader (LOC2
unchanged). The measured full receiver path is slower and full transfer larger.
Do not promote RLB2 solely on rank/index bytes or warm query timing.
Git: release/RLB2 working index 16655642/14051888B; warm p95 26.695605/3.565939ms; full pinned receiver PASS 3.975793/24.309359s.
CPython: release/RLB2 working index 42949918/35549102B; warm p95 58.481976/6.940924ms; full pinned receiver PASS 11.122073/68.083329s.
The query percentiles belong to the accepted24-query series; receiver times use
q20/limit10 and an already-local archive, not network delivery/canonical rebuild.
This is a trade-off on these corpora/configurations, not a universal ranking.

Rule for subsequent modules and declared links: account EACH conversion separately:
input/output format/version and hashes; every additional transmitted, executable,
map, temporary and persistent file/byte; cold/warm walltime including validation;
loaded RAM and observed/measurement-scoped peak; original guarantees retained or
lost. Give first handoff and each next result separately. Include conversion costs
in the whole receiver boundary; no free adapters/recovery helpers/second indexes.
Missing per-conversion RAM measurements remain UNMEASURED, not zero. Reuse an
existing compatible mechanism first; review format/license/dependency cost before
borrowing. Only separately accepted measurements can justify a future format switch.
