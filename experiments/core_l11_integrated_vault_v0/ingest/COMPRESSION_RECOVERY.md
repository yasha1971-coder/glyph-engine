# Compression recovery / 2026-11-08

Owner requirement: an accepted retained result must not exceed its authenticated
source byte budget. Payload-only savings do not satisfy this requirement.
Do not promote the expanded random8MiB STREAM_V2 index as a compact Vault.

Recovered existing compression code and its32tests from PR14 at
54e4aad8266684dd900311f3960d0455eaeadc00 into its original
experiments/personal_memory_1tb_v2 paths. Whole-file dedup + best actual
raw/deflate/bzip2/xz candidate; reversible pinned Precomp candidate after exact
roundtrip. Real Precomp executable was not invoked here; precomp unit tests use
a fake. Historical VIKA249967975 ->196861728B is an OWNER-REPORTED real-data
archive result, not a fresh test or full indexed-Vault size acceptance.
Historical refs982533/1f8d unavailable in this clone/API; preserved PR14 sources
recover the implementation. Existing PR14 bridge/browser needs incompatible host
interfaces; not copied into current branch or claimed integrated.

New synthetic mixed replay exposed self-referential JSON manifest-size oscillation.
Both old archive serializers now use canonical counted padding only if their old
convergence algorithm fails. Successful historical outputs are unchanged. Padding
is stored, hashed and included in total; it is never hidden from budget accounting.

storage_budget.py counts EVERY regular file in a complete candidate tree, including
manifests, indexes, authentication, history, decoder if required. Symlinks/special
files reject. Caller supplies authenticated source budget; it cannot exclude a
required representation to pass. This is an ACCEPTANCE gate, not a publisher,
source authenticator, crash guarantee or roundtrip proof. It never deletes data.
Usage: python3 -I .../ingest/storage_budget.py --candidate /ABS/CANDIDATE
--source-bytes N. Exit3=BUDGET_REJECTED. Keep candidate outside active data.
Writer integration must enforce this before promotion; current research imports
can still create oversized experimental candidates. They are not product acceptance.

Run existing32tests,3budget tests, then recovery_gate.py --output /ABS/NEW.
One1MiB random source restores exactly but complete archive1050555B rejects.
Mixed1MiB random +1MiB text restores exactly; complete archive1051416B passes.
Results are compression-only; no added self-index or production trust claims.
Source directories/temp/code/install/allocated blocks counted separately, never
call that archive-only ratio a total installed footprint.

Next: connect recovered preservation to the existing immutable snapshot writer
and independent restoration under an explicit new contract; enforce full retained
budget before acceptance, including necessary search representation. Reuse the
current exact-search gates and report unsupported coverage explicitly. Avoid two
full retained media representations. Do not substitute full source scans for the
established interactive TRUST2 path or weaken exactness/UNTRUSTED semantics.
No main/frozen/CONTROL/OVH/private data/merge/new tracker changes.
