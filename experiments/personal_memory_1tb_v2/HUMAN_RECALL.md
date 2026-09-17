# Human recall experiment, 2026-09-16

Research basis: Keeping Found Things Found observed people using multiple ways
to save and re-access information; saving and later retrieval are different
tasks. This is old empirical research, not a claim of a new 2026 discovery:
https://www.microsoft.com/en-us/research/publication/keeping-and-re-finding-information-on-the-web-what-do-people-do-and-what-do-they-need-to-do/

SQLite tokenizer reference consulted:
https://sqlite.org/fts5.html
Implementation uses Python Unicode casefold on both indexed text and query,
with a case-sensitive trigram index; it does not assume SQLite implements the
same full Unicode folding. No diacritic stripping, NFC normalization, stemming,
translation or semantic model is implied.

All fixture documents, dates and queries are invented. They are informed by
real re-finding problems, NOT collected participant data, a longitudinal study,
or a representative personal archive. No private laptop data is used.

## Implemented

ContentIndex(..., casefold=True) is an opt-in candidate search mode. Default
exact search stays unchanged. Original bytes are re-read and SHA-256 checked
before showing excerpts. Character expansion (Straße/STRASSE, ligature ffi)
maps back to complete original character spans and UTF-8 byte coordinates.
Matches may cover part of a folded expansion; the quoted original character is
shown whole. A match is a candidate, never proof of intended semantic relevance.
The GUI and planner do not yet automatically select this mode.

## Executed results

Nine hand-selected needs, each run against exact and folded modes:

| Need | Exact | Casefold |
|---|---|---|
| Lowercase project name | GAP | MET (three candidates, not one answer) |
| Uppercase German street | GAP | MET |
| Same amount in two unrelated documents | MET | MET (both candidates) |
| Exact old delivery phrase | MET | MET |
| Receipt remembered as kitchen equipment | GAP | GAP |
| Misspelled appliance | GAP | GAP |
| Ukrainian paraphrase of Russian text | GAP | GAP |
| Approved train delivery, excluding cancellation | GAP | GAP |
| Absent literal phrase | MET | MET (indexed-scope absence only) |

Counts 3/9 versus 5/9 describe ONLY this diagnostic set. They are not accuracy
estimates or proof of broad improvement. No LLM was invoked or evaluated.
In the negation case, folded mode finds both the old decision and cancellation;
retrieval does not decide which is valid. This is intentionally recorded as GAP.
Old/new decisions in this set are separate synthetic documents; genuine archive
version separation is covered by the existing personal_recall probe.

Four recall test methods passed, including the previous baseline, in 0.837 s.
Checks include source spans after Unicode expansion, literal query operators,
permission exclusion before reads, revoked scope and corruption after indexing.
The 10 existing exact-index tests are also run as the regression gate.

Run from repository root:
```sh
python3 experiments/personal_memory_1tb_v2/tests/test_human_recall.py
python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -p 'test_*recall.py' -v
python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -p test_indexed_memory.py -v
```

## Remaining human acceptance work

Wrong remembered date, unknown document type, event relationships, OCR damage,
unindexed media, changed permissions and conflicting historical decisions need
an integrated dialogue test. Success must be judged separately for candidate
recall, wrong-version selection, grounded answer, appropriate clarification and
honest incomplete coverage. No-match is not proof that a memory never existed.
Real-user evaluation and a held-out corpus are needed before tuning claims.
The next semantic experiment must invoke an actual model, give it only the user
query and permitted evidence, and preserve unanswered/ambiguous cases in results.
