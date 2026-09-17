# Personal recall: executable baseline, 2026-09-15

The product goal is finding the right original version from an imperfect memory.
Exact substring retrieval alone does not meet that goal. This probe uses two
real compressed GLYPH archives made from synthetic text. The same opaque filename
contains an old train-delivery decision and a newer air-delivery decision. Dates
2006/2026 are text in fixtures, NOT evidence of twenty years of retention.

Run from repository root:

```sh
python3 experiments/personal_memory_1tb_v2/tests/test_personal_recall.py
python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -p test_personal_recall.py -v
```

Measured baseline: three literal controls MET, three human-recall cases GAP.
The exact old phrase returns only the old version; the new phrase only the new
version; the common project name returns both. Every returned byte range is
checked against independently decoded source bytes. Both originals also match.
Paraphrase, changed letter case, and a Ukrainian paraphrase return no matches.

This is a diagnostic selection, NOT a representative accuracy percentage.
No model is called; it does not measure the existing LLM planner or a competitor.
Passing unittest means the baseline measurement is reproduced, NOT that the
product passes six acceptance scenarios. The GAP assertions intentionally lock
this exact-backend baseline; future retrieval strategies need separate results.

## Next integration decision

Preserve exact retrieval for names, numbers and citations. Add a separate
candidate-retrieval layer for normalized lexical search and semantic retrieval.
Candidates must retain archive version identity. Re-read original bytes before
showing evidence. Normalization changes offsets (e.g. Unicode casefold expansion),
so normalized offsets must never be used directly as original byte coordinates.
Ambiguous historical requests should expose competing versions or ask a question.
No match must not become a claim that the memory never existed.

Evaluate real local models against this same fixture without supplying the
expected phrase or answer to the model. Record query planning separately from
retrieval and final version selection. Do not substitute canned plans for a model
and report semantic success. PDF/OCR, audio, persistent indexes, GUI integration,
retention, backup recovery and competing applications remain unmeasured here.

The current change adds only a reproducible probe. No archive format, production
search behavior, laptop files, permissions or deletion behavior changed.
