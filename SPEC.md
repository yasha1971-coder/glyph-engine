# GLYPH Exact Retrieval Layer — SPEC v0.1
## 1. Purpose
Deterministic exact retrieval from raw bytes.

query → anchors → FM → chunk_ids

---
## 2. Outcomes (Contract)
| Outcome                 | Meaning |
|------------------------|--------|
| EMPTY_QUERY            | query_len == 0 |
| TOO_SHORT              | query_len < frag_len |
| QUERY_TOO_LONG         | query_len > max_query_bytes |
| NON_SELECTIVE          | no anchors passed entropy filter |
| NO_HIT                 | anchors selected but FM returned 0 |
| EXACT_UNIQUE           | exactly one chunk matched |
| EXACT_MULTI            | multiple chunks matched |
| INVALID_PARTIAL_HIT    | some anchors hit, others failed |
---
## 3. Result Format

shortlist_size
total_count
truncated
shortlist_top

---
## 4. Explain Mode

accepted_anchors
dropped_by_entropy
selected_anchors
min_selected_sa_hits
max_selected_sa_hits
anchors_with_zero_hits

---
## 5. Limits / Safety
| Param                  | Default |
|------------------------|--------|
| frag_len               | 48 |
| window_step            | 64 |
| max_windows            | 64 |
| pick_k                 | 3 |
| entropy_min            | 2.0 |
| non_selective_threshold| 16 |
| max_query_bytes        | 1MB |
| limit                  | 100 |
---
## 6. Guarantees
- deterministic results
- no false positives
- bounded FM calls (≤ max_windows)
- explainable decisions
---
## 7. Known Limitations (v0.1)
- no FULL_EXACT (distance constraints)
- no scoring for EXACT_MULTI
- no index versioning
- no concurrency guarantees
- no internal timeout control
---
## 8. Test Coverage
Covered:
- empty query
- short query
- long query
- low entropy
- absent data
- exact unique
- exact multi
- partial corruption
---
## 9. Fuzz Coverage

rare_anchor_fuzz_suite_v1.py

Covers:
- adversarial queries
- mutation
- binary noise
- boundary cases
---
## 10. Future (v0.2)
- FULL_EXACT (distance constraint)
- max_hits hard cap
- index versioning
- concurrency model
- internal timeout
---
