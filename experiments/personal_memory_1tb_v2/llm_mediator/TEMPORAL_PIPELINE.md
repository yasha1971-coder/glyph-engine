# Temporal hint pilot

2026-09-16. Implemented, no live inference performed in the development workspace.
Five known questions in two orders, paired direct/staged: up to 30 HTTP calls.
No fixture or selection oracle changes. No personal files or archive operations.

The staged condition changes BOTH the selection prompt and adds a model-generated
hint. Any improvement would support the combined treatment, not prove that a
separate intent stage caused it. A prompt-only ablation and independent fixtures
are required before a causal/generalization claim. This is NOT held-out testing.

Intent category, schema, literal anchor and final selection are graded separately.
An anchor that literally appears in the question may omit an important entity;
event completeness and semantic support require human review. Incorrect but
well-formed hints reach selection: the test does not repair them using the oracle.
Invalid hints block that staged decision and count as unsuccessful, not not_found.
No temporal criterion is not equivalent to ambiguity about the document.

All requests/responses persist as fsynced JSONL. Socket inactivity timeout 120s,
not total deadline. Fail fast on transport errors and non-stop finishes. No retry,
no proxies/redirects; literal loopback origin only. Cache uncontrolled, modes
alternate order; timings are not cold-start measurements. Weights unverified.

Run alongside compare_settings.py and original version_choice.py:

```sh
python3 temporal_pipeline.py --output /absolute/new-pipeline.jsonl
```

Technical tests are authored/mock responses, not model accuracy measurements.
The unchanged 4/10 laptop baseline is not replaced by these technical results.
