# Selection ablation V1

Implemented 2026-09-16. No live model inference in development environment.
Uses only the existing synthetic five-case fixture in both candidate orders.
Thirty calls, three conditions, no repeated trials:

1. direct: original selector prompt.
2. instruction_only: original plus exactly the previous temporal selection rules.
3. oracle_intent: same prompt as (2), plus human-authored temporal classification.

Mode 3 uses privileged information and is NOT autonomous model performance.
It supplies neither the answer document ID nor an answer quote. It diagnoses
selection under correct temporal interpretation, not a theoretical upper bound.
No intent-generating model is called; the prior anchor gate cannot block selection.
This does not change or rescore the prior pipeline's 4/10 result.

The exact prior instruction text mentions a hint even when none is supplied in
condition 2. This deliberate fixed instruction permits isolating hint presence;
it is not an optimized production prompt. Schema/literal quote validity does not
establish semantic support. An unspecified time is not by itself ambiguity.

Cases are already known; this is not held-out evidence, prompt-injection safety,
real-user accuracy, GUI or archive retrieval. Cache is uncontrolled and mode
order rotated. Thirty calls with one trial do not establish robust causal effects.
No new models, personal data, archive mutation or external inference endpoints.
Same loopback-only transport, fail-fast, non-stop rejection, exclusive fsynced
JSONL as the settings runner. The 120s timeout is socket inactivity, not wall time.

Run beside version_choice.py, compare_settings.py, temporal_pipeline.py:

```sh
python3 selection_ablation.py --output /absolute/new-ablation.jsonl
```

Before deployment require independent scenarios and repeats. If instruction_only
matches oracle_intent, do not add another model call without evidence of benefit.
If oracle_intent still misses before/after, the tested selector remains inadequate.
