# Local settings comparison: known diagnostic, not a held-out benchmark

Baseline version_choice.py is unchanged. This runner compares greedy non-thinking
with a requested Qwen3.5 non-thinking general-task profile, on the SAME five
questions, in two document orders. Default: 20 requests. Three repeats: 60.
Repeated cases are not independent evidence of generalization. The seed is
recorded but deterministic execution and parameter support are not guaranteed.

Profile source, checked 2026-09-16:
https://huggingface.co/Qwen/Qwen3.5-4B
The general non-thinking recommendation specifies temperature 0.7, top_p 0.8,
top_k 20, min_p 0, presence_penalty 1.5, repetition_penalty 1.
The runner sends llama.cpp's repeat_penalty spelling. Actual parameter application
must be confirmed on the deployed server; profiles are REQUESTED settings.
The source page has differing reasoning recommendations across sections: this
experiment explicitly uses the general non-thinking profile, not reasoning.

Every response, finish reason, server timings, usage, prompt and score are saved.
JSONL is exclusive-created, flushed and fsynced after each line. An interrupted
last line may be partial; only a final complete=true summary marks completion.
HTTP failures and non-stop finish reasons stop the run. Semantic failures do not.
The timeout is socket inactivity, NOT a total wall-clock deadline.
No external hosts, proxies or redirects; only a literal loopback HTTP origin.
No private data, archive writes, deletion tools or model downloads.

Alternating profile order reduces systematic order advantage; cache state remains
uncontrolled. This is NOT a cold-start latency comparison. No model hash is
authenticated. A correct literal quote can still support the wrong decision.

Run beside version_choice.py with an already running local server:

```sh
python3 compare_settings.py --output /absolute/new-report.jsonl
```

Technical tests use mocked/authored responses, NOT real model inference:

```sh
python3 -m unittest discover -s . -p 'test_compare_settings.py' -v
```

Next gate: verify runtime/weights provenance, compare repeated settings, then
freeze an independent synthetic evaluation set before changing the prompt or
adding temporal extraction. This commit does not implement that architecture,
measure end-to-end retrieval, or claim improved model accuracy.
