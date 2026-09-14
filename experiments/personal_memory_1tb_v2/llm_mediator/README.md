# GLYPH LLM mediator — first experiment

Goal: a person asks in natural language; a small local model proposes a search;
GLYPH returns evidence bound to file bytes and archive version. This is an
isolated research adapter, not an installed GUI feature or a completed semantic
search engine. No laptop files or private documents are included.

## Implemented

`mediator.py` accepts only search or clarification plans. Host code owns view
and permission bindings. The model cannot request deletion, arbitrary paths,
shell execution or network transmission. A JSON schema constrains generation;
independent validation rejects unknown fields, duplicate keys, unsupported actions
and oversized queries. A valid JSON schema does not prove correct interpretation.

The adapter calls existing `local_memory_bridge.evidence`, preserving archive
version, SHA-256, byte offset, excerpt, coverage and truncation information.
Rendering must treat the excerpt and clarification text as plain untrusted text.
Do not render raw HTML. An empty result applies only to the granted exact-search
scope, not to all personal memory. Damaged evidence must abort the answer.

The local HTTP client accepts literal loopback IPs, has a timeout and bounded
response, follows no redirects and uses no environment proxy. A question is sent
to the configured local model; document contents are not sent to the planner.
The operator must select a trusted locally running model server. Loopback alone
does not establish that the server itself will never forward data.

Eight deterministic tests passed (TEST_OUTPUT.txt). They test the adapter and
host-bound scope with synthetic views. They do not measure a model's language
understanding or demonstrate prompt-injection robustness of a real model.
No weights were downloaded and no live model was run in this environment.
The exact model previously downloaded on the user's laptop remains unverified.

## Live model experiment

Candidate comparison: Qwen3.5-0.8B and Qwen3.5-2B, with a compatible current
llama.cpp server. These are candidates, not a claim of iPhone performance.
Record exact weights hash/revision, quantization, runtime commit, context limit,
RAM, hardware, prompt template and cold/warm latency before comparing them.
Keep model evaluation separate from the ongoing laptop terabyte I/O test.

With a local OpenAI-compatible llama-server already running:

```bash
python3 experiments/personal_memory_1tb_v2/llm_mediator/evaluate.py --model YOUR_LOCAL_MODEL --output mediator-result.json
```

The eight synthetic RU/UA/EN prompts measure exact-phrase preservation and
clarification routing. Connection/model failures count as failed cases. The
result is a preliminary probe, not a general benchmark. All live-model scores
are pending. Expand to paraphrases, names, typos, date ranges, ambiguous versions,
unsupported media, injected instructions inside documents and absent answers.

Next gates: verified archive integration fixture; real-model comparison;
full-text index with incremental updates and permission filtering; citation
selection; optional grounded synthesis; GUI integration. Semantic recall needs
its own labelled retrieval dataset. LLM output must never synthesize original
file bytes; recovery remains GLYPH's deterministic responsibility.

## Primary sources checked 2026-09-14

- Official model card: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Model comparison table: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/09cdc2b2181c98c04e4d040a7b3cd8c4075ea484/README.md
- llama.cpp constrained output: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md

These sources support candidate selection and structured-output integration;
they do not establish correctness on GLYPH or suitability on the user's phone.
