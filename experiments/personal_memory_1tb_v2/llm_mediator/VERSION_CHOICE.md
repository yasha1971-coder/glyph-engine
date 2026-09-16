# Live version-choice probe

2026-09-16: runner implemented; actual model NOT RUN. This workspace has no
ollama, llama-server, llama_cpp, transformers or torch installation. No model
weights or inference service were provisioned. No laptop changes were made.

Five synthetic questions are sent in two candidate orders: prior decision,
latest decision, Ukrainian request, ambiguous project name and absent bank
details. Candidates include a same-name hotel and hostile instructions inside
a document. This does NOT measure real-user accuracy, twenty-year retention,
end-to-end retrieval, prompt-injection safety or industrial readiness.

The model gets only a question and candidates, never the expected decision.
No callable tools, credentials or private files are supplied. The runner accepts
only literal loopback HTTP addresses; it does not follow redirects or use proxies.
The external model server's own privacy behavior is outside this runner.

Recorded separately: server response received, reported model, duration, token
usage, schema validity, literal quote membership and expected version/action.
Literal quotation does not prove semantic support; that field is explicitly
NOT_EVALUATED. Clarification wording quality needs human review. Model weights
are not authenticated by the model name returned from a server. Exceptions are
recorded as failed cases, never counted as successful abstention.

Technical tests use authored responses to test the grader ONLY; three passed.
They are not model measurements. No live result is included in this commit.

With an already running local OpenAI-compatible model server:
```sh
python3 experiments/personal_memory_1tb_v2/llm_mediator/version_choice.py --model ACTUAL_MODEL_ID --output version-choice.json
```
The default endpoint is http://127.0.0.1:8080. --endpoint may select another
loopback port. Output must not exist. An interrupted process may leave an empty
or incomplete report; do not treat it as a completed measurement. Run sequentially;
ten calls each have a socket timeout, not a hard whole-process deadline.
Candidates are synthetic text labels, not cryptographically bound archive views.
Integration must bind selected IDs to host-authorized immutable archive versions
and independently verify excerpts before displaying them as archive evidence.
