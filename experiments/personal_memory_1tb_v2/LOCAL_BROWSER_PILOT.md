# Local browser pilot

Run `python3 experiments/personal_memory_1tb_v2/vault_browser.py --help`.
Supply an existing archive path and its independently retained receipt SHA-256;
for Precomp supply the executable path and binary SHA-256. No new inventory or
compression is performed. Open the printed token URL in the same laptop browser.
Server binds only 127.0.0.1. Ctrl+C stops it. Default port 8765 is not reused if busy.

User workflow: search names/folders, inspect results, explicitly click restore
and download. Browser saves verified bytes using its normal download settings.
The archive and installed `glyph` launcher are not rewritten. Whole files up to
64 MiB are supported. At most 200 search results are displayed; refine the query.

This is a companion browser over the existing verified compression experiment,
not the completed Vault integration. No daily intake, full-text search, Qwen,
version history GUI, encryption or complete new storage ratio claim. It does not
replace frozen Phase A/B: here the human selects an exact file directly from a
pinned manifest instead of a model-generated candidate. A receipt pin establishes
identity relative to the pin, not authenticated origin or proof of currentness.

The token URL is a local capability; do not share it. Host/Origin checks, no-store,
HTML escaping, CSP and no request logging reduce browser exposure. No remote bind,
cloud calls, automatic browser opening, or model invocation. Same-machine hostile
processes are outside this pilot threat model.

Each selected decode runs in a child: 1 GiB address space, 128 MiB file-size cap,
300 CPU seconds, 330 s timeout; timeout kills its process group. These limits are
not a security sandbox, and can reject legitimate expensive files. Ordinary
decoder limitations and experimental Precomp supply-chain caveats still apply.

Validation: three HTTP tests cover rendered name escaping, forbidden Host/Origin,
token requirement, verified click download through child and corruption refusal.
Additionally the actual pinned Precomp v0.4.7 binary reconstructed a synthetic
176363-byte JPEG through this worker; SHA-256
c8e0043197b3b9fea33b65bf189b51473d12df582419b5d039ceb120c8613ad5,
codec precomp-cn. This is not a VIKA or Windows-browser replay.
