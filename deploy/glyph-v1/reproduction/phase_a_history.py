import os
import re
import json
import hashlib
import textwrap
import datetime
from pathlib import Path

BASE  = Path(os.environ["BASE"])
REL   = Path(os.environ["REL"])
RES   = Path(os.environ["RES"])
RUN   = Path(os.environ["RUN"])
QUERY = os.environ["QUERY"]

EXPECTED_MANIFEST = \
"de3161726e336127647af4596ee5e7c0588965bb1cc21b687276d42259b0e36b"

EXPECTED_RESOLVER = \
"cefc8611ed96a490d978efc851611ca7ed4f1ab7690be0ace72687a3b5e0e6d9"

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(1024*1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

manifest=REL/"RELEASE_MANIFEST.json"

assert sha256_file(manifest)==EXPECTED_MANIFEST, \
    "RELEASE MANIFEST IDENTITY FAILURE"

assert sha256_file(RES)==EXPECTED_RESOLVER, \
    "FROZEN RESOLVER IDENTITY FAILURE"

source=RES.read_text(encoding="utf-8")

# ================================================================
# We execute the frozen source in two exact pieces:
#
# 1. Everything before results=[]:
#    imports, constants, BRIDGE, tokenizer/stemmer,
#    authentication, provenance parents, Rank/FM contexts,
#    locate(), path_family_match().
#
# 2. The existing V6.13 per-case body from:
#       for case in CASES:
#    up to, but NOT including:
#       # Oracle.
#
# The target/oracle section and historical receipt writer are NEVER run.
# No retrieval/scoring code is edited.
# ================================================================

prefix_marker="\nresults=[]\n"
if prefix_marker not in source:
    raise RuntimeError("cannot find frozen prefix boundary")

prefix=source.split(prefix_marker,1)[0]

loop_marker="for case in CASES:\n"
oracle_marker=(
    "    # ========================================================\n"
    "    # Oracle.\n"
)

if loop_marker not in source:
    raise RuntimeError("cannot find frozen case loop")

after_loop=source.split(loop_marker,1)[1]

if oracle_marker not in after_loop:
    raise RuntimeError("cannot find frozen oracle boundary")

frozen_body=after_loop.split(oracle_marker,1)[0]
frozen_body=textwrap.dedent(frozen_body)

# Isolated namespace. This executes authenticated setup only.
ns={
    "__name__":"glyph_v1_live_acceptance_phase_a",
}

t0=datetime.datetime.now(datetime.timezone.utc)

exec(
    compile(
        prefix,
        str(RES),
        "exec"
    ),
    ns,
    ns
)

setup_complete=datetime.datetime.now(datetime.timezone.utc)

# ------------------------------------------------
# Our live case has NO expected target.
# ------------------------------------------------
case={
    "id":"live_acceptance_aceapex_gpu_history",
    "query":QUERY,
}

ns["case"]=case

# Execute EXACT frozen V6.13 retrieval/scoring body.
# Stops immediately before its Oracle section.
exec(
    compile(
        frozen_body,
        str(RES)+"::<LIVE_CASE_BODY>",
        "exec"
    ),
    ns,
    ns
)

finished=datetime.datetime.now(datetime.timezone.utc)

state=ns.get("state")
winner=ns.get("winner")
shortlist=ns.get("shortlist",[])
families=ns.get("families",{})
meta_accept=bool(ns.get("meta_accept",False))
content_ms=ns.get("content_ms")
fm_report=ns.get("fm_report",{})
parents=ns.get("parents",{})

winner_name=None
winner_parent=None

if winner is not None:
    winner_parent=winner
    winner_name=parents[winner]["name"]

# ------------------------------------------------
# No automatic acceptance in this LIVE UX even if
# frozen V6.13 internally reports CANDIDATE_READY.
# We expose it, then stop for explicit human choice.
# ------------------------------------------------

print()
print("="*78)
print("LIVE ACCEPTANCE — DISCOVERY RESULT")
print("="*78)

print("FROZEN_INTERNAL_STATE =",state)
print("FROZEN_META_ACCEPT    =",meta_accept)
print("FROZEN_WINNER         =",winner_name)
print("SHORTLIST_COUNT       =",len(shortlist))

print()
print("CANDIDATES FOR HUMAN:")

for i,x in enumerate(shortlist,1):
    print()
    print(f"[{i}] {x['name']}")
    print("    parent            =",x["parent"])
    print("    metadata_families =",x["metadata_families"])
    print("    content_families  =",x["content_families"])
    print("    family_count      =",x["family_count"])
    print("    metadata_score    =",x["metadata_score"])
    print("    content_score     =",x["content_score"])
    print("    evidence_profile  =",x["evidence_profile"])

receipt={
    "format":"GLYPH_V1_LIVE_ACCEPTANCE_PHASE_A_FROZEN_V613",
    "timestamp_start_utc":t0.isoformat(),
    "timestamp_end_utc":finished.isoformat(),

    "release_manifest_sha256":
        sha256_file(manifest),

    "frozen_resolver_sha256":
        sha256_file(RES),

    "query":QUERY,

    "retrieval_source":
        "verbatim frozen V6.13 body before Oracle",

    "query_specific_bridge_added":False,
    "target_oracle_present":False,
    "target_name_embedded":False,
    "target_file_names_embedded":False,

    "families":{
        k:sorted(v)
        for k,v in families.items()
    },

    "frozen_internal_state":state,
    "frozen_meta_accept":meta_accept,
    "frozen_winner":winner_name,
    "frozen_winner_parent":winner_parent,

    "shortlist":shortlist,

    "content_ms":content_ms,
    "fm_report":fm_report,

    "human_selection_required":True,

    "qwen_used":False,
    "payload_touched":False,
    "materialized":False,

    "canonical_write_attempted":False,
    "release_write_attempted":False,

    "acceptance_status":
        "WAITING_FOR_HUMAN_SELECTION"
        if shortlist
        else "DISCOVERY_RED_INSUFFICIENT"
}

stamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

out=RUN/(
    "GLYPH_V1_LIVE_ACCEPTANCE_PHASE_A_"
    +stamp+
    ".json"
)

out.write_text(
    json.dumps(
        receipt,
        ensure_ascii=False,
        indent=2,
        sort_keys=True
    )+"\n",
    encoding="utf-8"
)

print()
print("="*78)
print("PHASE A BOUNDARY")
print("="*78)

print(
    "ACCEPTANCE_STATUS =",
    receipt["acceptance_status"]
)

print("QWEN              = NOT USED")
print("PAYLOAD           = NOT TOUCHED")
print("MATERIALIZATION   = NOT PERFORMED")
print("TARGET ORACLE     = ABSENT")
print("QUERY TUNING      = ABSENT")
print("HUMAN AUTHORITY   = REQUIRED")

print("RECEIPT            =",out)
print("RECEIPT_SHA256     =",sha256_file(out))

print()
print("STOP HERE.")

if shortlist:
    print(
        "Choose ONLY a displayed candidate number."
    )
else:
    print(
        "No candidate found. This is a real discovery RED."
    )

# ------------------------------------------------
# Close authenticated read contexts.
# ------------------------------------------------
for c in ns.get("contexts",[]):
    try:
        if c["loc"] is not None:
            c["loc"].close()
    except Exception:
        pass

    try:
        c["rank"].close()
    except Exception:
        pass

raise SystemExit(0)
