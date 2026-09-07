import os
import json
import hashlib
import importlib.util
import datetime
import shutil
from pathlib import Path, PurePosixPath

BASE=Path(os.environ["BASE"])
REL=Path(os.environ["REL"])
VAULT=Path(os.environ["VAULT"])
PIN=Path(os.environ["PIN"])
HARD=Path(os.environ["HARD"])
RUN=Path(os.environ["RUN"])
CACHE=Path(os.environ["CACHE"])

SELECTED_PARENT=os.environ["SELECTED_PARENT"]
PHASE_A_SHA=os.environ["PHASE_A_SHA"]
RELEASE_SHA=os.environ["RELEASE_SHA"]
RESOLVER_SHA=os.environ["RESOLVER_SHA"]

def sha_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        while True:
            b=f.read(1024*1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def parent_of(path):
    p=PurePosixPath(str(path))
    par=str(p.parent)
    if par in ("",".","/"):
        return None
    return par

def object_expected_sha(o):
    # Fail closed. Accept only an explicit 64-hex digest field.
    preferred=[
        "sha256",
        "sha256_hex",
        "content_sha256",
        "object_sha256",
        "hash",
    ]

    for k in preferred:
        v=o.get(k)
        if isinstance(v,str):
            vv=v.lower().strip()
            if len(vv)==64 and all(c in "0123456789abcdef" for c in vv):
                return vv,k

    # Last resort: inspect scalar fields but only accept a literal
    # 64-hex SHA-looking value. Never manufacture one.
    for k,v in o.items():
        if isinstance(v,str):
            vv=v.lower().strip()
            if (
                len(vv)==64
                and all(c in "0123456789abcdef" for c in vv)
                and ("sha" in str(k).lower() or "hash" in str(k).lower())
            ):
                return vv,str(k)

    return None,None

manifest=REL/"RELEASE_MANIFEST.json"
resolver=REL/"components"/"hybrid_resolver_v613.py"

assert sha_file(manifest)==RELEASE_SHA, "release manifest identity failure"
assert sha_file(resolver)==RESOLVER_SHA, "resolver identity failure"

# Find exact Phase A receipt by content hash.
phase_a=None
for p in RUN.glob("GLYPH_V1_LIVE_ACCEPTANCE_PHASE_A_*.json"):
    try:
        if sha_file(p)==PHASE_A_SHA:
            phase_a=p
            break
    except Exception:
        pass

assert phase_a is not None, "bound Phase A receipt not found"

a=json.loads(phase_a.read_text(encoding="utf-8"))

assert a["human_selection_required"] is True
assert a["qwen_used"] is False
assert a["payload_touched"] is False
assert a["materialized"] is False

shortlist=a.get("shortlist",[])
assert shortlist, "Phase A shortlist empty"

# Human explicitly selected displayed #1.
selected=shortlist[0]

assert selected["name"]==SELECTED_PARENT, \
    "human selection does not match Phase A candidate #1"

print("PHASE_A_BOUND =",phase_a)
print("PHASE_A_SHA256 =",sha_file(phase_a))
print("HUMAN_SELECTED =",selected["name"])
print()

# ============================================================
# AUTHENTICATE CANONICAL OBJECT MAPS.
# ============================================================

hard=load("hard_live_b",HARD)
pin=json.loads(PIN.read_text(encoding="utf-8"))

hard.req(
    pin.get("format")=="GLYPH_LAPTOP_AUTH_READPATH_PIN_V2_MULTI",
    "pin format"
)

qpath=Path(pin["query_reader"])

hard.req(qpath.is_file(),"query reader missing")
hard.req(
    hard.sha(qpath)==pin["query_reader_sha256"],
    "query reader authentication failed"
)

root,rp=hard.authenticate_root_chain(VAULT,pin)
by_sid={x["segment_id"]:x for x in pin["segments"]}

selected_objects=[]

for sid in pin["segment_ids"]:
    sp=by_sid[sid]

    objects,rlb,rankp,locp,authlocp = hard.authenticate_segment(
        VAULT,
        root,
        sp,
        pin["query_reader_sha256"]
    )

    for idx,o in enumerate(objects["objects"]):
        if parent_of(o["path"])==SELECTED_PARENT:
            selected_objects.append({
                "sid":sid,
                "index":idx,
                "object":o,
                "rlb":str(rlb),
            })

print("AUTHENTICATED_SELECTED_OBJECTS =",len(selected_objects))

if not selected_objects:
    raise RuntimeError(
        "selected logical parent absent from authenticated object maps"
    )

for x in selected_objects:
    o=x["object"]
    print(
        f"  sid={x['sid']} object={x['index']} "
        f"bytes={o.get('bytes')} path={o.get('path')}"
    )

print()

# ============================================================
# POST-SELECTION MATERIALIZATION ONLY.
#
# Use an existing preservation-v72 corpus for each needed SID.
# Exact selected object bytes are accepted ONLY if their SHA256
# equals the digest in the authenticated canonical object map.
# ============================================================

needed_sids=sorted({x["sid"] for x in selected_objects})
missing=[]

for sid in needed_sids:
    cp=CACHE/sid/"corpus.bin"
    if not cp.is_file():
        missing.append(sid)

if missing:
    print("POST_SELECTION_LANE_BUILD_REQUIRED =",missing)
    print("PAYLOAD = NOT MATERIALIZED")
    print("FAIL_CLOSED = TRUE")
    raise SystemExit(20)

stamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
OUT=RUN/f"materialized-{stamp}"
OUT.mkdir(parents=True,exist_ok=False)

verified=[]

for x in selected_objects:
    sid=x["sid"]
    idx=x["index"]
    o=x["object"]

    off=int(o["offset"])
    size=int(o["bytes"])

    expected,sha_field=object_expected_sha(o)

    if expected is None:
        shutil.rmtree(OUT,ignore_errors=True)
        raise RuntimeError(
            f"{sid}:{idx}: authenticated object has no explicit SHA256"
        )

    corpus=CACHE/sid/"corpus.bin"

    with open(corpus,"rb") as f:
        f.seek(off)
        data=f.read(size)

    if len(data)!=size:
        shutil.rmtree(OUT,ignore_errors=True)
        raise RuntimeError(
            f"{sid}:{idx}: short preservation read "
            f"{len(data)} != {size}"
        )

    actual=hashlib.sha256(data).hexdigest()

    if actual!=expected:
        shutil.rmtree(OUT,ignore_errors=True)
        raise RuntimeError(
            f"{sid}:{idx}: OBJECT SHA MISMATCH "
            f"expected={expected} actual={actual}"
        )

    # Preserve basename but make collisions impossible.
    basename=PurePosixPath(o["path"]).name or f"object-{idx}"
    dest=OUT/f"{sid}-{idx:04d}-{basename}"
    dest.write_bytes(data)

    verified.append({
        "segment_id":sid,
        "object_index":idx,
        "canonical_path":o["path"],
        "bytes":size,
        "offset":off,
        "sha_field":sha_field,
        "expected_sha256":expected,
        "materialized_sha256":actual,
        "materialized_path":str(dest),
    })

print("="*78)
print("VERIFIED MATERIALIZED OBJECTS")
print("="*78)

for i,v in enumerate(verified,1):
    print()
    print(f"[{i}] {v['canonical_path']}")
    print("    SID     =",v["segment_id"])
    print("    OBJECT  =",v["object_index"])
    print("    BYTES   =",v["bytes"])
    print("    SHA256  =",v["materialized_sha256"])
    print("    FILE    =",v["materialized_path"])

    p=Path(v["materialized_path"])
    raw=p.read_bytes()

    # Preview is presentation only, AFTER hash verification.
    try:
        text=raw.decode("utf-8")
        text=text.replace("\r\n","\n")
        print("    --- VERIFIED TEXT PREVIEW ---")
        lines=text.splitlines()
        for line in lines[:28]:
            print("    "+line[:220])
        if len(lines)>28:
            print("    ...")
    except UnicodeDecodeError:
        print("    PREVIEW = NON-UTF8 OBJECT")

# No target filename/content oracle. The human/model can inspect
# verified objects only now, after exact SHA check.

receipt={
    "format":"GLYPH_V1_LIVE_ACCEPTANCE_PHASE_B",
    "timestamp_utc":
        datetime.datetime.now(datetime.timezone.utc).isoformat(),

    "phase_a_receipt":str(phase_a),
    "phase_a_sha256":PHASE_A_SHA,

    "release_manifest_sha256":sha_file(manifest),
    "frozen_resolver_sha256":sha_file(resolver),

    "human_selection":{
        "candidate_number":1,
        "name":SELECTED_PARENT,
        "parent":selected["parent"],
    },

    "authenticated_object_count":len(selected_objects),
    "needed_segments":needed_sids,

    "objects":verified,

    "all_object_hashes_verified":
        len(verified)==len(selected_objects)
        and len(verified)>0,

    "payload_touched_after_human_selection":True,
    "qwen_used":False,

    "canonical_mutated":False,
    "release_mutated":False,

    "status":
        "GREEN_AUTHENTICATED_MATERIALIZATION"
        if verified and len(verified)==len(selected_objects)
        else "RED"
}

receipt_path=RUN/f"GLYPH_V1_LIVE_ACCEPTANCE_PHASE_B_{stamp}.json"

receipt_path.write_text(
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
print("PHASE B RESULT")
print("="*78)
print("STATUS =",receipt["status"])
print("OBJECTS =",len(verified))
print("ALL_OBJECT_HASHES_VERIFIED =",receipt["all_object_hashes_verified"])
print("QWEN = NOT USED")
print("PAYLOAD_TOUCHED = AFTER HUMAN SELECTION ONLY")
print("CANONICAL = UNCHANGED")
print("RELEASE = UNCHANGED")
print("RECEIPT =",receipt_path)
print("RECEIPT_SHA256 =",sha_file(receipt_path))
print("MATERIALIZED_DIR =",OUT)

raise SystemExit(
    0 if receipt["status"]=="GREEN_AUTHENTICATED_MATERIALIZATION"
    else 4
)
