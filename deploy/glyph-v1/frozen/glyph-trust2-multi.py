#!/usr/bin/env python3

import bisect,hashlib,json,subprocess,sys,time
from pathlib import Path

class Bad(Exception):
    pass

def req(x,msg):
    if not x:
        raise Bad(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def js(p,label):
    p=Path(p)
    req(p.is_file(),f"{label} missing")
    try:
        return json.loads(p.read_text())
    except Exception as e:
        raise Bad(f"{label} JSON: {e}")

def authenticate_root_chain(v,pin):
    roots=v/"manifests"/"roots"
    rp=roots/pin["root_name"]

    req(rp.is_file(),"pinned root missing")
    req(sha(rp)==pin["root_sha256"],"root authentication failed")

    root=js(rp,"root")
    req(root.get("format")=="GLYPH_VAULT_ROOT_MANIFEST_V0","root format")

    cur=root
    seen={rp.name}

    while cur.get("parent_root_name") is not None:
        n=cur["parent_root_name"]
        hs=cur["parent_root_sha256"]

        req(n not in seen,"root cycle")
        seen.add(n)

        pp=roots/n
        req(pp.is_file(),"parent root missing")
        req(sha(pp)==hs,"parent root authentication failed")

        cur=js(pp,"parent root")

    req(root.get("segments")==pin["segment_ids"],
        "pinned complete segment set mismatch")

    return root,rp

def authenticate_segment(v,root,sp,qsha):
    sid=sp["segment_id"]

    entries=[
        x for x in root.get("segment_entries",[])
        if x.get("segment_id")==sid
    ]
    req(len(entries)==1,f"{sid}: segment entry cardinality")

    seg=v/"segments"/sid
    mp=seg/"segment-manifest.json"

    req(mp.is_file(),f"{sid}: segment manifest missing")

    mh=sha(mp)

    req(mh==entries[0]["segment_manifest_sha256"],
        f"{sid}: manifest root binding failed")
    req(mh==sp["segment_manifest_sha256"],
        f"{sid}: manifest pin mismatch")

    m=js(mp,f"{sid}: manifest")
    req(m.get("segment_id")==sid,f"{sid}: segment id")

    # objects: authenticate full file BEFORE semantic interpretation
    orec=m["files"]["objects"]
    op=seg/orec["name"]

    req(op.is_file(),f"{sid}: objects missing")
    req(op.stat().st_size==orec["bytes"],
        f"{sid}: objects size failed")
    req(orec["bytes"]==sp["objects_bytes"],
        f"{sid}: objects pin size mismatch")
    req(sha(op)==orec["sha256"],
        f"{sid}: objects authentication failed")
    req(orec["sha256"]==sp["objects_sha256"],
        f"{sid}: objects pin mismatch")

    objects=js(op,f"{sid}: objects")
    req(objects.get("format")=="GLYPH_PERSONAL_VAULT_V1_OBJECT_MAP",
        f"{sid}: objects format")

    # RLB: canonical identity authenticated by root->manifest.
    # Interactive query authenticates compressed blocks it actually touches.
    rrec=m["files"]["rlb3x"]

    req(rrec["sha256"]==sp["canonical_rlb_sha256"],
        f"{sid}: RLB manifest/pin mismatch")
    req(rrec["bytes"]==sp["rlb_bytes"],
        f"{sid}: RLB manifest size mismatch")

    rlb=Path(sp["rlb_path"])
    req(rlb.is_file(),f"{sid}: RLB missing")
    req(rlb.stat().st_size==sp["rlb_bytes"],
        f"{sid}: RLB physical size mismatch")

    # LOC: canonical identity authenticated by root->manifest.
    # Payload is not whole-hashed here; touched pages are authenticated by reader.
    lrec=m["files"]["loc2"]

    req(lrec["sha256"]==sp["canonical_loc_sha256"],
        f"{sid}: LOC manifest/pin mismatch")
    req(lrec["bytes"]==sp["loc_bytes"],
        f"{sid}: LOC manifest size mismatch")

    loc=Path(sp["loc_path"])
    req(loc.is_file(),f"{sid}: LOC missing")
    req(loc.stat().st_size==sp["loc_bytes"],
        f"{sid}: LOC physical size mismatch")

    rank=Path(sp["rank_path"])
    authloc=Path(sp["authloc_path"])

    req(sha(rank)==sp["rank_sha256"],
        f"{sid}: rank sidecar authentication failed")
    req(sha(authloc)==sp["authloc_sha256"],
        f"{sid}: AUTHLOC authentication failed")

    return objects,rlb,rank,loc,authloc

def map_hits(objects,offsets,plen,sid):
    objs=objects["objects"]
    starts=[int(o["offset"]) for o in objs]

    valid=[]
    rejected=0

    for pos in offsets:
        i=bisect.bisect_right(starts,pos)-1

        if i<0:
            rejected+=1
            continue

        o=objs[i]
        start=int(o["offset"])
        n=int(o["bytes"])

        if pos>=start and pos+plen<=start+n:
            valid.append({
                "segment_id":sid,
                "corpus_offset":pos,
                "object_id":o["id"],
                "object_offset":pos-start,
                "path":o["path"],
                "pattern_bytes":plen,
            })
        else:
            rejected+=1

    return valid,rejected

def main():
    if len(sys.argv)!=5 or sys.argv[1]!="search":
        raise SystemExit(
            "usage: glyph-trust2-multi.py search VAULT PIN PATTERN"
        )

    v=Path(sys.argv[2]).resolve()
    pinp=Path(sys.argv[3]).resolve()
    pattern=sys.argv[4]
    pb=pattern.encode("utf-8")

    if not pb:
        raise SystemExit("empty pattern")

    t0=time.perf_counter_ns()

    try:
        pin=js(pinp,"readpath pin")

        req(
            pin.get("format")=="GLYPH_LAPTOP_AUTH_READPATH_PIN_V2_MULTI",
            "readpath pin format"
        )

        q=Path(pin["query_reader"])
        req(q.is_file(),"query reader missing")
        req(sha(q)==pin["query_reader_sha256"],
            "query reader authentication failed")

        root,rp=authenticate_root_chain(v,pin)

        by_sid={x["segment_id"]:x for x in pin["segments"]}
        req(set(by_sid)==set(pin["segment_ids"]),
            "pin segment cardinality mismatch")

        all_hits=[]
        segment_results=[]

        total_rlb_bytes=0
        total_loc_bytes=0
        verified_rlb_blocks=0
        verified_loc_pages=0

        for sid in pin["segment_ids"]:
            sp=by_sid[sid]

            objects,rlb,rank,loc,authloc=authenticate_segment(
                v,root,sp,pin["query_reader_sha256"]
            )

            cmd=[
                sys.executable,str(q),
                str(rlb),str(rank),
                str(loc),str(authloc),
                pb.hex()
            ]

            cp=subprocess.run(cmd,text=True,capture_output=True)

            req(cp.stdout.strip(),f"{sid}: query engine no output")

            try:
                r=json.loads(cp.stdout)
            except Exception as e:
                raise Bad(f"{sid}: query output JSON: {e}")

            if cp.returncode==3 or r.get("state")=="UNTRUSTED":
                raise Bad(
                    f"{sid}: authenticated query: "
                    +str(r.get("error","UNTRUSTED"))
                )

            req(cp.returncode==0,
                f"{sid}: query engine rc={cp.returncode}: "
                +cp.stderr.strip())

            raw_count=int(r["count"])
            offsets=list(map(int,r.get("offsets",[])))

            req(len(offsets)==raw_count,
                f"{sid}: not all FM occurrences located")

            valid,rejected=map_hits(
                objects,offsets,len(pb),sid
            )

            all_hits.extend(valid)

            vrb=int(r["verified_blocks"])
            vlp=int(r["verified_loc_pages"])
            arb=int(r["authenticated_bytes"])
            alb=int(r["authenticated_loc_bytes"])

            verified_rlb_blocks+=vrb
            verified_loc_pages+=vlp
            total_rlb_bytes+=arb
            total_loc_bytes+=alb

            segment_results.append({
                "segment_id":sid,
                "raw_count":raw_count,
                "valid_count":len(valid),
                "rejected_cross_object_count":rejected,
                "verified_rlb_blocks":vrb,
                "verified_rlb_block_ids":r["verified_block_ids"],
                "authenticated_rlb_bytes":arb,
                "verified_loc_pages":vlp,
                "verified_loc_page_ids":r["verified_loc_page_ids"],
                "authenticated_loc_bytes":alb,
                "query_engine_ns":r["total_ns"],
            })

        # Global absence is legal ONLY after all committed segments
        # completed their authenticated query path.
        state="FOUND" if all_hits else "PROVEN_EMPTY"

        out={
            "format":"GLYPH_PERSONAL_VAULT_SEARCH_TRUST2_MULTI",
            "state":state,
            "pattern":pattern,
            "pattern_hex":pb.hex(),

            "committed_segments_authenticated":len(segment_results),
            "segment_results":segment_results,

            "valid_count":len(all_hits),
            "hits":all_hits,

            "verified_rlb_blocks_total":verified_rlb_blocks,
            "verified_loc_pages_total":verified_loc_pages,
            "authenticated_rlb_bytes_total":total_rlb_bytes,
            "authenticated_loc_bytes_total":total_loc_bytes,

            "authenticated_root":rp.name,
            "authenticated_root_sha256":pin["root_sha256"],

            "trust_law":
              "UNTRUSTED dominates; PROVEN_EMPTY only after every "
              "committed segment proves authenticated absence",

            "total_ns":time.perf_counter_ns()-t0,
        }

        print(json.dumps(out,ensure_ascii=False,sort_keys=True))

    except Bad as e:
        print(json.dumps({
            "format":"GLYPH_PERSONAL_VAULT_SEARCH_TRUST2_MULTI",
            "state":"UNTRUSTED",
            "pattern":pattern,
            "error":str(e),
            "total_ns":time.perf_counter_ns()-t0,
        },ensure_ascii=False,sort_keys=True))
        raise SystemExit(3)

if __name__=="__main__":
    main()
