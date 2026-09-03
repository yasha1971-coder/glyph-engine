#!/usr/bin/env python3
import argparse,bisect,json,sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import query_rlb3x_loc2 as q3


def load_objects(path):
    data=json.loads(path.read_text())
    objects=sorted(data["objects"],key=lambda o:(o["offset"],o["id"]))
    starts=[o["offset"] for o in objects]
    return objects,starts


def map_hit(objects,starts,corpus_offset,pattern_bytes):
    i=bisect.bisect_right(starts,corpus_offset)-1
    if i<0:
        return None
    o=objects[i]
    start=o["offset"]; end=start+o["bytes"]
    if corpus_offset<start or corpus_offset+pattern_bytes>end:
        return None
    return {
        "object_id":o["id"],
        "path":o["path"],
        "corpus_offset":corpus_offset,
        "object_offset":corpus_offset-start,
        "pattern_bytes":pattern_bytes,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rlb3x",type=Path,required=True)
    ap.add_argument("--locate-core",type=Path,required=True)
    ap.add_argument("--objects",type=Path,required=True)
    ap.add_argument("--pattern-hex",required=True)
    a=ap.parse_args()

    pattern=bytes.fromhex(a.pattern_hex)
    if not pattern:
        raise SystemExit("empty pattern")
    rt=q3.Runtime3X(a.rlb3x,a.locate_core)
    try:
        raw=rt.query(pattern,-1)
    finally:
        rt.close()
    if not raw["locate_offsets_complete"]:
        raise SystemExit("object-aware filtering requires complete locate offsets")

    objects,starts=load_objects(a.objects)
    valid=[]
    rejected=[]
    for off in raw["locate_offsets"]:
        hit=map_hit(objects,starts,off,len(pattern))
        if hit is None:
            rejected.append(off)
        else:
            valid.append(hit)

    out={
        "format":"GLYPH_RLB3X_OBJECT_AWARE_QUERY_V0",
        "pattern_hex":pattern.hex(),
        "raw_count":raw["count"],
        "raw_locate_offsets":raw["locate_offsets"],
        "valid_count":len(valid),
        "valid_hits":valid,
        "rejected_cross_object_count":len(rejected),
        "rejected_cross_object_offsets":rejected,
        "maximum_lf_steps":raw["maximum_lf_steps"],
        "total_lf_steps":raw["total_lf_steps"],
        "rlb2_not_used":raw["rlb2_not_used"],
        "object_boundary_filter_applied":True,
    }
    print(json.dumps(out,sort_keys=True,separators=(",",":")))

if __name__=="__main__":
    main()
