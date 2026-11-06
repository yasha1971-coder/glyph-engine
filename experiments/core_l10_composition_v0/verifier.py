#!/usr/bin/env python3
"""Independent CORE-L10 append-layout verifier, no producer imports.

Parses and hashes a separately pinned research root, authenticates every leaf,
independently inverts original RLB2 via original frozen decoder and a separate
LF implementation, checks EF sample geometry against an independently derived
small suffix array, then scans all document bytes for exact query coordinates.

This is O(total source bytes + temporary suffix comparison work) for tiny
fixtures, not a succinct proof nor historical Composition V1 operator replay.
"""
import argparse
import hashlib
import json
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA="GLYPH_CORE_L10_COMPOSED_EXACT_RESEARCH_V0"
RESULT_SCHEMA="GLYPH_CORE_L10_EXACT_RESULT_V0"
LAYOUT_DOMAIN=b"GLYPH_CORE_L10_LAYOUT_V0\x00"
SNAPSHOT_DOMAIN=b"GLYPH_CORE_L10_SNAPSHOT_V0\x00"
RUNTIME_DOMAIN=b"GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1\x00"
MANIFEST_DOMAIN=b"GLYPH_OPERATOR_CORPUS_MANIFEST_V1\x00"
FROZEN="c33cb493bc75f13ee4a904fabb190a29a196c94b"
INDEX=("rlb2","rlr2","ef")
HEX=set("0123456789abcdef")
MAX_DOCS=64
MAX_BYTES_PER_DOC=4096

def check(ok,why):
    if not ok:raise ValueError(why)

def canonical(o):
    return (json.dumps(o,sort_keys=True,separators=(",",":"),
                       ensure_ascii=True,allow_nan=False)+"\n").encode("ascii")

def h(raw):
    return hashlib.sha256(raw).hexdigest()

def hfile(path):
    sha=hashlib.sha256()
    with path.open("rb") as file:
        for b in iter(lambda:file.read(1<<18),b""):sha.update(b)
    return sha.hexdigest()

def u64(x):
    check(type(x) is int and 0<=x<(1<<64),"not unsigned u64")
    return x.to_bytes(8,"big")

def valid_hash(s):
    return type(s) is str and len(s)==64 and set(s)<=HEX

def pairs(items):
    d={}
    for k,v in items:
        check(k not in d,"duplicate JSON key")
        d[k]=v
    return d

def read_json(path,cap):
    check(path.exists() and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode),
          "missing/nonregular root or receipt")
    check(0<path.stat().st_size<=cap,"JSON admission cap")
    raw=path.read_bytes()
    obj=json.loads(raw.decode("ascii"),object_pairs_hook=pairs,
                   parse_constant=lambda s:(_ for _ in ()).throw(ValueError("nonfinite JSON")))
    check(type(obj) is dict and raw==canonical(obj),"noncanonical JSON")
    return obj,raw

def check_root(root,pin,tree):
    check(valid_hash(pin) and h(canonical(root))==pin,"external root digest mismatch")
    check(set(root)=={"schema","complete","version","parent_root_sha256",
                     "runtime_corpus_id","source_manifest_id",
                     "research_snapshot_id","research_layout_id","documents","groups"},
          "unrecognized root schema")
    check(root["schema"]==SCHEMA and root["complete"] is True and
          root["version"] in ("V0","V1"),"wrong version/incomplete snapshot")
    check((root["version"]=="V0" and root["parent_root_sha256"] is None) or
          (root["version"]=="V1" and valid_hash(root["parent_root_sha256"])),
          "invalid append predecessor identity")
    docs=root["documents"]
    groups=root["groups"]
    check(type(docs) is list and 1<=len(docs)<=MAX_DOCS
          and type(groups) is list and len(groups)>0,"invalid coverage dimensions")
    flat=[]
    for g in groups:
        check(type(g) is list and g,"empty/non-array block")
        flat.extend(g)
    check(flat==list(range(len(docs))),"missing/duplicated/reordered document coverage")
    seen_path=[]
    oracle=[]
    for i,rec in enumerate(docs):
        check(type(rec) is dict and
              set(rec)=={"doc_id","path_hex","bytes","sha256","kind","artifacts"},
              "invalid leaf record field set")
        check(type(rec["doc_id"]) is int and rec["doc_id"]==i,
              "unstable or duplicated global document ID")
        check(type(rec["bytes"]) is int and 0<=rec["bytes"]<=MAX_BYTES_PER_DOC,
              "invalid source length")
        check(valid_hash(rec["sha256"]),"bad source sha256")
        check(type(rec["path_hex"]) is str and len(rec["path_hex"])%2==0
              and set(rec["path_hex"])<=HEX,"bad canonical raw path")
        path=bytes.fromhex(rec["path_hex"])
        check(path==("vault/%08d.bin"%i).encode("ascii"),
              "nonmonotonic/unsafe append path or renumbered ID")
        seen_path.append(path)
        if rec["kind"]=="explicit-empty-document":
            check(rec["bytes"]==0 and rec["sha256"]==h(b"")
                  and rec["artifacts"]=={},"empty document was elided or forged")
            oracle.append(b"")
        else:
            check(rec["kind"]=="indexed-document" and rec["bytes"]>0
                  and type(rec["artifacts"]) is dict
                  and set(rec["artifacts"])==set(INDEX),"missing physical indexed leaf")
            leaf={}
            for k in INDEX:
                meta=rec["artifacts"][k]
                check(type(meta) is dict and set(meta)=={"path","bytes","sha256"}
                      and valid_hash(meta["sha256"])
                      and type(meta["bytes"]) is int and 0<meta["bytes"]<1<<20,
                      "invalid physical component")
                check(meta["path"]==("leaves/d%03d/"%i)+k,
                      "unsafe or mismatched immutable component path")
                file=tree/meta["path"]
                check(file.is_file() and not file.is_symlink() and
                      file.stat().st_size==meta["bytes"]
                      and hfile(file)==meta["sha256"],
                      "missing/mutated pinned leaf component: "+meta["path"])
                leaf[k]=file
            oracle.append(leaf)
    check(seen_path==sorted(set(seen_path)),"duplicate/reordered source paths")
    corpus=bytearray(RUNTIME_DOMAIN+u64(len(docs)))
    paths=bytearray(MANIFEST_DOMAIN+u64(len(docs)))
    for i,rec in enumerate(docs):
        raw_path=seen_path[i]
        record=u64(i)+u64(rec["bytes"])+bytes.fromhex(rec["sha256"])
        corpus.extend(record)
        paths.extend(u64(i)+u64(len(raw_path))+raw_path+
                     u64(rec["bytes"])+bytes.fromhex(rec["sha256"]))
    check(root["runtime_corpus_id"]==h(corpus),
          "normative ordered runtime corpus identity mismatch")
    check(root["source_manifest_id"]==h(paths),
          "normative global source manifest identity mismatch")
    key=[{"doc_id":rec["doc_id"],"path_hex":rec["path_hex"],
          "bytes":rec["bytes"],"sha256":rec["sha256"]} for rec in docs]
    check(root["research_snapshot_id"]==h(SNAPSHOT_DOMAIN+canonical(key)),
          "research snapshot binding mismatch")
    layout={"runtime_corpus_id":root["runtime_corpus_id"],
            "source_manifest_id":root["source_manifest_id"],
            "groups":groups,"leaf_artifacts":[x["artifacts"] for x in docs]}
    check(root["research_layout_id"]==h(LAYOUT_DOMAIN+canonical(layout)),
          "research physical layout binding mismatch")
    return oracle

def independent_inverse(bwt_path,record):
    data=bwt_path.read_bytes()
    check(len(data)>=56,"truncated reconstructed BWT")
    try:
        magic,version,n,rows,alphabet,sentinel,width,plen,fnv=struct.unpack_from(
            "<8sIQQIIIQQ",data)
    except struct.error as e:
        raise ValueError("invalid frozen BWT header") from e
    check(magic==b"GLYBWT1\x00" and version==1 and rows==n+1
          and n==record["bytes"] and alphabet==257 and sentinel==256
          and width==16 and plen==rows*2 and len(data)==56+plen,
          "wrong binary-safe BWT geometry")
    B=list(x[0] for x in struct.iter_unpack("<H",data[56:]))
    check(len(B)==rows and B.count(256)==1,"wrong sentinel cardinality")
    freq=[0]*257
    for c in B:
        check(0<=c<=256,"invalid alphabet symbol")
        freq[c]+=1
    C=[0]*257
    cursor=1
    for c in range(256):
        C[c]=cursor
        cursor+=freq[c]
    check(cursor==rows,"BWT histogram invalid")
    seen=[0]*257
    lf=[]
    for c in B:
        lf.append(0 if c==256 else C[c]+seen[c])
        seen[c]+=1
    source=bytearray(n)
    row=0
    for p in range(n-1,-1,-1):
        c=B[row]
        check(c!=256,"premature logical terminator")
        source[p]=c
        row=lf[row]
    check(B[row]==256 and lf[row]==0,"LF terminal cycle invalid")
    payload=bytes(source)
    check(h(payload)==record["sha256"],"independent source SHA mismatch")
    return payload

def getbits(b,pos,width):
    if width==0:return 0
    start=pos>>3
    size=(pos%8+width+7)//8
    return (int.from_bytes(b[start:start+size],"little")>>(pos%8))&((1<<width)-1)

def check_ef(ef_path,source):
    raw=ef_path.read_bytes()
    check(len(raw)>=80,"invalid physical EF")
    magic,n,step,m,hb,low,qwidth,stride,lbs,hbs,qbs,cbs,flags=struct.unpack_from(
        "<8sQQQQBBHQQQQI",raw)
    check(magic==b"GL3EF001" and n==len(source)+1 and step==2 and
          m==(n-1)//step+1 and stride==256 and flags==0,
          "EF header/policy/geometry mismatch")
    check(low==(n//m).bit_length()-1 and qwidth==(m-1).bit_length()
          and hb==((n-1)>>low)+m+1,"EF noncanonical widths")
    check(lbs==(m*low+7)//8 and hbs==(hb+7)//8 and
          qbs==(m*qwidth+7)//8 and cbs==4*((m+255)//256)
          and len(raw)==80+lbs+hbs+qbs+cbs,"EF section lengths")
    i=80
    lows=raw[i:i+lbs];i+=lbs
    highs=raw[i:i+hbs];i+=hbs
    q=raw[i:i+qbs];i+=qbs
    cp=raw[i:i+cbs]
    for section,bits in ((lows,m*low),(highs,hb),(q,m*qwidth)):
        if bits%8:
            check(section[-1]>>(bits%8)==0,"EF noncanonical bit padding")
    mapped=tuple(c+1 for c in source)+(0,)
    SA=sorted(range(n),key=lambda p:mapped[p:])
    expected={row:sa//step for row,sa in enumerate(SA) if sa%step==0}
    sample=0
    for hi in range(hb):
        if (highs[hi>>3]>>(hi&7))&1:
            check(sample<m,"too many EF high bits")
            row=((hi-sample)<<low)|getbits(lows,sample*low,low)
            quotient=getbits(q,sample*qwidth,qwidth)
            check(row in expected and quotient==expected[row],
                  "EF sampled suffix not equal to independently built SA")
            if sample%stride==0:
                check(struct.unpack_from("<I",cp,(sample//stride)*4)[0]==hi,
                      "EF checkpoint corruption or invalid repin")
            sample+=1
    check(sample==m and len(expected)==m,"EF missing sampled suffix row")

def restore_leaves(root,tree,frozen):
    check(subprocess.check_output(["git","-C",str(frozen),"rev-parse","HEAD"],
          text=True).strip()==FROZEN,"wrong frozen decoder")
    result=[]
    with tempfile.TemporaryDirectory(prefix="glyph-core-l10-replay-") as temp:
        for rec in root["documents"]:
            if rec["kind"]=="explicit-empty-document":
                result.append(b"")
                continue
            leaves=rec["artifacts"]
            rlb2=tree/leaves["rlb2"]["path"]
            rlr2=tree/leaves["rlr2"]["path"]
            ef=tree/leaves["ef"]["path"]
            rank=subprocess.run([sys.executable,str(frozen/"tools/rlbwt_rank_v2.py"),
                                 "verify",str(rlb2),str(rlr2)],
                                capture_output=True,text=True,timeout=45)
            check(rank.returncode==0,"frozen rank file verification rejected")
            bwt=Path(temp)/("d%03d.bwt"%rec["doc_id"])
            decoded=subprocess.run([sys.executable,str(frozen/"tools/rlbwt_container_v2.py"),
                                    "decode",str(rlb2),str(bwt)],
                                   capture_output=True,text=True,timeout=45)
            check(decoded.returncode==0,"independent BWT decoding failed")
            content=independent_inverse(bwt,rec)
            bwt.unlink()
            check_ef(ef,content)
            result.append(content)
    return result

def scan_docs(docs,pattern):
    positions=[]
    for did,source in enumerate(docs):
        offset=source.find(pattern)
        while offset>=0:
            positions.append([did,offset])
            offset=source.find(pattern,offset+1)
    return positions

def verify(tree,root_path,pin,result_path,frozen,parent_path=None,parent_pin=None):
    root,_=read_json(root_path,256*1024)
    check(h(canonical(root))==pin,"external root pin mismatch")
    leaf_records=check_root(root,pin,tree)
    if root["version"]=="V1":
        check(parent_path is not None and valid_hash(parent_pin)
              and root["parent_root_sha256"]==parent_pin,
              "missing/untrusted parent root lineage")
        prior,_=read_json(parent_path,256*1024)
        check_root(prior,parent_pin,tree)
        check(prior["version"]=="V0" and
              root["documents"][:len(prior["documents"])]==prior["documents"]
              and len(root["documents"])>=len(prior["documents"])+3,
              "append rewrote/reordered/removed old leaf records")
    else:
        check(parent_path is None and parent_pin is None,"V0 cannot cite future root")
    sources=restore_leaves(root,tree,frozen)
    for i,leaf in enumerate(leaf_records):
        if leaf==b"":
            check(sources[i]==b"","empty doc not preserved")
    result,_=read_json(result_path,256*1024)
    check(set(result)=={"schema","root_sha256","version","research_snapshot_id",
                       "research_layout_id","runtime_corpus_id","source_manifest_id",
                       "pattern_hex","pattern_bytes","max_offsets","match_count",
                       "returned_count","offsets_complete","coordinates","excerpts",
                       "expected_blocks","verified_blocks","queried_blocks",
                       "status","reason"},"result schema mismatch")
    check(result["schema"]==RESULT_SCHEMA and result["root_sha256"]==pin
          and result["version"]==root["version"],"wrong result committed root/version")
    for key in ("research_snapshot_id","research_layout_id",
                "runtime_corpus_id","source_manifest_id"):
        check(result[key]==root[key],"result identity mismatch: "+key)
    check(type(result["pattern_hex"]) is str and
          len(result["pattern_hex"])%2==0 and set(result["pattern_hex"])<=HEX,
          "noncanonical binary query")
    pattern=bytes.fromhex(result["pattern_hex"])
    check(len(pattern)>0 and result["pattern_bytes"]==len(pattern),
          "invalid empty/truncated query")
    coordinates=scan_docs(sources,pattern)
    check(len(coordinates)<(1<<64),"global match count overflow")
    limit=result["max_offsets"]
    check(limit is None or (type(limit) is int and 0<=limit<=256),
          "wrong global limit")
    returned=coordinates if limit is None else coordinates[:limit]
    limited=len(returned)<len(coordinates)
    status="UNTRUSTED" if limited else "FOUND" if coordinates else "PROVEN_EMPTY"
    reason="LIMITED_COMPLETENESS" if limited else "FULL_PINNED_SINGLE_DOCUMENT_LEAF_COVERAGE"
    check(result["status"]==status and result["reason"]==reason,
          "false FOUND/PROVEN_EMPTY or incomplete receipt")
    check(result["match_count"]==len(coordinates)
          and result["returned_count"]==len(returned)
          and result["offsets_complete"] is (not limited)
          and result["coordinates"]==returned,
          "false count or incomplete/wrong global doc coordinate prefix")
    expected=list(range(len(root["groups"])))
    for key in ("expected_blocks","verified_blocks","queried_blocks"):
        check(result[key]==expected,"incomplete ordered group coverage")
    excerpts=result["excerpts"]
    check(type(excerpts) is list and len(excerpts)==len(returned),
          "invalid source excerpt count")
    for j,(did,off) in enumerate(returned):
        check(excerpts[j]=={"doc_id":did,"offset":off,
                            "sha256":h(pattern),"bytes":len(pattern)}
              and sources[did][off:off+len(pattern)]==pattern,
              "forged excerpt/incorrect corpus coordinate")
    return {"status":"CORE_L10_INDEPENDENT_EXACT_LAYOUT_REPLAY_PASS",
            "version":root["version"],"root_sha256":pin,
            "runtime_corpus_id":root["runtime_corpus_id"],
            "source_manifest_id":root["source_manifest_id"],
            "research_layout_id":root["research_layout_id"],
            "document_count":len(sources),"block_count":len(root["groups"]),
            "match_count":len(coordinates),"returned_count":len(returned),
            "outcome":status,"complete_cover":True,
            "source_bytes_read":sum(len(x) for x in sources),
            "reindexed_existing_leaves":False}

def main():
    p=argparse.ArgumentParser()
    for k in ("tree","root","root-sha256","result","frozen"):
        p.add_argument("--"+k,required=True)
    p.add_argument("--parent")
    p.add_argument("--parent-sha256")
    a=p.parse_args()
    try:
        out=verify(Path(a.tree),Path(a.root),a.root_sha256,Path(a.result),
                   Path(a.frozen),Path(a.parent) if a.parent else None,a.parent_sha256)
        print(json.dumps(out,sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status":"UNTRUSTED","error":str(exc)[:320]},sort_keys=True),
              file=sys.stderr)
        return 3

if __name__=="__main__":
    sys.exit(main())
