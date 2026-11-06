#!/usr/bin/env python3
"""GLYPH CORE-L10 research-only one-document RLB2/EF leaf and layout producer.

NOT GLYPH_COMPOSITION_ROOT_V1: these leaf packages are not historical O2
operator runtime units. All identities are explicitly partitioned by profile.
"""
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

SCHEMA="GLYPH_CORE_L10_COMPOSED_EXACT_RESEARCH_V0"
LAYOUT_DOMAIN=b"GLYPH_CORE_L10_LAYOUT_V0\x00"
SNAPSHOT_DOMAIN=b"GLYPH_CORE_L10_SNAPSHOT_V0\x00"
RUNTIME_DOMAIN=b"GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1\x00"
MANIFEST_DOMAIN=b"GLYPH_OPERATOR_CORPUS_MANIFEST_V1\x00"
FROZEN="c33cb493bc75f13ee4a904fabb190a29a196c94b"
U64_MAX=(1<<64)-1
INDEX=("rlb2","rlr2","ef")
HERE=Path(__file__).resolve().parent
EXPERIMENTS=HERE.parent
sys.path.insert(0,str(EXPERIMENTS/"loc_text_ef_v0"))
from real_enwik8_ef_physical_gate import build as build_ef, verify as verify_ef
from query_ef_research import make_runtime
from self_index_exact_range_extractor import EFByteExtractor

DOCS=(
    b"\x00banana",
    b"",
    b"bandana\xff",
    b"banana",
    b"banana",
    b"X",
    b"Y",
    b"\xff\x00banana",
    b"abracadabra",
    b"V1_NEEDLE\x00",
    b"banana banana",
    b"X Y",
)
PARTITIONS=((3,3,3),(2,4,3))
NEW_PARTITIONS=((3,3,3,3),(2,4,3,3))
QUERY_CASES=(
    b"banana",b"\x00",b"\xff",b"\xff\x00",b"ana",b"XY",b"X Y",
    b"V1_NEEDLE",b"not-present",b"Banana",b"  ",b"nana",
    b"bandanabanana",b"ana\xff",b"abracadabra",
)

def must(ok,msg):
    if not ok:raise ValueError(msg)

def canonical(obj):
    return (json.dumps(obj,sort_keys=True,separators=(",",":"),
                       ensure_ascii=True,allow_nan=False)+"\n").encode("ascii")

def h(raw):
    return hashlib.sha256(raw).hexdigest()

def hfile(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):d.update(b)
    return d.hexdigest()

def u64(v):
    must(type(v) is int and 0<=v<=U64_MAX,"invalid u64")
    return v.to_bytes(8,"big")

def docpath(i):
    return ("vault/%08d.bin"%i).encode("ascii")

def runtime_corpus_id(docs):
    data=bytearray(RUNTIME_DOMAIN+u64(len(docs)))
    for i,doc in enumerate(docs):
        data.extend(u64(i)+u64(doc["bytes"])+bytes.fromhex(doc["sha256"]))
    return h(data)

def source_manifest_id(docs):
    data=bytearray(MANIFEST_DOMAIN+u64(len(docs)))
    for i,doc in enumerate(docs):
        p=bytes.fromhex(doc["path_hex"])
        data.extend(u64(i)+u64(len(p))+p+u64(doc["bytes"])+bytes.fromhex(doc["sha256"]))
    return h(data)

def load_frozen_tool(path):
    spec=importlib.util.spec_from_file_location("core_l10_original_rlb2",path)
    must(spec is not None and spec.loader is not None,"missing frozen builder")
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

def run(args):
    p=subprocess.run([str(x) for x in args],capture_output=True,text=True,timeout=90)
    must(p.returncode==0,"frozen tool failed: "+p.stderr[-350:])
    return p

def create_leaf(store,frozen,i,source,rlbtool):
    dest=store/("d%03d"%i)
    dest.mkdir(parents=True,exist_ok=False)
    if not source:
        return {"doc_id":i,"path_hex":docpath(i).hex(),"bytes":0,"sha256":h(b""),
                "kind":"explicit-empty-document","artifacts":{}}
    mapped=tuple(x+1 for x in source)+(0,)
    sa=sorted(range(len(mapped)),key=lambda off:mapped[off:])
    symbols=[256 if off==0 else source[off-1] for off in sa]
    payload=b"".join(struct.pack("<H",x) for x in symbols)
    fnv=rlbtool.FNV_OFFSET
    for byte in payload:
        fnv=((fnv^byte)*rlbtool.FNV_PRIME)&rlbtool.U64_MASK
    bwt=dest/"construction.bwt"
    bwt.write_bytes(rlbtool.BWT_HEADER.pack(
        rlbtool.BWT_MAGIC,rlbtool.BWT_VERSION,
        len(source),len(mapped),rlbtool.ALPHABET_SIZE,
        rlbtool.LOGICAL_SENTINEL,rlbtool.SYMBOL_WIDTH_BITS,len(payload),fnv)+payload)
    safile=dest/"construction.sa"
    safile.write_bytes(bytes(60)+struct.pack("<"+"I"*len(sa),*sa))
    paths={key:dest/key for key in INDEX}
    run([sys.executable,frozen/"tools/rlbwt_container_v2.py",
         "encode",bwt,paths["rlb2"]])
    run([sys.executable,frozen/"tools/rlbwt_rank_v2.py",
         "build",paths["rlb2"],paths["rlr2"],"--rank-step","4"])
    geometry=build_ef(safile,len(sa),2,paths["ef"])
    report=verify_ef(safile,paths["ef"],geometry)
    must(report["sampled_sa_records_verified"]==(len(sa)-1)//2+1,
         "physical EF build mismatch")
    safile.unlink()
    bwt.unlink()
    return {"doc_id":i,"path_hex":docpath(i).hex(),
            "bytes":len(source),"sha256":h(source),
            "kind":"indexed-document","artifacts":{
                key:{"path":str(Path("leaves")/("d%03d"%i)/key),
                     "bytes":paths[key].stat().st_size,"sha256":hfile(paths[key])}
                for key in INDEX}}

def create_all(root,frozen):
    must(not root.exists(),"refuse existing research tree")
    root.mkdir(parents=True)
    store=root/"leaves"
    store.mkdir()
    frozen=Path(frozen).resolve()
    must(subprocess.check_output(["git","-C",str(frozen),"rev-parse","HEAD"],
         text=True).strip()==FROZEN,"wrong original binary-safe checkout")
    tool=load_frozen_tool(frozen/"tools/rlbwt_container_v2.py")
    # Historical snapshot is built FIRST. Appended leaf data does not exist.
    return [create_leaf(store,frozen,i,DOCS[i],tool) for i in range(9)]

def append_three(root,frozen,old_documents):
    frozen=Path(frozen).resolve()
    must(len(old_documents)==9 and len(list((root/"leaves").glob("d*")))==9,
         "unexpected V0 indexed leaf inventory")
    # Nine V0 document slots (eight indexed, one hashed empty slot).
    original={(p.relative_to(root).as_posix()):(p.stat().st_size,hfile(p))
              for p in (root/"leaves").rglob("*") if p.is_file()}
    tool=load_frozen_tool(frozen/"tools/rlbwt_container_v2.py")
    added=[create_leaf(root/"leaves",frozen,i,DOCS[i],tool) for i in range(9,12)]
    must(original=={(p.relative_to(root).as_posix()):(p.stat().st_size,hfile(p))
                   for p in (root/"leaves").rglob("*")
                   if p.is_file() and p.relative_to(root).as_posix() in original},
         "append modified existing stored leaf data")
    return old_documents+added,original

def groups_from_lengths(lengths):
    groups=[]
    base=0
    for length in lengths:
        must(length>0,"empty physical research group")
        groups.append(list(range(base,base+length)))
        base+=length
    return groups

def make_root(docs,version,group_lengths,parent_pin):
    groups=groups_from_lengths(group_lengths)
    must(sum(group_lengths)==len(docs),"incomplete group ranges")
    runtime=runtime_corpus_id(docs)
    manifest=source_manifest_id(docs)
    content_key=[{"doc_id":x["doc_id"],"path_hex":x["path_hex"],
                  "bytes":x["bytes"],"sha256":x["sha256"]} for x in docs]
    snapshot=h(SNAPSHOT_DOMAIN+canonical(content_key))
    layout=h(LAYOUT_DOMAIN+canonical({
        "runtime_corpus_id":runtime,"source_manifest_id":manifest,
        "groups":groups,"leaf_artifacts":[x["artifacts"] for x in docs]}))
    return {"schema":SCHEMA,"complete":True,"version":version,
            "parent_root_sha256":parent_pin,
            "runtime_corpus_id":runtime,"source_manifest_id":manifest,
            "research_snapshot_id":snapshot,"research_layout_id":layout,
            "documents":docs,"groups":groups}

def store_root(tree,root,name):
    path=tree/(name+".json")
    path.write_bytes(canonical(root))
    return {"name":name,"path":path,"sha256":hfile(path),
            "research_snapshot_id":root["research_snapshot_id"],
            "research_layout_id":root["research_layout_id"],
            "runtime_corpus_id":root["runtime_corpus_id"],
            "source_manifest_id":root["source_manifest_id"]}

def validate_ef_select(loc):
    import struct as _struct
    encountered=0
    for pos in range(loc.highbits):
        if (loc.high[pos>>3]>>(pos&7))&1:
            if encountered%loc.stride==0:
                must(_struct.unpack_from("<I",loc.cp,4*(encountered//loc.stride))[0]==pos,
                     "EF checkpoint is invalid, including for zero-hit patterns")
            encountered+=1
    must(encountered==loc.m,"EF high-bit coverage mismatch")
    for view,bits in ((loc.lows,loc.m*loc.low),(loc.high,loc.highbits),
                      (loc.values,loc.m*loc.width)):
        if bits%8:
            must(view[-1]>>(bits%8)==0,"noncanonical EF padding")

def verify_record_files(root_dir,rec):
    for name in INDEX:
        a=rec["artifacts"][name]
        file=root_dir/a["path"]
        must(file.is_file() and not file.is_symlink() and
             file.stat().st_size==a["bytes"] and hfile(file)==a["sha256"],
             "mandatory indexed leaf changed: "+str(file))

def produce_query(root_dir,root,pin,pattern,limit,frozen):
    must(type(pattern) is bytes and len(pattern)>0,"nonempty query only")
    must(limit is None or (type(limit) is int and 0<=limit<=256),
         "unsupported global cap")
    must(h(canonical(root))==pin,"external research root pin mismatch")
    coords=[]
    blocks=[]
    for group_ordinal,ids in enumerate(root["groups"]):
        # All blocks, even those returning zero, must be counted and covered.
        for docid in ids:
            rec=root["documents"][docid]
            if rec["kind"]=="explicit-empty-document":
                must(rec["bytes"]==0 and rec["sha256"]==h(b"") and not rec["artifacts"],
                     "invalid empty document commitment")
                continue
            verify_record_files(root_dir,rec)
            paths={x:root_dir/rec["artifacts"][x]["path"] for x in INDEX}
            run([sys.executable,frozen/"tools/rlbwt_rank_v2.py",
                 "verify",paths["rlb2"],paths["rlr2"]])
            runtime=make_runtime(frozen/"tools",paths["rlb2"],paths["rlr2"],paths["ef"])
            try:
                validate_ef_select(runtime.locate)
                must(runtime.corpus_bytes==rec["bytes"],"source length mismatch")
                extractor=EFByteExtractor(runtime)
                raw,_=extractor.extract(0,rec["bytes"])
                must(h(raw)==rec["sha256"],"full leaf source SHA mismatch")
                answer=runtime.query(pattern,-1)
                offsets=answer["locate_offsets"]
                must(answer["count"]==len(offsets)
                     and offsets==sorted(set(offsets)),"noncanonical full physical locate")
                for off in offsets:
                    excerpt,_=extractor.extract(off,len(pattern))
                    must(excerpt==pattern,"false match excerpt from physical EF")
                    coords.append([docid,off])
            finally:
                runtime.close()
        blocks.append(group_ordinal)
    must(coords==sorted(coords),"noncanonical global coordinates")
    count=len(coords)
    limited=limit is not None and limit<count
    output_coords=coords if limit is None else coords[:limit]
    status="UNTRUSTED" if limited else "FOUND" if count else "PROVEN_EMPTY"
    result={
        "schema":"GLYPH_CORE_L10_EXACT_RESULT_V0",
        "root_sha256":pin,
        "version":root["version"],
        "research_snapshot_id":root["research_snapshot_id"],
        "research_layout_id":root["research_layout_id"],
        "runtime_corpus_id":root["runtime_corpus_id"],
        "source_manifest_id":root["source_manifest_id"],
        "pattern_hex":pattern.hex(),
        "pattern_bytes":len(pattern),
        "max_offsets":limit,
        "match_count":count,
        "returned_count":len(output_coords),
        "offsets_complete":not limited,
        "coordinates":output_coords,
        "excerpts":[{"doc_id":did,"offset":off,
                     "sha256":h(pattern),"bytes":len(pattern)}
                    for did,off in output_coords],
        "expected_blocks":list(range(len(root["groups"]))),
        "verified_blocks":blocks,
        "queried_blocks":blocks,
        "status":status,
        "reason":"LIMITED_COMPLETENESS" if limited else "FULL_PINNED_SINGLE_DOCUMENT_LEAF_COVERAGE",
    }
    return result

def run_fixture_compatibility_probe(frozen,reference):
    # Old reference is deliberately separate, NOT silently imported as a
    # library for the new production path. Gate invokes its standalone tests.
    must(frozen.resolve()!=reference.resolve(),"distinct historic profiles required")
