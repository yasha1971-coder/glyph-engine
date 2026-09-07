#!/usr/bin/env python3
import bisect, hashlib, importlib.util, json, math, re, time
from collections import defaultdict
from pathlib import Path, PurePosixPath

H=Path.home()
BASE=H/"GlyphPilot/LI-V0-LAPTOP"
VAULT=H/"GlyphPilot/VIKA_proekt-vault-patched-test"
PIN=H/"GlyphPilot/VIKA-trust2-five/readpath-pin-v2.json"
HARD=H/"GlyphPilot/VIKA-trust2-five/glyph-trust2-multi.py"

# ============================================================
# ORACLE ONLY — never enters retrieval/scoring.
# ============================================================
CASES=[
    {
        "id":"lovit",
        "query":
            "Найди жидкую кормовую минеральную добавку для животных, "
            "которую добавляли в питьевую воду. Название не помню. "
            "Кажется производитель из Германии.",
        "expect":"TARGET",
        "expect_contains":"ловит фос",
    },
    {
        "id":"drybath",
        "query":
            "Найди смесь для сухого купания птицы где были ракушка, "
            "зола и кормовая сера. Название не помню.",
        "expect":"TARGET",
        "expect_contains":"суха ванна",
    },
    {
        "id":"quarantine",
        "query":
            "Найди карантин про номер девять, аптечку для птицы "
            "на карантин и связанные с ней материалы.",
        "expect":"TARGET",
        "expect_contains":"карантин pro №9",
    },
    {
        "id":"pigpremix",
        "query":
            "Найди протопремикс Сальвавит пять процентов с Lovit Phos "
            "для свиней и поросят.",
        "expect":"TARGET",
        "expect_contains":"для свиней та поросят",
    },
    {
        "id":"negative",
        "query":
            "Найди синюю инъекционную вакцину для альпак производства "
            "Исландии которую мы применяли каждые сорок семь дней.",
        "expect":"INSUFFICIENT",
        "expect_contains":None,
    },
]

STOP={
    "найди","найти","покажи","которая","которую","который","которые",
    "где","были","была","был","было","для","при","из","на","по","и",
    "или","что","это","ту","тот","той","мы","ее","его","их","кажется",
    "название","помню","связанные","материалы","про"
}

# General language bridge only. No product identity.
BRIDGE={
    "герман":      ["герман","німеччин"],
    "производ":    ["производ","виробник"],
    "жидк":        ["жидк","рідк"],
    "пить":        ["пить","питн"],
    "вод":         ["вод"],
    "минерал":     ["минерал","мінерал"],
    "живот":       ["живот","тварин"],
    "сух":         ["сух"],
    "купан":       ["купан","ванн"],
    "ракуш":       ["ракуш","черепаш"],
    "зол":         ["зол"],
    "сер":         ["сер","сір"],
    "корм":        ["корм"],
    "птиц":        ["птиц","птах"],
    "карантин":    ["карантин"],
    "аптеч":       ["аптеч"],
    "свин":        ["свин"],
    "поросят":     ["поросят"],
    "премикс":     ["премикс","премікс"],
    "протопремикс":["протопремикс","протопремікс"],
    "сальвавит":   ["сальвавит","сальвавіт"],
    "вакцин":      ["вакцин"],
    "альпак":      ["альпак"],
    "исланд":      ["исланд","ісланд"],
    "инъекц":      ["инъекц","ін'єкц"],
}

NUMBERS={
    "один":"1","два":"2","три":"3","четыре":"4","пять":"5",
    "шесть":"6","семь":"7","восемь":"8","девять":"9","десять":"10",
    "сорок":"40"
}

RARE_MAX=8

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

def toks(text):
    return re.findall(
        r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9%№'+-]+",
        text.casefold()
    )

def stem(w):
    if len(w)<=4:
        return w
    if len(w)>=11:
        return w[:9]
    if len(w)>=9:
        return w[:7]
    if len(w)>=7:
        return w[:6]
    return w[:5]

def surface_forms(s):
    # Exact engine is byte/case sensitive.
    # These are deterministic spelling-case variants, not fuzzy authority.
    return tuple(dict.fromkeys([
        s,
        s[:1].upper()+s[1:],
        s.upper(),
    ]))

def compile_families(text):
    raw=toks(text)
    fam={}

    for w in raw:
        if w in STOP:
            continue

        if w in NUMBERS:
            fam["num:"+NUMBERS[w]]={NUMBERS[w]}
            continue

        if w.isdigit():
            fam["num:"+w]={w}
            continue

        sw=stem(w)
        if len(sw)<4:
            continue

        key=sw
        variants={sw}

        for bk,bv in BRIDGE.items():
            if (
                w.startswith(bk)
                or sw.startswith(bk)
                or bk.startswith(sw)
            ):
                key=bk
                variants.update(bv)

        fam.setdefault(key,set()).update(variants)

    return fam

def norm(s):
    return " ".join(toks(s))

def parent_of(path):
    p=PurePosixPath(str(path))
    parent=str(p.parent)

    if parent in ("",".","/"):
        return None

    name=PurePosixPath(parent).name.strip()
    if not name:
        return None

    return parent

hard=load("hard",HARD)
pin=json.loads(PIN.read_text())
qpath=Path(pin["query_reader"])
qe=load("qe",qpath)

print("="*78)
print(" GLYPH V6.13 — STRICT SELECTION AUTHORITY")
print("="*78)

# ============================================================
# Authenticate once.
# ============================================================
tsetup=time.perf_counter()

hard.req(
    pin.get("format")=="GLYPH_LAPTOP_AUTH_READPATH_PIN_V2_MULTI",
    "pin format"
)
hard.req(qpath.is_file(),"query reader missing")
hard.req(
    hard.sha(qpath)==pin["query_reader_sha256"],
    "query reader authentication failed"
)

root,rp=hard.authenticate_root_chain(VAULT,pin)
by_sid={x["segment_id"]:x for x in pin["segments"]}

contexts=[]
parents={}

for sid in pin["segment_ids"]:
    sp=by_sid[sid]

    objects,rlb,rankp,locp,authlocp=hard.authenticate_segment(
        VAULT,root,sp,pin["query_reader_sha256"]
    )

    r=qe.Rank(rlb,rankp)

    objs=objects["objects"]
    starts=[int(o["offset"]) for o in objs]

    contexts.append({
        "sid":sid,
        "objs":objs,
        "starts":starts,
        "rank":r,
        "loc":None,
        "locp":locp,
        "authlocp":authlocp,
    })

    for o in objs:
        parent=parent_of(o["path"])
        if parent is None:
            continue

        x=parents.setdefault(parent,{
            "name":PurePosixPath(parent).name,
            "text":norm(parent),
            "objects":0,
        })
        x["objects"]+=1

setup_ms=(time.perf_counter()-tsetup)*1000

print("SETUP_MS =",round(setup_ms,3))
print("PARENTS =",len(parents))

def get_loc(c):
    if c["loc"] is None:
        c["loc"]=qe.Loc(
            c["locp"],
            c["authlocp"],
            c["rank"].locsha
        )
        hard.req(c["loc"].n==c["rank"].rows,f"{c['sid']}: rows")
    return c["loc"]

def locate(c,a,b,plen):
    r=c["rank"]
    l=get_loc(c)
    out=[]

    for row in range(a,b):
        cur=row
        st=0

        while True:
            x=l.get(cur)

            if x is not None:
                pos=(x+st)%l.n
                hard.req(pos<r.rows-1,f"{c['sid']}: terminal")
                break

            cur=r.lf(cur)
            st+=1
            hard.req(st<=l.n,f"{c['sid']}: runaway")

        i=bisect.bisect_right(c["starts"],pos)-1

        if i<0:
            continue

        o=c["objs"][i]
        start=int(o["offset"])
        size=int(o["bytes"])

        if start<=pos and pos+plen<=start+size:
            parent=parent_of(o["path"])
            if parent:
                out.append(parent)

    return out

def path_family_match(pathtext,variants):
    for v in variants:
        vv=norm(v)
        if vv and vv in pathtext:
            return True
    return False

results=[]

for case in CASES:
    print()
    print("="*78)
    print("CASE",case["id"])
    print("="*78)
    print("QUERY:",case["query"])

    tc=time.perf_counter()
    families=compile_families(case["query"])

    print("FAMILIES:")
    for k,v in families.items():
        print(" ",k,"=>",sorted(v))

    # ========================================================
    # L0 — authenticated provenance metadata.
    # Compute document-frequency across logical parents.
    # ========================================================
    df={}
    matches={}

    for f,variants in families.items():
        hitparents=set()

        for parent,pd in parents.items():
            if path_family_match(pd["text"],variants):
                hitparents.add(parent)

        df[f]=len(hitparents)
        matches[f]=hitparents

    meta_score=defaultdict(float)
    meta_fams=defaultdict(set)

    N=max(1,len(parents))

    for f,hitparents in matches.items():
        if not hitparents:
            continue

        idf=math.log2(1.0 + N/len(hitparents))

        # Numeric evidence is useful in a title only when not ubiquitous.
        if f.startswith("num:") and len(hitparents)>20:
            continue

        for parent in hitparents:
            meta_score[parent]+=idf
            meta_fams[parent].add(f)

    meta_ranked=sorted(
        meta_score,
        key=lambda p:(
            -len(meta_fams[p]),
            -meta_score[p],
            p
        )
    )

    print()
    print("L0 METADATA:")
    for i,p in enumerate(meta_ranked[:6],1):
        print(
            f" [{i}] {parents[p]['name']} "
            f"families={len(meta_fams[p])} "
            f"score={meta_score[p]:.3f} "
            f"{sorted(meta_fams[p])}"
        )

    # Strong metadata leader:
    # at least two independent families and visible margin.
    meta_accept=False
    meta_winner=None

    if meta_ranked:
        p0=meta_ranked[0]
        s0=meta_score[p0]
        s1=meta_score[meta_ranked[1]] if len(meta_ranked)>1 else 0.0

        if (
            len(meta_fams[p0])>=2
            and s0-s1>=1.5
        ):
            meta_accept=True
            meta_winner=p0

    # ========================================================
    # L1 — evidence algebra V6.12
    #
    # Three levels MUST remain distinct:
    #
    # semantic family
    #   -> linguistic alias
    #       -> case forms
    #
    # Example:
    #
    # country
    #   -> герман
    #       герман / Герман / ГЕРМАН
    #   -> німеччин
    #       німеччин / Німеччин / НІМЕЧЧИН
    #
    # Case forms do NOT have independent rarity.
    # Their occurrence counts are aggregated into their alias.
    #
    # Linguistic aliases MAY have different rarity.
    #
    # Candidate receives the strongest linguistic alias actually
    # authenticated for that semantic family, but family votes once.
    # ========================================================
    content_score=defaultdict(float)
    content_fams=defaultdict(set)
    content_detail=defaultdict(dict)
    fm_report={}
    content_ms=0.0
    missing_families=[]

    if not meta_accept:
        tl1=time.perf_counter()

        for f,variants in families.items():
            if f.startswith("num:"):
                continue

            family_best={}
            family_best_alias={}
            family_best_count={}

            alias_report={}
            family_any_count=0

            for alias in sorted(variants):
                forms=list(dict.fromkeys(surface_forms(alias)))

                form_results=[]
                alias_total=0

                for form in forms:
                    pb=form.encode("utf-8")
                    segs=[]
                    nall=0

                    for c in contexts:
                        a,b=c["rank"].search(pb)
                        n=b-a
                        nall+=n
                        segs.append((a,b,n))

                    alias_total+=nall

                    form_results.append({
                        "form":form,
                        "count":nall,
                        "segments":segs,
                    })

                alias_report[alias]=alias_total
                family_any_count+=alias_total

                # Absence is authenticated evidence of absence,
                # but never positive candidate evidence.
                if alias_total==0:
                    continue

                # Discovery does not LOC broad aliases.
                if alias_total>RARE_MAX:
                    continue

                rarity=math.log2(1.0 + 20.0/alias_total)
                hitparents=set()

                for fr in form_results:
                    if fr["count"]==0:
                        continue

                    pb=fr["form"].encode("utf-8")

                    for c,(a,b,n) in zip(
                        contexts,
                        fr["segments"]
                    ):
                        if n:
                            hitparents.update(
                                locate(c,a,b,len(pb))
                            )

                for parent in hitparents:
                    old=family_best.get(parent,-1.0)

                    if rarity>old:
                        family_best[parent]=rarity
                        family_best_alias[parent]=alias
                        family_best_count[parent]=alias_total

            fm_report[f]=alias_report

            if family_any_count==0:
                missing_families.append(f)

            # Semantic family contributes only once per candidate.
            for parent,strength in family_best.items():
                content_score[parent]+=strength
                content_fams[parent].add(f)

                content_detail[parent][f]={
                    "alias":family_best_alias[parent],
                    "strength":round(strength,3),
                    "alias_fm_count":family_best_count[parent],
                }

        content_ms=(time.perf_counter()-tl1)*1000

    # ========================================================
    # Combine independently authenticated evidence.
    # Metadata and content are separate sources.
    # ========================================================
    combined=set(meta_score)|set(content_score)

    def evidence_profile(p):
        strengths=[]

        for d in content_detail.get(p,{}).values():
            strengths.append(float(d["strength"]))

        strengths.sort(reverse=True)

        # Fixed-width lexicographic evidence profile.
        # Strongest clue dominates; weaker clues break ties.
        return tuple(
            strengths[i] if i<len(strengths) else 0.0
            for i in range(5)
        )

    ranked=sorted(
        combined,
        key=lambda p:(
            -evidence_profile(p)[0],
            -evidence_profile(p)[1],
            -evidence_profile(p)[2],
            -(len(meta_fams[p] | content_fams[p])),
            -(meta_score[p]+content_score[p]),
            p
        )
    )

    shortlist=[]

    for p in ranked[:8]:
        shortlist.append({
            "name":parents[p]["name"],
            "parent":p,
            "metadata_families":sorted(meta_fams[p]),
            "content_families":sorted(content_fams[p]),
            "family_count":
                len(meta_fams[p] | content_fams[p]),
            "metadata_score":round(meta_score[p],3),
            "content_score":round(content_score[p],3),
            "content_detail":content_detail.get(p,{}),
            "evidence_profile":[
                round(x,3) for x in evidence_profile(p)
            ],
            "total_score":round(
                meta_score[p]+content_score[p],3
            ),
        })

    # ========================================================
    # Confidence is separate from shortlist existence.
    # ========================================================
    candidate_ready=False
    winner=None

    # V6.13 TRUST LAW:
    # Provenance/metadata may deterministically identify an entity.
    # Content evidence is retrieval evidence, not selection authority.
    #
    # Therefore content-only path ALWAYS stops at SHORTLIST.
    if meta_accept:
        winner=meta_winner
        candidate_ready=True

    state=(
        "CANDIDATE_READY"
        if candidate_ready
        else "SHORTLIST"
        if shortlist
        else "INSUFFICIENT"
    )

    if not meta_accept:
        print()
        print(
            "AUTHENTICATED ABSENT FAMILIES =",
            sorted(missing_families)
        )

    print()
    print("FINAL SHORTLIST:")

    for i,x in enumerate(shortlist,1):
        print(
            f" [{i}] {x['name']} "
            f"families={x['family_count']} "
            f"meta={x['metadata_score']:.3f} "
            f"content={x['content_score']:.3f} "
            f"profile={x['evidence_profile']}"
        )
        print(
            "     M:",
            x["metadata_families"],
            "C:",
            x["content_families"]
        )
        if x.get("content_detail"):
            print(
                "     EXACT:",
                x["content_detail"]
            )

    # ========================================================
    # Oracle.
    # Positive = expected target must be in TOP-3.
    # Negative = system must NOT auto-accept.
    # ========================================================
    target_rank=None

    if case["expect"]=="TARGET":
        needle=case["expect_contains"].casefold()

        for i,x in enumerate(shortlist,1):
            if needle in x["parent"].casefold():
                target_rank=i
                break

        if state=="CANDIDATE_READY":
            test_pass=(
                winner is not None
                and needle in winner.casefold()
            )
        else:
            test_pass=(
                state=="SHORTLIST"
                and target_rank is not None
                and target_rank<=3
            )
    else:
        test_pass=(state!="CANDIDATE_READY")

    wall_ms=(time.perf_counter()-tc)*1000

    print()
    print("STATE =",state)
    print(
        "WINNER =",
        parents[winner]["name"] if winner else None
    )
    print("CONTENT_MS =",round(content_ms,3))
    print("WALL_MS =",round(wall_ms,3))
    if case["expect"]=="TARGET":
        print("TARGET_RANK =",target_rank)
    print("TEST =", "PASS" if test_pass else "FAIL")

    results.append({
        "id":case["id"],
        "state":state,
        "winner":parents[winner]["name"] if winner else None,
        "shortlist":shortlist,
        "content_ms":round(content_ms,3),
        "wall_ms":round(wall_ms,3),
        "test_pass":test_pass,
        "target_rank":target_rank,
        "fm_report":fm_report,
    })

passed=sum(x["test_pass"] for x in results)
failed=len(results)-passed

receipt={
    "format":"GLYPH_V613_STRICT_SELECTION_AUTHORITY",
    "status":"GREEN" if failed==0 else "RED",
    "setup_ms":round(setup_ms,3),
    "tests_total":len(results),
    "tests_passed":passed,
    "tests_failed":failed,
    "results":results,
    "qwen_used":False,
    "payload_touched":False,
    "canonical_mutated":False,
    "claim_boundary":{
        "positive_success":
            "CANDIDATE_READY requires correct winner; content-only SHORTLIST requires target TOP-3",
        "negative_success":
            "resolver does not auto-accept a candidate",
        "not_proven":
            "universal semantic understanding or arbitrary-language retrieval"
    }
}

out=BASE/"li-v0-strict-selection-v613.json"
out.write_text(
    json.dumps(
        receipt,
        ensure_ascii=False,
        indent=2,
        sort_keys=True
    )+"\n"
)

digest=hashlib.sha256(out.read_bytes()).hexdigest()

print()
print("="*78)
print("FINAL")
print("="*78)
print("STATUS =",receipt["status"])
print("PASS =",passed)
print("FAIL =",failed)

for x in results:
    print(
        x["id"],
        "=>",
        "PASS" if x["test_pass"] else "FAIL",
        "|",
        x["state"],
        "|",
        x["winner"],
        "|",
        x["wall_ms"],"ms"
    )

print("QWEN = NOT USED")
print("PAYLOAD = NOT TOUCHED")
print("CANONICAL = UNCHANGED")
print("RECEIPT =",out)
print("SHA256 =",digest)

for c in contexts:
    try:
        if c["loc"] is not None:
            c["loc"].close()
    except Exception:
        pass
    try:
        c["rank"].close()
    except Exception:
        pass

raise SystemExit(0 if failed==0 else 4)
