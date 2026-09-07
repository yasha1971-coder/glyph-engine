#!/usr/bin/env python3

import bisect,hashlib,json,lzma,mmap,struct,sys,time
from pathlib import Path

RLB_HDR=struct.Struct("<8sIQQIIQ")
REC=struct.Struct("<QQQQ")
AH=struct.Struct("<8sIQQQIIIIQ32s32s")
LH=struct.Struct("<4sIQIIQ")

class Bad(Exception): pass

def req(x,s):
    if not x: raise Bad(s)

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.digest()

def uleb(b,p):
    v=0; sh=0
    for _ in range(10):
        req(p<len(b),"uleb eof")
        x=b[p]; p+=1
        v|=(x&127)<<sh
        if not x&128:
            req(v>0,"zero run")
            return v,p
        sh+=7
    raise Bad("uleb")

class Rank:
    def __init__(self,rp,ap):
        self.rf=open(rp,"rb")
        self.r=mmap.mmap(self.rf.fileno(),0,access=mmap.ACCESS_READ)

        (
            mg,v,self.rows,self.runs,
            self.br,self.esc,self.blocks
        )=RLB_HDR.unpack_from(self.r,0)

        req(mg==b"RLB3X001" and v==1,"rlb header")

        self.af=open(ap,"rb")
        self.a=mmap.mmap(self.af.fileno(),0,access=mmap.ACCESS_READ)

        (
            am,av,ar,aru,ab,
            abr,ae,self.width,
            self.stride,self.ncp,
            dsha,self.locsha
        )=AH.unpack_from(self.a,0)

        req(am==b"RAUTHS01" and av==1,"auth header")
        req((ar,aru,ab,abr,ae)==
            (self.rows,self.runs,self.blocks,self.br,self.esc),
            "geometry")

        de=RLB_HDR.size+self.blocks*REC.size
        req(
            hashlib.sha256(
                self.r[RLB_HDR.size:de]
            ).digest()==dsha,
            "directory auth"
        )

        self.dir=[
            REC.unpack_from(
                self.r,RLB_HDR.size+i*REC.size
            )
            for i in range(self.blocks)
        ]

        self.starts=[x[1] for x in self.dir]

        p=AH.size

        self.gc=struct.unpack_from("<257Q",self.a,p)
        p+=257*8

        self.ps=257*self.width
        self.pb=p
        p+=self.ncp*self.ps

        self.hb=p
        req(
            len(self.a)==self.hb+self.blocks*32,
            "auth size"
        )

        self.C=[0]*257
        z=1

        for s in range(256):
            self.C[s]=z
            z+=self.gc[s]

        self.C[256]=0

        req(z==self.rows and self.gc[256]==1,"counts")

        self.cache={}
        self.verified=set()
        self.decoded_blocks=0
        self.decoded_runs=0
        self.rank_calls=0
        self.lf_calls=0
        self.scanned=0
        self.authbytes=0

    def close(self):
        self.a.close(); self.af.close()
        self.r.close(); self.rf.close()

    def cp(self,checkpoint_index,s):
        off=self.pb+checkpoint_index*self.ps+s*self.width
        return struct.unpack_from(
            "<I" if self.width==4 else "<Q",
            self.a,off
        )[0]

    def eh(self,i):
        return self.a[self.hb+i*32:self.hb+(i+1)*32]

    def decode(self,i):
        if i in self.cache:
            return self.cache[i]

        rs,row,off,cl=self.dir[i]
        comp=self.r[off:off+cl]

        req(
            hashlib.sha256(comp).digest()==self.eh(i),
            f"block authentication failed {i}"
        )

        self.verified.add(i)
        self.authbytes+=cl

        try:
            raw=lzma.decompress(comp,format=lzma.FORMAT_XZ)
        except Exception as e:
            raise Bad("decompress "+str(e))

        p=0; rr=[]

        while p<len(raw):
            h=raw[p]; p+=1

            if h==self.esc:
                req(p<len(raw),"escape")
                tag=raw[p]; p+=1
                req(tag in (0,1),"tag")
                h=self.esc if tag==0 else 256

            ln,p=uleb(raw,p)
            rr.append((h,ln))

        req(
            len(rr)==min(self.br,self.runs-rs),
            "run geometry"
        )

        self.cache[i]=rr
        self.decoded_blocks+=1
        self.decoded_runs+=len(rr)
        return rr

    def block(self,pos):
        i=bisect.bisect_right(self.starts,pos)-1
        req(i>=0,"block")
        return i

    def prefix_at_block(self,bi,s):
        cpi=bi//self.stride
        start=cpi*self.stride
        ans=self.cp(cpi,s)

        for j in range(start,bi):
            for h,l in self.decode(j):
                if h==s:
                    ans+=l

        return ans

    def rank(self,s,pos):
        self.rank_calls+=1
        req(0<=pos<=self.rows,"rank pos")

        if pos==self.rows:
            return self.gc[s]

        bi=self.block(pos)
        ans=self.prefix_at_block(bi,s)
        cur=self.starts[bi]

        for h,l in self.decode(bi):
            take=min(l,pos-cur)
            if h==s:
                ans+=take
            self.scanned+=take
            cur+=take
            if cur>=pos:
                break

        req(cur==pos,"rank scan")
        return ans

    def symrank(self,pos):
        bi=self.block(pos)
        cur=self.starts[bi]

        # Need prefix for whichever symbol is found.
        local=[0]*257

        for h,l in self.decode(bi):
            if pos<cur+l:
                base=self.prefix_at_block(bi,h)
                self.scanned+=pos-cur
                return h,base+local[h]+pos-cur

            local[h]+=l
            cur+=l
            self.scanned+=l

        raise Bad("symrank")

    def lf(self,row):
        h,r=self.symrank(row)
        self.lf_calls+=1
        return self.C[h]+r

    def search(self,p):
        l=0; r=self.rows
        for s in reversed(p):
            l=self.C[s]+self.rank(s,l)
            r=self.C[s]+self.rank(s,r)
            if l>=r:
                return l,l
        return l,r

class Loc:
    LAH=struct.Struct("<8sIIQQII32s")
    LAR=struct.Struct("<II32s")

    def __init__(self,path,authpath,expected):
        # Experimental binding:
        # Rank AUTH header binds SHA256(AUTH-LOC directory).
        req(sha(authpath)==expected,"AUTH-LOC authentication failed")

        self.f=open(path,"rb")
        self.m=mmap.mmap(
            self.f.fileno(),0,
            access=mmap.ACCESS_READ
        )

        self.af=open(authpath,"rb")
        self.am=mmap.mmap(
            self.af.fileno(),0,
            access=mmap.ACCESS_READ
        )

        mg,v,self.n,self.step,self.w,self.c=LH.unpack_from(
            self.m,0
        )

        req(mg==b"LOC2" and v==1,"loc header")
        req(self.w in (4,8) and self.step>0,"loc geometry")
        req(
            len(self.m)==LH.size+self.c*self.w,
            "loc size"
        )

        (
            am,av,self.page_samples,
            an,ac,astep,self.pages,
            header_sha
        )=self.LAH.unpack_from(self.am,0)

        req(
            am==b"ALOC0001" and av==1,
            "authloc header"
        )

        req(
            (an,ac,astep)==
            (self.n,self.c,self.step),
            "authloc geometry"
        )

        req(
            hashlib.sha256(
                self.m[:LH.size]
            ).digest()==header_sha,
            "loc header authentication failed"
        )

        req(
            len(self.am)==
            self.LAH.size+self.pages*self.LAR.size,
            "authloc size"
        )

        self.verified_pages=set()

        # Bytes actually authenticated on this read path.
        # Current experiment hashes the small AUTH-LOC directory once,
        # then only touched LOC payload pages.
        self.authbytes=len(self.am)+LH.size

    def close(self):
        self.am.close()
        self.af.close()
        self.m.close()
        self.f.close()

    def verify_page(self,page):
        if page in self.verified_pages:
            return

        req(
            0<=page<self.pages,
            "authloc page range"
        )

        moff=self.LAH.size+page*self.LAR.size

        pi,pn,pdigest=self.LAR.unpack_from(
            self.am,moff
        )

        req(pi==page,"authloc page id")

        sample0=page*self.page_samples

        req(
            sample0<self.c,
            "authloc sample range"
        )

        expected_n=min(
            self.page_samples,
            self.c-sample0
        )

        req(pn==expected_n,"authloc page geometry")

        off=LH.size+sample0*self.w
        size=pn*self.w

        payload=self.m[off:off+size]

        req(
            len(payload)==size,
            "loc page truncated"
        )

        req(
            hashlib.sha256(payload).digest()==pdigest,
            f"LOC2 page authentication failed {page}"
        )

        self.verified_pages.add(page)
        self.authbytes+=size

    def get(self,row):
        if row%self.step:
            return None

        i=row//self.step

        if i>=self.c:
            return None

        page=i//self.page_samples
        self.verify_page(page)

        off=LH.size+i*self.w

        return struct.unpack_from(
            "<I" if self.w==4 else "<Q",
            self.m,off
        )[0]

def main():
    rp,ap,lp,lap,hx=sys.argv[1:6]
    pat=bytes.fromhex(hx)

    t0=time.perf_counter_ns()
    r=l=None

    try:
        r=Rank(rp,ap)
        l=Loc(lp,lap,r.locsha)

        req(l.n==r.rows,"rows")

        tq=time.perf_counter_ns()

        a,b=r.search(pat)
        offs=[]

        maxlf=0
        totlf=0

        # Production-exact path: locate EVERY FM row.
        # Cost therefore scales honestly with result cardinality.
        if b>a:
            for row in range(a,b):
                cur=row
                st=0

                while True:
                    x=l.get(cur)

                    if x is not None:
                        suffix=(x+st)%l.n
                        req(suffix<r.rows-1,"terminal")
                        offs.append(suffix)
                        maxlf=max(maxlf,st)
                        totlf+=st
                        break

                    cur=r.lf(cur)
                    st+=1
                    req(st<=l.n,"lf runaway")

        qns=time.perf_counter_ns()-tq

        print(json.dumps({
            "state":"FOUND" if b>a else "PROVEN_EMPTY",
            "count":b-a,
            "fm_interval":[a,b],
            "first_offset":offs[0] if offs else None,
            "offsets":offs,
            "located_count":len(offs),
            "max_lf_steps":maxlf,
            "total_lf_steps":totlf,
            "stride":r.stride,
            "verified_blocks":len(r.verified),
            "verified_block_ids":sorted(r.verified),
            "authenticated_bytes":r.authbytes,
            "verified_loc_pages":len(l.verified_pages),
            "verified_loc_page_ids":sorted(l.verified_pages),
            "authenticated_loc_bytes":l.authbytes,
            "decoded_blocks":r.decoded_blocks,
            "decoded_runs":r.decoded_runs,
            "rank_calls":r.rank_calls,
            "lf_calls":r.lf_calls,
            "scanned_symbols":r.scanned,
            "query_ns":qns,
            "total_ns":time.perf_counter_ns()-t0
        },sort_keys=True))

    except Bad as e:
        print(json.dumps({
            "state":"UNTRUSTED",
            "error":str(e),
            "total_ns":time.perf_counter_ns()-t0
        },sort_keys=True))
        raise SystemExit(3)

    finally:
        if l: l.close()
        if r: r.close()

if __name__=="__main__":
    main()
