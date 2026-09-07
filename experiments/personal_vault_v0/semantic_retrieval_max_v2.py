#!/usr/bin/env python3
import argparse,json,math,re
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModel

CASES=[
  {'id':'alice_sem','query':'Найди детскую фантастическую историю про девочку, белого кролика и странный мир.','path':'canterbury/alice29.txt'},
  {'id':'fields_sem','query':'Найди исходник на C, который разбирает строку на отдельные поля.','path':'canterbury/fields.c'},
  {'id':'xargs_sem','query':'Найди Unix руководство про запуск команды с аргументами, читаемыми из стандартного ввода.','path':'canterbury/xargs.1'},
  {'id':'arden_sem','query':'Найди пьесу Шекспира про Розалинду и Орландо в лесу.','path':'canterbury/asyoulik.txt'},
  {'id':'hardy_sem','query':'Найди английский роман о молодой женщине, ферме и нескольких поклонниках.','path':'calgary/book1'},
  {'id':'geo_sem','query':'Найди данные или текст, связанный с геофизикой и сейсмическими измерениями.','path':'calgary/geo'},
  {'id':'pascal_sem','query':'Найди исходный код программы на языке Pascal.','path':'calgary/progp'},
  {'id':'lisp_sem','query':'Найди исходный код программы на языке Lisp.','path':'calgary/progl'},
]
WORD_RE=re.compile(r"[A-Za-z0-9_]+")

def textlike_bytes(b):
    if not b:return True
    s=b[:65536]
    if b'\x00' in s:return False
    good=sum((32<=x<127) or x in (9,10,13) or x>=128 for x in s)
    ctrl=sum(x<32 and x not in (9,10,13) for x in s)
    return good/max(1,len(s))>=0.92 and ctrl/max(1,len(s))<=0.01

def decode_text(p):
    b=p.read_bytes()
    if not textlike_bytes(b):return None
    for enc in ('utf-8','latin-1'):
        try:return b.decode(enc)
        except Exception:pass
    return None

def chunks(text,size=2200,overlap=350):
    t=' '.join(text.split())
    if not t:return []
    out=[]; i=0
    while i<len(t):
        out.append(t[i:i+size])
        if i+size>=len(t):break
        i+=size-overlap
    return out

def tokenize(s):return [x.lower() for x in WORD_RE.findall(s)]

def bm25_scores(query,docs,k1=1.2,b=0.75):
    toks=[tokenize(x) for x in docs]; q=tokenize(query)
    N=len(toks); avg=sum(map(len,toks))/max(1,N)
    df=Counter()
    for d in toks:
        for t in set(d):df[t]+=1
    scores=[]
    for d in toks:
        tf=Counter(d); score=0.0
        for t in q:
            n=df.get(t,0)
            if not n:continue
            idf=math.log(1+(N-n+0.5)/(n+0.5))
            f=tf[t]; denom=f+k1*(1-b+b*len(d)/max(1,avg))
            score+=idf*f*(k1+1)/denom
        scores.append(score)
    return np.asarray(scores,dtype=float)

def norm(x):
    x=np.asarray(x,dtype=float)
    if len(x)==0:return x
    lo=float(x.min()); hi=float(x.max())
    return np.zeros_like(x) if hi<=lo else (x-lo)/(hi-lo)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path)
    ap.add_argument('--embed-model',default='jinaai/jina-embeddings-v5-text-small')
    ap.add_argument('--reranker',default='jinaai/jina-reranker-v3.5')
    a=ap.parse_args()
    labels=[]; passages=[]; excluded=[]
    for p in sorted(x for x in a.root.rglob('*') if x.is_file() and x.name!='SHA256SUMS'):
        rel=p.relative_to(a.root).as_posix(); txt=decode_text(p)
        if txt is None:
            excluded.append(rel); continue
        for c in chunks(txt):
            labels.append(rel); passages.append(f'Path: {rel}\nDocument: {c}')
    emb_model=SentenceTransformer(a.embed_model,trust_remote_code=True)
    doc_vec=emb_model.encode(['Document: '+x for x in passages],batch_size=8,normalize_embeddings=True,show_progress_bar=False)
    reranker=AutoModel.from_pretrained(a.reranker,trust_remote_code=True,dtype='auto')
    reranker.eval()
    results=[]; r1=r5=0; stage1_r5=0
    for c in CASES:
        qv=emb_model.encode(['Query: '+c['query']],normalize_embeddings=True,show_progress_bar=False)[0]
        dense=np.asarray(doc_vec)@qv
        sparse=bm25_scores(c['query'],passages)
        fused=0.88*norm(dense)+0.12*norm(sparse)
        best_chunk={}
        for i,path in enumerate(labels):
            if path not in best_chunk or fused[i]>best_chunk[path][0]:best_chunk[path]=(float(fused[i]),i)
        stage1=sorted(best_chunk.items(),key=lambda kv:(-kv[1][0],kv[0]))
        stage1_paths=[x[0] for x in stage1]
        s1rank=stage1_paths.index(c['path'])+1 if c['path'] in stage1_paths else 10**9
        stage1_r5+=s1rank<=5
        # broad candidate pool: top 12 docs, each represented by its strongest semantic chunk
        pool=stage1[:12]
        pool_docs=[passages[v[1]] for _,v in pool]
        rr=reranker.rerank(c['query'],pool_docs)
        reranked=[pool[x['index']][0] for x in rr]
        rank=reranked.index(c['path'])+1 if c['path'] in reranked else 10**9
        r1+=rank==1; r5+=rank<=5
        results.append({'id':c['id'],'expected':c['path'],'stage1_rank':s1rank,'stage1_top5':stage1_paths[:5],'rerank_rank':rank,'rerank_top5':reranked[:5]})
    n=len(CASES)
    report={
      'format':'GLYPH_SEMANTIC_RETRIEVAL_MAX_V2',
      'first_stage':a.embed_model+' + small BM25 fusion + binary filtering + path metadata',
      'reranker':a.reranker,
      'cases':n,
      'stage1_recall_at_5':stage1_r5/n,
      'recall_at_1':r1/n,
      'recall_at_5':r5/n,
      'excluded_nontext':excluded,
      'results':results,
      'acceptance':{'min_recall_at_1':0.625,'min_recall_at_5':0.875},
      'all_checks_passed':r1>=5 and r5>=7,
      'important_non_claim':'Candidate discovery only. This stack cannot emit FOUND or modify canonical GLYPH state. Acceptance thresholds and frozen queries are unchanged from V0.'
    }
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
    if not report['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':main()
