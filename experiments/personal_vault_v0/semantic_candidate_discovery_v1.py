#!/usr/bin/env python3
import argparse,json,string
from pathlib import Path
import numpy as np
from FlagEmbedding import BGEM3FlagModel

CASES=[
  {'id':'alice_sem','query':'Найди детскую фантастическую историю про девочку, белого кролика и странный мир.','path':'canterbury/alice29.txt'},
  {'id':'fields_sem','query':'Найди исходник на C, который разбирает строку на отдельные поля.','path':'canterbury/fields.c'},
  {'id':'xargs_sem','query':'Найди Unix руководство про запуск команды с аргументами, читаемыми из стандартного ввода.','path':'canterbury/xargs.1'},
  {'id':'ardEN_sem','query':'Найди пьесу Шекспира про Розалинду и Орландо в лесу.','path':'canterbury/asyoulik.txt'},
  {'id':'hardy_sem','query':'Найди английский роман о молодой женщине, ферме и нескольких поклонниках.','path':'calgary/book1'},
  {'id':'geo_sem','query':'Найди данные или текст, связанный с геофизикой и сейсмическими измерениями.','path':'calgary/geo'},
  {'id':'pascal_sem','query':'Найди исходный код программы на языке Pascal.','path':'calgary/progp'},
  {'id':'lisp_sem','query':'Найди исходный код программы на языке Lisp.','path':'calgary/progl'},
]

def textlike_bytes(b):
    if not b: return True
    if b'\x00' in b[:65536]: return False
    s=b[:65536]
    good=sum((32<=x<127) or x in (9,10,13) or x>=128 for x in s)
    controls=sum(x<32 and x not in (9,10,13) for x in s)
    return good/len(s)>=0.92 and controls/len(s)<=0.01

def decode_text(p):
    b=p.read_bytes()
    if not textlike_bytes(b): return None
    try: return b.decode('utf-8')
    except UnicodeDecodeError:
        try: return b.decode('latin-1')
        except Exception: return None

def chunks(text,size=1800,overlap=300):
    t=' '.join(text.split())
    if not t:return []
    out=[]; i=0
    while i<len(t):
        out.append(t[i:i+size])
        if i+size>=len(t): break
        i+=size-overlap
    return out

def colbert_score(q,d):
    if len(q)==0 or len(d)==0:return -1.0
    sims=np.asarray(q) @ np.asarray(d).T
    return float(np.max(sims,axis=1).mean())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--model',default='BAAI/bge-m3')
    a=ap.parse_args()
    model=BGEM3FlagModel(a.model,use_fp16=False)
    labels=[]; passages=[]; excluded=[]
    for p in sorted(x for x in a.root.rglob('*') if x.is_file() and x.name!='SHA256SUMS'):
        rel=p.relative_to(a.root).as_posix(); txt=decode_text(p)
        if txt is None:
            excluded.append(rel); continue
        cs=chunks(txt)
        if not cs: continue
        for c in cs:
            labels.append(rel); passages.append(c)
    enc=model.encode(passages,batch_size=8,max_length=1024,return_dense=True,return_sparse=True,return_colbert_vecs=True)
    results=[]; r1=r5=0
    for c in CASES:
        q=model.encode([c['query']],batch_size=1,max_length=256,return_dense=True,return_sparse=True,return_colbert_vecs=True)
        dense=np.asarray(enc['dense_vecs']) @ np.asarray(q['dense_vecs'][0])
        best={}
        for i,path in enumerate(labels):
            late=colbert_score(q['colbert_vecs'][0],enc['colbert_vecs'][i])
            score=0.55*float(dense[i])+0.45*late
            if path not in best or score>best[path]: best[path]=score
        ranked=[p for p,_ in sorted(best.items(),key=lambda kv:(-kv[1],kv[0]))]
        rank=ranked.index(c['path'])+1 if c['path'] in ranked else 10**9
        r1+=rank==1; r5+=rank<=5
        results.append({'id':c['id'],'expected':c['path'],'rank':rank,'top5':ranked[:5]})
    n=len(CASES)
    report={'format':'GLYPH_SEMANTIC_CANDIDATE_DISCOVERY_V1','model':a.model,'retrieval':'binary-filtered text + BGE-M3 dense/late-interaction hybrid','cases':n,'recall_at_1':r1/n,'recall_at_5':r5/n,'excluded_nontext':excluded,'results':results,'acceptance':{'min_recall_at_1':0.625,'min_recall_at_5':0.875},'all_checks_passed':r1>=5 and r5>=7,'important_non_claim':'Candidate discovery only; never emits FOUND. Thresholds were inherited unchanged from V0. Binary-like files are excluded from the text semantic index, but remain canonical GLYPH objects.'}
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
    if not report['all_checks_passed']: raise SystemExit(1)
if __name__=='__main__': main()
