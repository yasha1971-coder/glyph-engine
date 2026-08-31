#!/usr/bin/env python3
import argparse,json,math,re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

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

TEXT_EXT={'.txt','.html','.c','.lsp','.1',''}

def decode_text(p):
    b=p.read_bytes()
    for enc in ('utf-8','latin-1'):
        try: return b.decode(enc)
        except Exception: pass
    return ''

def chunks(text,size=1600,overlap=250):
    text=' '.join(text.split())
    if not text: return []
    out=[]; i=0
    while i<len(text):
        out.append(text[i:i+size])
        if i+size>=len(text): break
        i+=size-overlap
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--model',default='intfloat/multilingual-e5-small')
    a=ap.parse_args()
    model=SentenceTransformer(a.model)
    docs=[]; texts=[]
    for p in sorted(x for x in a.root.rglob('*') if x.is_file() and x.name!='SHA256SUMS'):
        rel=p.relative_to(a.root).as_posix()
        txt=decode_text(p)
        if not txt: continue
        for j,ch in enumerate(chunks(txt)):
            docs.append((rel,j)); texts.append('passage: '+ch)
    emb=model.encode(texts,batch_size=32,normalize_embeddings=True,show_progress_bar=False)
    results=[]; r1=r5=0
    for c in CASES:
        q=model.encode(['query: '+c['query']],normalize_embeddings=True,show_progress_bar=False)[0]
        sims=np.asarray(emb)@q
        best={}
        for idx,s in enumerate(sims):
            path,_=docs[idx]
            if path not in best or s>best[path]: best[path]=float(s)
        ranked=[p for p,_ in sorted(best.items(),key=lambda kv:(-kv[1],kv[0]))]
        rank=ranked.index(c['path'])+1 if c['path'] in ranked else 10**9
        if rank==1:r1+=1
        if rank<=5:r5+=1
        results.append({'id':c['id'],'expected':c['path'],'rank':rank,'top5':ranked[:5]})
    report={'format':'GLYPH_SEMANTIC_CANDIDATE_DISCOVERY_V0','model':a.model,'cases':len(CASES),'recall_at_1':r1/len(CASES),'recall_at_5':r5/len(CASES),'results':results,'acceptance':{'min_recall_at_1':0.625,'min_recall_at_5':0.875},'all_checks_passed':r1>=5 and r5>=7,'important_non_claim':'This is candidate discovery only. It never emits FOUND and therefore cannot override GLYPH evidence. Queries are frozen human-style paraphrases over the pinned 30-object corpus.'}
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
    if not report['all_checks_passed']: raise SystemExit(1)
if __name__=='__main__': main()
