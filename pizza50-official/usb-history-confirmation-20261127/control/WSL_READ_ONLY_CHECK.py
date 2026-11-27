import pathlib,json,hashlib
spec=json.loads('{"corpus": {"path": "/home/contour/glyph-corpora/pizzachili/english50m-official-20261127-v1/english.prefix-50000000.bin", "sha256": "974837f10883d6d0b4b2d4896ff9c4adc23363fced1f5f9b031f287c54c6b437", "bytes": 50000000, "range": [0, 50000000]}, "query": {"path": "/home/contour/glyph-experiments/glyph04-pizza-official-_ys8wm32/query.bin", "sha256": "b9776d7ddf459c9ad5b0e1d6ac61e27befb5e99fd62446677600d7cacef544d0", "bytes": 3, "hex": "746865"}}')
result={}
for name,d in spec.items():
 p=pathlib.Path(d["path"]); h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 result[name]={"path":str(p),"bytes":p.stat().st_size,"sha256":h.hexdigest(),"ok":p.stat().st_size==d["bytes"] and h.hexdigest()==d["sha256"]}
 if name=="query":result[name]["hex"]=p.read_bytes().hex()
print(json.dumps(result,sort_keys=True))
assert all(v["ok"] for v in result.values())
