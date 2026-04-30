#!/usr/bin/env python3

import asyncio
import json
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

proc = None
lock = asyncio.Lock()


class Query(BaseModel):
    hex: str


@app.on_event("startup")
async def startup():
    global proc
    proc = subprocess.Popen(
        [
            "./glyph_segmented_live.py",
            "--config", "config/shards_8gb_demo.json",
            "--server-bin", "build/query_fm_server_v1"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # ждём READY
    while True:
        line = proc.stderr.readline().strip()
        if line.startswith("READY"):
            break


@app.on_event("shutdown")
async def shutdown():
    global proc
    if proc:
        proc.kill()


@app.get("/health")
async def health():
    if proc is None or proc.poll() is not None:
        raise HTTPException(status_code=500, detail="engine down")
    return {"status": "ok"}


@app.post("/query")
async def query(q: Query):
    global proc

    if proc is None or proc.poll() is not None:
        raise HTTPException(status_code=500, detail="engine down")

    async with lock:
        try:
            proc.stdin.write("HEX " + q.hex + "\n")
            proc.stdin.flush()

            line = await asyncio.wait_for(
                asyncio.to_thread(proc.stdout.readline),
                timeout=5.0
            )

            return json.loads(line.strip())

        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="timeout")
