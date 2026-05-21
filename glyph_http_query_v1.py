#!/usr/bin/env python3

import asyncio
import json
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parent

DEFAULT_FM = ROOT / "examples" / "mini" / "out" / "fm.bin"
DEFAULT_BWT = ROOT / "examples" / "mini" / "out" / "bwt.bin"
SERVER_BIN = ROOT / "build" / "query_fm_server_v1"


app = FastAPI(title="GLYPH Query Protocol V1")

proc = None
lock = asyncio.Lock()


class Query(BaseModel):
    pattern_hex: str


class BatchQuery(BaseModel):
    patterns: list[str]


@app.on_event("startup")
async def startup():
    global proc

    proc = subprocess.Popen(
        [
            str(SERVER_BIN),
            str(DEFAULT_FM),
            str(DEFAULT_BWT),
            "--json",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    while True:
        line = proc.stderr.readline().strip()
        if line == "READY":
            break
        if proc.poll() is not None:
            raise RuntimeError("query server failed to start")


@app.on_event("shutdown")
async def shutdown():
    global proc

    if proc:

        try:

            if proc.stdin:
                proc.stdin.write("__EXIT__\n")
                proc.stdin.flush()
                proc.stdin.close()

            if proc.stdout:
                proc.stdout.close()

            if proc.stderr:
                proc.stderr.close()

            proc.wait(timeout=2)

        except Exception:

            proc.kill()

        finally:

            proc = None


@app.get("/health")
async def health():
    if proc is None or proc.poll() is not None:
        raise HTTPException(status_code=500, detail="engine down")
    return {"status": "ok"}


async def query_engine(pattern_hex: str):
    if proc is None or proc.poll() is not None:
        raise HTTPException(status_code=500, detail="engine down")

    proc.stdin.write(pattern_hex + "\n")
    proc.stdin.flush()

    line = await asyncio.wait_for(
        asyncio.to_thread(proc.stdout.readline),
        timeout=5.0,
    )

    try:
        return json.loads(line.strip())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="bad engine response")


@app.post("/query")
async def query(q: Query):
    async with lock:
        try:
            return await query_engine(q.pattern_hex)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="timeout")


@app.post("/query_batch")
async def query_batch(q: BatchQuery):
    async with lock:
        try:
            results = []
            for pattern_hex in q.patterns:
                obj = await query_engine(pattern_hex)
                results.append(
                    {
                        "pattern_hex": obj["pattern_hex"],
                        "interval": obj["interval"],
                        "count": obj["count"],
                        "verified": obj["verified"],
                    }
                )

            return {
                "results": results,
                "fm_version": "FMBINv2",
            }

        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="timeout")
