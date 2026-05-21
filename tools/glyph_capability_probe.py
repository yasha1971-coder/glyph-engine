#!/usr/bin/env python3

import json
import os
import platform
import subprocess

def run(x):
    try:
        return subprocess.check_output(
            x,
            shell=True,
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        return "unknown"

obj = {}

obj["cpu_arch"] = platform.machine()

obj["runtime_os"] = platform.system()

obj["runtime_kernel"] = platform.release()

obj["compiler_family"] = "gcc"

obj["compiler_version"] = run("g++ --version | head -n1")

flags = run(
    "grep -m1 flags /proc/cpuinfo"
)

simd = []

for f in [
    "avx2",
    "avx512f",
    "sse4_2",
]:
    if f in flags:
        simd.append(f)

obj["simd_capability"] = simd

obj["query_protocol_version"] = \
"GLYPH_QUERY_PROTOCOL_V1"

obj["server_protocol_version"] = \
"GLYPH_SERVER_PROTOCOL_V1"

obj["http_protocol_version"] = \
"GLYPH_HTTP_QUERY_V1"

obj["retrieval_contract_version"] = \
"GLYPH_RETRIEVAL_CONTRACT_V1"

obj["capability_contract_version"] = \
"GLYPH_CAPABILITY_CONTRACT_V1"

print(
    json.dumps(
        obj,
        sort_keys=True,
        indent=2
    )
)
