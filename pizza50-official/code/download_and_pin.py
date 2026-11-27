#!/usr/bin/env python3
"""Download the official public corpus and pin a decimal 50MB raw prefix."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import urllib.request

SOURCE_PAGE = "https://pizzachili.dcc.uchile.cl/texts/nlang/"
SOURCE_URL = SOURCE_PAGE + "english.50MB.gz"
PREFIX_BYTES = 50_000_000


def main():
    work = Path(sys.argv[1]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    status_file = work / "DOWNLOAD_STATUS.json"
    status = {"phase": "STARTED", "pid": os.getpid(), "started_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source_page": SOURCE_PAGE,
        "requested_url": SOURCE_URL, "performance_queries_started": 0}

    def save():
        temp = work / "DOWNLOAD_STATUS.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temp.replace(status_file)
        print(json.dumps({"phase": status["phase"], "pid": os.getpid()}), flush=True)

    save()
    try:
        compressed = work / "english.50MB.gz"
        raw_file = work / "english.50MB"
        prefix = work / "english.prefix-50000000.bin"
        if any(p.exists() for p in (compressed, raw_file, prefix)):
            raise FileExistsError("refusing to replace existing corpus files")
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "GLYPH-local-calibration/1"})
        status["phase"] = "DOWNLOADING"
        save()
        compressed_hash = hashlib.sha256()
        compressed_bytes = 0
        with urllib.request.urlopen(request, timeout=45) as response, compressed.open("xb") as out:
            status["response"] = {"status": response.status, "final_url": response.url,
                                  "headers": dict(response.headers.items())}
            if response.status != 200:
                raise RuntimeError("unexpected HTTP response: " + str(response.status))
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                compressed_hash.update(block)
                compressed_bytes += len(block)
        status["compressed"] = {"path": str(compressed), "bytes": compressed_bytes,
                                "sha256": compressed_hash.hexdigest()}
        status["phase"] = "GZIP_TEST"
        save()
        argv = ["gzip", "-t", "--", str(compressed)]
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        (work / "gzip-test.stdout").write_bytes(result.stdout)
        (work / "gzip-test.stderr").write_bytes(result.stderr)
        status["gzip_test"] = {"argv": argv, "exit_code": result.returncode,
                               "stdout_file": "gzip-test.stdout", "stderr_file": "gzip-test.stderr"}
        if result.returncode:
            raise RuntimeError("gzip integrity check failed, exit " + str(result.returncode))
        status["phase"] = "DECOMPRESS_AND_PIN"
        save()
        full_hash, prefix_hash = hashlib.sha256(), hashlib.sha256()
        full_bytes = prefix_bytes = prefix_nuls = 0
        with gzip.open(compressed, "rb") as src, raw_file.open("xb") as full, prefix.open("xb") as chosen:
            while True:
                block = src.read(1024 * 1024)
                if not block:
                    break
                full.write(block)
                full_hash.update(block)
                full_bytes += len(block)
                take = block[:max(0, PREFIX_BYTES - prefix_bytes)]
                if take:
                    chosen.write(take)
                    prefix_hash.update(take)
                    prefix_bytes += len(take)
                    prefix_nuls += take.count(b"\0")
        if prefix_bytes != PREFIX_BYTES:
            raise RuntimeError("uncompressed file shorter than required prefix")
        reread_hash = hashlib.sha256()
        with prefix.open("rb") as src:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                reread_hash.update(block)
        if reread_hash.hexdigest() != prefix_hash.hexdigest():
            raise RuntimeError("prefix readback SHA-256 mismatch")
        status["decompressed"] = {"path": str(raw_file), "bytes": full_bytes,
                                  "sha256": full_hash.hexdigest()}
        status["prefix"] = {"path": str(prefix), "bytes": prefix_bytes,
                            "range": [0, PREFIX_BYTES], "sha256": prefix_hash.hexdigest(),
                            "readback_sha256": reread_hash.hexdigest(), "nul_bytes": prefix_nuls}
        for p in (compressed, raw_file, prefix):
            p.chmod(0o444)
        status["phase"] = "PASS"
        status["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status["exit_code"] = 0
        save()
        return 0
    except BaseException as error:
        status["failed_at_phase"] = status["phase"]
        status["phase"] = "FAIL"
        status["error"] = str(error)
        status["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status["exit_code"] = 1
        save()
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
