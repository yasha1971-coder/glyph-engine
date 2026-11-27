#!/usr/bin/env python3
"""Build isolated stock indexes and compare six replies to the frozen byte oracle.

This is an untimed correctness preflight, not a p95 benchmark.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def spec(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}


def main():
    work = Path(sys.argv[1]).resolve()
    status_file = work / "control/native-preflight.json"
    status = {"phase": "STARTED", "controller_pid": os.getpid(), "performance_samples": 0,
              "correctness_query_count": 0, "stages": [], "results": []}

    def save():
        temp = work / "control/native-preflight.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(status_file)

    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               LC_ALL="C", PYTHONHASHSEED="0")

    def command(stage, argv, stdin=None):
        status["phase"] = stage
        save()
        out, err = work / ("control/" + stage + ".stdout"), work / ("control/" + stage + ".stderr")
        rec = {"stage": stage, "argv": argv, "cwd": str(work),
               "stdout": str(out), "stderr": str(err)}
        if stdin is not None:
            path = work / ("control/" + stage + ".stdin")
            path.write_text(stdin, encoding="ascii")
            rec["stdin"] = str(path)
        with out.open("xb") as stdout, err.open("xb") as stderr:
            process = subprocess.Popen(argv, cwd=work, env=env,
                                       stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, text=True)
            rec["pid"] = process.pid
            status["active_command"] = rec
            save()
            process.communicate(stdin)
        rec["exit_code"] = process.returncode
        status["stages"].append(rec)
        status.pop("active_command", None)
        save()
        if process.returncode:
            raise RuntimeError(stage + " failed, exit " + str(process.returncode))
        return out.read_text()

    save()
    try:
        calibration = json.loads((work / "INPUT_LOCK.json").read_text())
        corpus = Path(calibration["corpus"]["path"])
        if digest(corpus) != calibration["corpus"]["sha256"]:
            raise ValueError("corpus SHA-256 differs before build")
        oracle_file = work / "oracle/oracle.json"
        if digest(oracle_file) != calibration["oracle"]["sha256"]:
            raise ValueError("oracle SHA-256 differs before build")
        oracle = json.loads(oracle_file.read_text())
        bundle = json.loads((work / "TOOL_BUNDLE_LOCK.json").read_text())
        for item in bundle["files"]:
            path = Path(item["path"])
            if digest(path) != item["sha256"]:
                raise ValueError("tool input differs: " + str(path))
        status["input_checks"] = "PASS"
        tool = work / "tools/glyph-producer"
        glyph_index, move_index = work / "glyph.gidx", work / "move.index"
        glyph_reader, move_reader = work / "tools/glyph-reader.native", work / "tools/move-native/engine.native"
        command("glyph-build", ["taskset", "-c", "0", sys.executable, str(tool / "glyph-search"),
                                "build", str(corpus), "--out", str(glyph_index)])
        command("move-build", ["taskset", "-c", "0", str(move_reader), "build", str(corpus), str(move_index)])
        manifest = json.loads((glyph_index / "manifest.json").read_text())
        if manifest["corpus_sha256"] != calibration["corpus"]["sha256"]:
            raise ValueError("built GLYPH manifest corpus SHA mismatch")
        for name, expected in manifest["files"].items():
            path = glyph_index / name
            if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
                raise ValueError("built GLYPH artifact SHA mismatch")
        status["indexes"] = {"GLYPH_manifest": spec(glyph_index / "manifest.json"),
                             "GLYPH_parts": {n: spec(glyph_index / n) for n in manifest["files"]},
                             "Move-r_index": spec(move_index)}
        save()
        request = "".join("746865\t" + str(k) + "\n" for k in (1, 10, 100)) + "QUIT\n"
        data = corpus.read_bytes()
        if hashlib.sha256(data).hexdigest() != calibration["corpus"]["sha256"]:
            raise ValueError("corpus SHA differs before byte checks")
        for engine, argv in [
            ("GLYPH", ["taskset", "-c", "0", str(glyph_reader), str(glyph_index / "bwt.rlbwt"),
                       str(glyph_index / "bwt.rlbwt.rank"), str(glyph_index / "locate_core_s128.bin")]),
            ("Move-r", ["taskset", "-c", "0", str(move_reader), "serve", str(move_index)]),
        ]:
            output = command(engine + "-preflight", argv, request)
            lines = output.splitlines()
            if engine == "Move-r":
                if json.loads(lines.pop(0)) != {"ready": True}:
                    raise ValueError("missing Move-r READY")
            if len(lines) != 3:
                raise ValueError("incomplete native responses")
            for limit, line in zip((1, 10, 100), lines):
                if engine == "GLYPH":
                    fields = line.split("\t")
                    if len(fields) != 9 or fields[0] != "OK":
                        raise ValueError("invalid GLYPH response")
                    count = int(fields[3])
                    raw_offsets = list(map(int, fields[8].split(","))) if fields[8] else []
                    if int(fields[4]) != len(raw_offsets) or int(fields[2]) - int(fields[1]) != count:
                        raise ValueError("GLYPH response dimensions differ")
                else:
                    parsed = json.loads(line)
                    if parsed["query_hex"] != "746865":
                        raise ValueError("Move-r query differs")
                    count, raw_offsets = parsed["count"], parsed["offsets"]
                    if raw_offsets != oracle["selected_sa_order_first100"][:limit]:
                        raise ValueError("Move-r raw selection is not the frozen SA prefix")
                offsets = sorted(raw_offsets)
                status["correctness_query_count"] += 1
                if count != oracle["count"] or offsets != oracle["offsets_by_limit"][str(limit)]:
                    raise ValueError(engine + " count or exact offset subset differs")
                if len(offsets) != min(limit, count) or len(set(offsets)) != len(offsets):
                    raise ValueError("incorrect bounded cardinality")
                for offset in offsets:
                    if not 0 <= offset <= len(data) - 3 or data[offset:offset + 3] != b"the":
                        raise ValueError("returned byte check failed")
                status["results"].append({"engine": engine, "limit": limit, "count": count,
                                          "offsets": offsets, "byte_check": True,
                                          "exact_oracle_list_match": True, "ok": True})
                save()
        if digest(corpus) != calibration["corpus"]["sha256"]:
            raise ValueError("corpus changed during preflight")
        status.update(phase="PASS", exit_code=0, corpus_unchanged=True)
        save()
        return 0
    except BaseException as error:
        status.update(failed_at_phase=status["phase"], phase="FAIL", exit_code=1, error=str(error))
        save()
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
