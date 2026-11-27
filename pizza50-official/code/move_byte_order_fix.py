#!/usr/bin/env python3
"""Expose an existing Move-r build parameter; retain the stock reader and old FAIL."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    w = Path(sys.argv[1]).resolve()
    status_file = w / "control/move-byteorder-fix.json"
    status = {"phase": "STARTED", "controller_pid": os.getpid(), "stages": [],
              "results": [], "performance_samples": 0, "prior_attempt": "FAIL retained",
              "changed_existing_library_parameter": {"reorder_alphabet": False}}
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               LC_ALL="C", PYTHONHASHSEED="0")

    def save():
        temp = w / "control/move-byteorder-fix.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(status_file)

    def run(name, command, cwd=None, stdin=None):
        status["phase"] = name
        rec = {"stage": name, "argv": command, "cwd": str(cwd or w)}
        out, err = w / ("control/" + name + ".stdout"), w / ("control/" + name + ".stderr")
        rec.update(stdout=str(out), stderr=str(err))
        if stdin is not None:
            path = w / ("control/" + name + ".stdin")
            path.write_text(stdin, encoding="ascii")
            rec["stdin"] = str(path)
        with out.open("xb") as stdout, err.open("xb") as stderr:
            child = subprocess.Popen(command, cwd=cwd or w, env=env,
                                     stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, text=True)
            rec["pid"] = child.pid
            status["active_command"] = rec
            save()
            child.communicate(stdin)
        rec["exit_code"] = child.returncode
        status["stages"].append(rec)
        status.pop("active_command", None)
        save()
        if child.returncode:
            raise RuntimeError(name + " failed, exit " + str(child.returncode))
        return out.read_text()

    save()
    try:
        locked = json.loads((w / "BUILD_FIX_INPUT_LOCK.json").read_text())
        for item in locked["files"]:
            if digest(item["path"]) != item["sha256"]:
                raise ValueError("changed build input: " + item["path"])
        oracle_file = w / "oracle/oracle.json"
        inputs = json.loads((w / "INPUT_LOCK.json").read_text())
        if digest(oracle_file) != inputs["oracle"]["sha256"]:
            raise ValueError("oracle SHA changed")
        oracle = json.loads(oracle_file.read_text())
        corpus = Path(inputs["corpus"]["path"])
        if digest(corpus) != inputs["corpus"]["sha256"]:
            raise ValueError("corpus SHA changed")
        flags = json.loads((w / "ACCEPTED_BUILD_FLAGS.json").read_text())
        compile_rec = flags["mover-compile.json"]
        compile_cmd = list(compile_rec["command"])
        compile_cmd[compile_cmd.index("-c") + 1] = str(w / "code/mover_build_byte_order.cpp")
        new_object = str(w / "control/mover-byteorder.o")
        compile_cmd[compile_cmd.index("-o") + 1] = new_object
        run("move-byteorder-compile", compile_cmd, compile_rec["cwd"])
        link_cmd = list(flags["mover-link.json"]["command"])
        link_cmd = [new_object if x.endswith("/mover_reader.o") else x for x in link_cmd]
        builder = w / "tools/move-native/build-byteorder.native"
        link_cmd[link_cmd.index("-o") + 1] = str(builder)
        run("move-byteorder-link", link_cmd)
        index = w / "move-byteorder.index"
        if index.exists():
            raise FileExistsError("corrected index already exists")
        run("move-byteorder-build", ["taskset", "-c", "0", str(builder), str(corpus), str(index)])
        reader = w / "tools/move-native/engine.native"
        if digest(reader) != "1828f0101c0da5c9cb615259618983d384527fb5e135f37f332ed3d7085ab1ad":
            raise ValueError("stock Move-r reader changed")
        request = "".join("746865\t" + str(k) + "\n" for k in (1, 10, 100)) + "QUIT\n"
        output = run("move-byteorder-query", ["taskset", "-c", "0", str(reader), "serve", str(index)], stdin=request)
        lines = output.splitlines()
        if len(lines) != 4 or json.loads(lines[0]) != {"ready": True}:
            raise ValueError("missing Move-r response")
        data = corpus.read_bytes()
        if hashlib.sha256(data).hexdigest() != inputs["corpus"]["sha256"]:
            raise ValueError("source bytes changed after build")
        for limit, line in zip((1, 10, 100), lines[1:]):
            reply = json.loads(line)
            raw = reply["offsets"]
            offsets = sorted(raw)
            if reply["query_hex"] != "746865" or reply["count"] != oracle["count"]:
                raise ValueError("query or full count mismatch")
            if raw != oracle["selected_sa_order_first100"][:limit]:
                raise ValueError("corrected Move-r SA-prefix selection mismatch")
            if offsets != oracle["offsets_by_limit"][str(limit)]:
                raise ValueError("corrected normalized coordinates mismatch")
            if len(offsets) != min(limit, reply["count"]) or len(set(offsets)) != len(offsets):
                raise ValueError("bounded cardinality mismatch")
            if not all(0 <= pos <= len(data) - 3 and data[pos:pos + 3] == b"the" for pos in offsets):
                raise ValueError("byte check failed")
            status["results"].append({"engine": "Move-r", "reorder_alphabet": False, "limit": limit,
                                      "count": reply["count"], "offsets": offsets, "byte_check": True,
                                      "exact_oracle_list_match": True, "ok": True})
        status.update(phase="PASS", exit_code=0, reader_unchanged=True,
                      builder={"path": str(builder), "sha256": digest(builder), "bytes": builder.stat().st_size},
                      index={"path": str(index), "sha256": digest(index), "bytes": index.stat().st_size})
        save()
        return 0
    except BaseException as error:
        status.update(failed_at_phase=status["phase"], phase="FAIL", exit_code=1, error=str(error))
        save()
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
