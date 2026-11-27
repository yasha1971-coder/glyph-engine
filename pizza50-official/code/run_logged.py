#!/usr/bin/env python3
"""Retain a command's output and completion independently of the Codex call."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=Path, required=True)
    ap.add_argument("--cwd", type=Path, required=True)
    ap.add_argument("--stdin-file", type=Path)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    command = a.command[1:] if a.command and a.command[0] == "--" else a.command
    if not command:
        ap.error("missing command")
    a.prefix.parent.mkdir(parents=True, exist_ok=True)
    status_file = Path(str(a.prefix) + ".status.json")
    if status_file.exists():
        raise FileExistsError("attempt already recorded")
    status = {"argv": command, "cwd": str(a.cwd), "wrapper_pid": os.getpid(),
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "phase": "STARTING", "performance_benchmark": False}

    def save():
        temp = Path(str(a.prefix) + ".status.tmp")
        temp.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temp.replace(status_file)

    save()
    started = time.monotonic()
    try:
        with Path(str(a.prefix) + ".stdout").open("xb") as out, \
                Path(str(a.prefix) + ".stderr").open("xb") as err:
            stdin = a.stdin_file.open("rb") if a.stdin_file else subprocess.DEVNULL
            try:
                process = subprocess.Popen(command, cwd=a.cwd, stdin=stdin, stdout=out, stderr=err)
                status.update(phase="RUNNING", child_pid=process.pid)
                save()
                code = process.wait()
            finally:
                if a.stdin_file:
                    stdin.close()
        status.update(phase="COMPLETE", exit_code=code,
                      preparation_or_control_seconds=time.monotonic() - started,
                      finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        return 0 if code == 0 else 1
    except BaseException as error:
        status.update(phase="WRAPPER_FAILURE", error=str(error),
                      preparation_or_control_seconds=time.monotonic() - started)
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
