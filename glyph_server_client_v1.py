#!/usr/bin/env python3
import subprocess


class FMServer:
    def __init__(self, bin_path, fm, bwt):
        self.proc = subprocess.Popen(
            [bin_path, fm, bwt],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        ready = self.proc.stderr.readline().strip()
        if ready != "READY":
            raise RuntimeError(f"Server did not start, got: {ready}")

    def query(self, hex_pattern):
        self.proc.stdin.write(hex_pattern + "\n")
        self.proc.stdin.flush()

        line = self.proc.stdout.readline().strip()
        l, r, cnt = line.split()
        return int(l), int(r), int(cnt)

    def close(self):
        self.proc.stdin.write("__EXIT__\n")
        self.proc.stdin.flush()
        self.proc.wait()


if __name__ == "__main__":
    server = FMServer(
        "build/query_fm_server_v1",
        "out_1gb/fm_s0.bin",
        "out_1gb/bwt_s0.bin",
    )

    print(server.query("00"))
    print(server.query("0000"))

    server.close()