# GLYPH OVH PRECHECK v1

Status: PRE-HTTP SAFETY CHECK

Required before HTTP/API:

RAM:
- free -h
- available should be >= 100G

SWAP:
- swapon --show
- production preference: swap disabled or explicitly accepted

DISK:
- df -h /
- df -h /tmp
- required free space:
  - >100G for 4GB rebuild
  - >300G for 16GB rebuild

PORT:
- ss -ltnp | grep 18080
- expected before start: empty
- expected after start: 127.0.0.1:18080 only

BIND:
- FastAPI must bind only to 127.0.0.1
- never bind 0.0.0.0 directly

RUNTIME SAFETY:
- one uvicorn worker initially
- lock around router stdin/stdout
- timeout required
- /health endpoint required
- graceful shutdown required
- logs required

SYSTEMD:
- MemoryMax=100G required for service mode
- Restart=on-failure

WARMUP:
- optional: vmtouch -t out_4gb/sa.bin
- do not assume vmtouch exists

DO NOT USE:
- raw http.server prototype
- public bind without nginx/auth
- multiple uvicorn workers sharing one stdin router
