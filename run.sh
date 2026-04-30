#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "[1/4] Starting HTTP server..."
echo "[note] using local prebuilt artifacts"
echo "[note] using local prebuilt artifacts"

# убить старые процессы
pkill -u $(whoami) -f "glyph_http_server" 2>/dev/null || true
pkill -u $(whoami) -f "glyph_segmented_live" 2>/dev/null || true

# запуск сервера в фоне
nohup python3 -m uvicorn glyph_http_server:app \
  --host 127.0.0.1 \
  --port 18080 \
  > run.log 2>&1 &

PID=$!
echo "Server PID: $PID"

echo "[2/4] Waiting for health..."

for i in {1..30}; do
  if curl -s http://127.0.0.1:18080/health | grep -q ok; then
    echo "Health OK"
    break
  fi
  sleep 1
done

echo "[3/4] Running test query..."

./glyph_cli.py --hex "$(xxd -p -c 999999 /tmp/query_41905.bin)"

echo "[4/4] Done"

echo "Logs:"
echo "  tail -f run.log"
