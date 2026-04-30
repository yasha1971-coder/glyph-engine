#!/usr/bin/env bash
set -e

echo "[mini] building trivial corpus"

cp examples/mini/data.txt /tmp/mini_corpus.txt

echo "[mini] NOTE: this is conceptual pipeline"

echo "[mini] start server"
pkill -f glyph_http_server 2>/dev/null || true

python3 -m uvicorn glyph_http_server:app \
  --host 127.0.0.1 \
  --port 18081 \
  > /tmp/mini.log 2>&1 &

sleep 2

echo "[mini] test query"

./glyph_cli.py "error" || echo "[mini] expected: no index"

echo "[mini] done"
