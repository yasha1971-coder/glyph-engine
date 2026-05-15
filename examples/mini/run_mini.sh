#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

exec "$ROOT/examples/mini/build_mini.sh"
