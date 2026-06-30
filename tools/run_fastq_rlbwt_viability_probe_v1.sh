#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SAMPLES="${GLYPH_FASTQ_SAMPLE_MIB:-256}"

if [ "$#" -gt 0 ]; then
  PATHS=("$@")
else
  echo "[search] looking for FASTQ-like corpora..."
  mapfile -t PATHS < <(
    find /home/glyph /mnt /srv \
      -maxdepth 8 \
      -type f \
      \( \
        -iname "*NA12878*" -o \
        -iname "*ERR194147*" -o \
        -iname "*.fastq" -o \
        -iname "*.fq" -o \
        -iname "*.fastq.gz" -o \
        -iname "*.fq.gz" \
      \) \
      -not -path "*/GLYPH_CPP_BACKEND/benchmarks/work/*" \
      -not -path "*/GLYPH_CPP_BACKEND/.git/*" \
      -size +100M \
      2>/dev/null | sort -u
  )
fi

if [ "${#PATHS[@]}" -eq 0 ]; then
  mkdir -p benchmarks/results
  cat > benchmarks/results/FASTQ_RLBWT_VIABILITY_PROBE_V1.md <<'EOF'
# FASTQ_RLBWT_VIABILITY_PROBE_V1

Status: no FASTQ corpus found

No FASTQ-like corpus was found automatically.

Run manually:

    GLYPH_FASTQ_SAMPLE_MIB=256 ./tools/run_fastq_rlbwt_viability_probe_v1.sh /path/to/NA12878.fastq /path/to/ERR194147.fastq

EOF
  echo "no FASTQ files found"
  exit 0
fi

echo "[paths]"
printf '%s\n' "${PATHS[@]}"

echo "[samples MiB] $SAMPLES"

python3 tools/fastq_rlbwt_viability_probe_v1.py \
  --sample-mib $SAMPLES \
  "${PATHS[@]}"

echo
echo "=== report ==="
sed -n '1,220p' benchmarks/results/FASTQ_RLBWT_VIABILITY_PROBE_V1.md
