#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output/test_inverse"
mkdir -p "$OUTPUT_DIR"

LOGFILE="${OUTPUT_DIR}/run.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1

echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

python "${SCRIPT_DIR}/refine_geotiff.py" \
    --input /data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif \
    --output "${OUTPUT_DIR}/nmin00_nmax07_tg5.5_sg1.0.tif" \
    --n_min 0 --n_max 7 --T_steps 28 --gamma 1.0 \
    --tar_guidance 5.5 --src_guidance 1.0 \
    --inverse_stretch --seed 42 \
    --device cuda:7

echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
