#!/bin/bash

# Grid search: FLUX.2 SDEdit parameters
# Loads model once, iterates over (strength, guidance_scale, detail_sigma)
#
# Usage:
#   ./run_grid_flux2.sh

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_flux2_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$OUTPUT_DIR"

LOGFILE="${OUTPUT_DIR}/grid_search.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1

PYTHON="/data/kevin_workspace/envs/skyfall-kontext/bin/python"
INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"

echo ""
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
echo "Grid Search: FLUX.2 SDEdit"
echo "Output: ${OUTPUT_DIR}"
echo ""

CUDA_VISIBLE_DEVICES=4,5,6,7 $PYTHON "${SCRIPT_DIR}/grid_search_flux2.py" \
    --input "$INPUT" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --strengths "0.20,0.25,0.30,0.35,0.40" \
    --guidances "4.0,6.0" \
    --sigmas "4.0,6.0" \
    --alphas "0.0,0.3,0.5,0.7" \
    --prompt "High resolution clean satellite orthographic top-down image with sharp details, vivid colors, buildings, roads, and vegetation"

echo ""
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
echo "Output files:"
ls -lhS "${OUTPUT_DIR}"/*.tif 2>/dev/null | head -20
echo "Total: $(ls "${OUTPUT_DIR}"/*.tif 2>/dev/null | wc -l) files"
