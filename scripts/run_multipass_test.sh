#!/bin/bash

# Multi-pass test: nmin=0, nmax=7, tg=5.5, sg=1.0, pass 1-6
# pass 1-4 results should already exist in grid_coarse_pass output
# This runs pass 5 and 6 on top of pass 4

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Find existing pass4 result
GRID_DIR=$(ls -dt "${SCRIPT_DIR}/output/grid_coarse_pass_"*/ 2>/dev/null | head -1)
PASS4="${GRID_DIR}nmin00_nmax07_tg5.5_sg1.0_pass4.tif"

if [ ! -f "$PASS4" ]; then
    echo "ERROR: pass4 not found at ${PASS4}"
    echo "Running pass 1-6 from scratch..."
    START_PASS=1
    CURRENT_INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"
else
    echo "Found pass4: ${PASS4}"
    START_PASS=5
    CURRENT_INPUT="$PASS4"
fi

OUTPUT_DIR="${GRID_DIR}"
LOGFILE="${OUTPUT_DIR}/multipass_test.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1

PYTHON="/data/kevin_workspace/envs/skyfall/bin/python"

echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
echo "Multi-pass test: nmin=0, nmax=7, tg=5.5, sg=1.0"
echo "Starting from pass ${START_PASS}"
echo ""

for P in $(seq "$START_PASS" 6); do
    if [ "$P" -eq 1 ]; then
        NO_STRETCH=""
    else
        NO_STRETCH="--no_stretch"
    fi

    OUTPUT="${OUTPUT_DIR}/nmin00_nmax07_tg5.5_sg1.0_pass${P}.tif"
    echo "===== Pass ${P}/6 ====="
    echo "  Input: ${CURRENT_INPUT}"
    echo "  Output: ${OUTPUT}"

    $PYTHON "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$CURRENT_INPUT" \
        --output "$OUTPUT" \
        --tile_size 1024 \
        --overlap 128 \
        --n_min 0 \
        --n_max 7 \
        --n_avg 1 \
        --T_steps 28 \
        --gamma 0.7 \
        --device cuda:7 \
        --tar_guidance 5.5 \
        --src_guidance 1.0 \
        --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution" \
        --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors" \
        $NO_STRETCH

    CURRENT_INPUT="$OUTPUT"
    echo ""
done

echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
