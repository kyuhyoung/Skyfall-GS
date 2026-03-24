#!/bin/bash

# Refine GeoTIFF using FlowEdit + FLUX (multi-pass support)
#
# Usage:
#   ./run_refine.sh                    # default 1 pass
#   ./run_refine.sh --passes 3         # 3 passes
#   ./run_refine.sh --passes 2 --n_max 5

# Log file
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${SCRIPT_DIR}/output"
LOGFILE="${SCRIPT_DIR}/output/run_refine.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1
echo ""
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

OUTPUT_DIR="${SCRIPT_DIR}/output/flux_grid_search"
mkdir -p "$OUTPUT_DIR"

# Parse --passes and --start-pass from arguments
PASSES=2
START_PASS=1
INPUT_OVERRIDE=""
EXTRA_ARGS=()
for arg in "$@"; do
    if [ "$prev_arg" = "--passes" ]; then
        PASSES="$arg"
        prev_arg=""
        continue
    fi
    if [ "$prev_arg" = "--start-pass" ]; then
        START_PASS="$arg"
        prev_arg=""
        continue
    fi
    if [ "$prev_arg" = "--input" ]; then
        INPUT_OVERRIDE="$arg"
        prev_arg=""
        continue
    fi
    if [ "$arg" = "--passes" ] || [ "$arg" = "--start-pass" ] || [ "$arg" = "--input" ]; then
        prev_arg="$arg"
        continue
    fi
    EXTRA_ARGS+=("$arg")
done

INPUT="${INPUT_OVERRIDE:-/data/satellite/seoul/gangnam/samsung/gwarp_out_ps_ba/fused_top_naive.tif}"
# Always use original stem for output naming
INPUT_STEM="fused_top_naive"
END_PASS=$(( START_PASS + PASSES - 1 ))
echo "Passes: ${START_PASS} to ${END_PASS}"
echo "Input: ${INPUT}"
echo ""

CURRENT_INPUT="$INPUT"
for P in $(seq "$START_PASS" "$END_PASS"); do
    echo "===== Pass ${P}/${PASSES} ====="

    if [ "$END_PASS" -eq 1 ] && [ "$START_PASS" -eq 1 ]; then
        OUTPUT="${OUTPUT_DIR}/${INPUT_STEM}_flux_nmin00_nmax07_tg5.5_sg1.0.tif"
    else
        OUTPUT="${OUTPUT_DIR}/${INPUT_STEM}_flux_nmin00_nmax07_tg5.5_sg1.0_pass${P}.tif"
    fi

    python refine_geotiff.py \
        --input "$CURRENT_INPUT" \
        --output "$OUTPUT" \
        --tile_size 1024 \
        --overlap 128 \
        --n_min 0 \
        --n_max 7 \
        --n_avg 1 \
        --T_steps 28 \
        --gamma 1.0 \
        --save_preview \
        --device cuda:2 \
        --tar_guidance 5.5 \
        --src_guidance 1.0 \
        --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution" \
        --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors" \
        "${EXTRA_ARGS[@]}"

    CURRENT_INPUT="$OUTPUT"
    echo ""
done

echo "========== All ${PASSES} passes complete =========="
echo "$(date '+%Y-%m-%d %H:%M:%S')"
