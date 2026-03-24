#!/bin/bash

# Grid search for FlowEdit + FLUX refinement parameters
#
# Usage:
#   ./run_refine_grid.sh              # run full grid
#   ./run_refine_grid.sh --dry-run    # print commands without running

# Log file
SCRIPT_DIR_LOG="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${SCRIPT_DIR_LOG}/output"
LOGFILE="${SCRIPT_DIR_LOG}/output/run_refine_grid.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1
echo ""
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

DRY_RUN=false
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=true
    fi
done

INPUT="/data/satellite/seoul/gangnam/samsung/gwarp_out_ps_ba/fused_top_naive.tif"
INPUT_DIR="$(dirname "$INPUT")"
INPUT_STEM="$(basename "$INPUT" .tif)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output/flux_grid_search"
mkdir -p "$OUTPUT_DIR"

# Grid parameters
N_MIN_VALUES=(0)
N_MAX_VALUES=(5 7 9)
TAR_GUIDANCE=5.5
SRC_GUIDANCE=1.0
GAMMA=1.0

TOTAL=$(( ${#N_MIN_VALUES[@]} * ${#N_MAX_VALUES[@]} ))
COUNT=0

echo "Grid search: ${#N_MIN_VALUES[@]} x ${#N_MAX_VALUES[@]} = ${TOTAL} runs"
echo "  n_min:         ${N_MIN_VALUES[*]}"
echo "  n_max:         ${N_MAX_VALUES[*]}"
echo "  tar_guidance:  ${TAR_GUIDANCE} (fixed)"
echo "  src_guidance:  ${SRC_GUIDANCE} (fixed)"
echo "  gamma:         ${GAMMA} (fixed)"
echo ""

for NMIN in "${N_MIN_VALUES[@]}"; do
    for NMAX in "${N_MAX_VALUES[@]}"; do
        COUNT=$((COUNT + 1))
        NMIN_PAD=$(printf "%02d" "$NMIN")
        NMAX_PAD=$(printf "%02d" "$NMAX")
        OUTPUT="${OUTPUT_DIR}/${INPUT_STEM}_flux_nmin${NMIN_PAD}_nmax${NMAX_PAD}_tg${TAR_GUIDANCE}_sg${SRC_GUIDANCE}.tif"

        echo "---------- [${COUNT}/${TOTAL}] n_min=${NMIN} n_max=${NMAX} ----------"

        if [ -f "$OUTPUT" ]; then
            echo "SKIP: $OUTPUT already exists"
            echo ""
            continue
        fi

        if [ "$DRY_RUN" = true ]; then
            echo "DRY-RUN: would produce $(basename "$OUTPUT")"
            echo ""
            continue
        fi

        python refine_geotiff.py \
            --input "$INPUT" \
            --output "$OUTPUT" \
            --tile_size 1024 \
            --overlap 128 \
            --n_min "$NMIN" \
            --n_max "$NMAX" \
            --n_avg 1 \
            --T_steps 28 \
            --gamma "$GAMMA" \
            --save_preview \
            --device cuda:2 \
            --tar_guidance "$TAR_GUIDANCE" \
            --src_guidance "$SRC_GUIDANCE" \
            --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution" \
            --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"

        echo ""
    done
done

echo "========== Grid search complete: ${COUNT}/${TOTAL} =========="
echo "$(date '+%Y-%m-%d %H:%M:%S')"
