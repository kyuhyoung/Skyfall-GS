#!/bin/bash

# Grid search: n_min × n_max (4 GPU parallel)
#
# Phase 1: 45 combinations, tar_guidance=5.5, src_guidance=1.0 fixed
# Estimated: ~54min on 4 GPUs
#
# Usage:
#   ./grid_search_nmin_nmax.sh --gpus 4,5,6,7

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_nmin_nmax_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$OUTPUT_DIR"

LOGFILE="${OUTPUT_DIR}/grid_search.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse --gpus
GPUS="4,5,6,7"
for arg in "$@"; do
    if [ "$prev_arg" = "--gpus" ]; then
        GPUS="$arg"
        prev_arg=""
        continue
    fi
    if [ "$arg" = "--gpus" ]; then
        prev_arg="$arg"
        continue
    fi
done

IFS=',' read -ra GPU_IDS <<< "$GPUS"
NUM_GPUS=${#GPU_IDS[@]}

INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"

# Fixed parameters
TAR_GUIDANCE=5.5
SRC_GUIDANCE=1.0
T_STEPS=28
GAMMA=1.0
TILE_SIZE=1024
OVERLAP=128

echo ""
echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  Grid Search: n_min × n_max${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPUs: ${GPUS} (${NUM_GPUS} total)"
echo "Input: ${INPUT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Fixed: tg=${TAR_GUIDANCE}, sg=${SRC_GUIDANCE}, T=${T_STEPS}, gamma=${GAMMA}"
echo ""

# Build job list: (n_min, n_max) where n_min < n_max
JOBS=()
for N_MIN in 0 1 2 3 4 5; do
    for N_MAX in 4 5 6 7 8 9 10 12; do
        if [ "$N_MIN" -lt "$N_MAX" ]; then
            JOBS+=("${N_MIN},${N_MAX}")
        fi
    done
done

TOTAL=${#JOBS[@]}
echo "Total combinations: ${TOTAL}"
echo "Estimated time: ~$(echo "scale=0; (${TOTAL} + ${NUM_GPUS} - 1) / ${NUM_GPUS} * 5" | bc) min"
echo ""

# Run jobs in batches of NUM_GPUS
T_START=$(date +%s)
DONE=0

run_job() {
    local gpu_id=$1
    local n_min=$2
    local n_max=$3
    local job_num=$4

    local NMIN_PAD=$(printf "%02d" "$n_min")
    local NMAX_PAD=$(printf "%02d" "$n_max")
    local TAG="nmin${NMIN_PAD}_nmax${NMAX_PAD}_tg${TAR_GUIDANCE}_sg${SRC_GUIDANCE}"
    local OUTFILE="${OUTPUT_DIR}/${TAG}.tif"

    echo -e "${YELLOW}[${job_num}/${TOTAL}] GPU ${gpu_id}: n_min=${n_min}, n_max=${n_max}${NC}"

    python "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$INPUT" \
        --output "$OUTFILE" \
        --tile_size "$TILE_SIZE" \
        --overlap "$OVERLAP" \
        --n_min "$n_min" \
        --n_max "$n_max" \
        --n_avg 1 \
        --T_steps "$T_STEPS" \
        --gamma "$GAMMA" \
        --device "cuda:${gpu_id}" \
        --tar_guidance "$TAR_GUIDANCE" \
        --src_guidance "$SRC_GUIDANCE" \
        --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution" \
        --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors" \
        > "${OUTPUT_DIR}/${TAG}.stdout.log" 2>&1

    echo -e "${GREEN}[${job_num}/${TOTAL}] GPU ${gpu_id}: n_min=${n_min}, n_max=${n_max} — DONE${NC}"
}

JOB_IDX=0
while [ "$JOB_IDX" -lt "$TOTAL" ]; do
    PIDS=()

    # Launch up to NUM_GPUS jobs in parallel
    for ((g=0; g<NUM_GPUS && JOB_IDX<TOTAL; g++)); do
        IFS=',' read -r N_MIN N_MAX <<< "${JOBS[$JOB_IDX]}"
        GPU_ID=${GPU_IDS[$g]}
        JOB_IDX=$((JOB_IDX + 1))

        run_job "$GPU_ID" "$N_MIN" "$N_MAX" "$JOB_IDX" &
        PIDS+=($!)
    done

    # Wait for this batch to finish
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done

    DONE=$JOB_IDX
    ELAPSED=$(( $(date +%s) - T_START ))
    AVG=$(echo "scale=1; ${ELAPSED} / ${DONE}" | bc)
    REMAINING=$(echo "scale=0; (${TOTAL} - ${DONE}) * ${AVG} / ${NUM_GPUS}" | bc)
    echo -e "${CYAN}--- Batch done: ${DONE}/${TOTAL} | Elapsed: ${ELAPSED}s | Est. remaining: ${REMAINING}s ---${NC}"
    echo ""
done

T_END=$(date +%s)
T_ELAPSED=$(( T_END - T_START ))

echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  Grid Search Complete${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Total time: ${T_ELAPSED}s ($(echo "scale=1; ${T_ELAPSED}/60" | bc)min)"
echo "Combinations: ${TOTAL}"
echo ""
echo "Output files:"
ls -lhS "${OUTPUT_DIR}"/*.tif 2>/dev/null
echo ""
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
