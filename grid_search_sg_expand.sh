#!/bin/bash

# Grid search: src_guidance expansion (sg=0.25, 0.75, 1.25, 2.0)
# nmax=7~9, tg=5.0~6.0, pass=1
# GPU 5,6,7 parallel
#
# Usage:
#   ./grid_search_sg_expand.sh
#   ./grid_search_sg_expand.sh --gpus 5,6,7

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_sg_expand_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$OUTPUT_DIR"

LOGFILE="${OUTPUT_DIR}/grid_search.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse --gpus
GPUS="5,6,7"
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

PYTHON="/data/kevin_workspace/envs/skyfall/bin/python"
INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"

# Fixed params
N_MIN=0
T_STEPS=28
GAMMA=0.7
TILE_SIZE=1024
OVERLAP=128

echo ""
echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  Grid Search: src_guidance expansion${NC}"
echo -e "${CYAN}  sg=0.25, 0.75, 1.25, 2.0 × nmax=7,8,9 × tg=5.0,5.5,6.0${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPUs: ${GPUS} (${NUM_GPUS} total)"
echo "Input: ${INPUT}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

JOBS=(
    "7,5.0,0.25"
    "7,5.0,0.75"
    "7,5.0,1.25"
    "7,5.0,2.0"
    "7,5.5,0.25"
    "7,5.5,0.75"
    "7,5.5,1.25"
    "7,5.5,2.0"
    "7,6.0,0.25"
    "7,6.0,0.75"
    "7,6.0,1.25"
    "7,6.0,2.0"
    "8,5.5,0.25"
    "8,5.5,0.75"
    "8,5.5,1.25"
    "8,5.5,2.0"
    "8,6.0,0.25"
    "8,6.0,0.75"
    "8,6.0,1.25"
    "8,6.0,2.0"
    "9,5.0,0.25"
    "9,5.0,0.75"
    "9,5.0,1.25"
    "9,5.0,2.0"
)

TOTAL=${#JOBS[@]}
echo "Total combinations: ${TOTAL}"
echo ""

T_START=$(date +%s)
DONE=0

run_job() {
    local gpu_id=$1
    local n_max=$2
    local tg=$3
    local sg=$4
    local job_num=$5

    local TAG=$(printf "nmin%02d_nmax%02d_tg%.1f_sg%.2f" "$N_MIN" "$n_max" "$tg" "$sg")
    local OUTFILE="${OUTPUT_DIR}/${TAG}.tif"

    echo -e "${YELLOW}[${job_num}/${TOTAL}] GPU ${gpu_id}: nmax=${n_max}, tg=${tg}, sg=${sg}${NC}"

    $PYTHON "${SCRIPT_DIR}/refine_geotiff.py" \
        --input "$INPUT" \
        --output "$OUTFILE" \
        --tile_size "$TILE_SIZE" \
        --overlap "$OVERLAP" \
        --n_min "$N_MIN" \
        --n_max "$n_max" \
        --n_avg 1 \
        --T_steps "$T_STEPS" \
        --gamma "$GAMMA" \
        --device "cuda:${gpu_id}" \
        --tar_guidance "$tg" \
        --src_guidance "$sg" \
        --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution" \
        --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors" \
        > "${OUTPUT_DIR}/${TAG}.stdout.log" 2>&1

    echo -e "${GREEN}[${job_num}/${TOTAL}] GPU ${gpu_id}: nmax=${n_max}, tg=${tg}, sg=${sg} — DONE${NC}"
}

JOB_IDX=0
while [ "$JOB_IDX" -lt "$TOTAL" ]; do
    PIDS=()

    for ((g=0; g<NUM_GPUS && JOB_IDX<TOTAL; g++)); do
        IFS=',' read -r NMAX TG SG <<< "${JOBS[$JOB_IDX]}"
        GPU_ID=${GPU_IDS[$g]}
        JOB_IDX=$((JOB_IDX + 1))

        run_job "$GPU_ID" "$NMAX" "$TG" "$SG" "$JOB_IDX" &
        PIDS+=($!)
    done

    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done

    DONE=$JOB_IDX
    ELAPSED=$(( $(date +%s) - T_START ))
    if [ "$DONE" -gt 0 ]; then
        AVG=$(echo "scale=1; ${ELAPSED} / ${DONE}" | bc)
        REMAINING=$(echo "scale=0; (${TOTAL} - ${DONE}) * ${AVG} / ${NUM_GPUS}" | bc)
    else
        REMAINING="?"
    fi
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
ls -lhS "${OUTPUT_DIR}"/*.tif 2>/dev/null | head -10
echo "..."
echo "Total: $(ls "${OUTPUT_DIR}"/*.tif 2>/dev/null | wc -l) files"
echo ""
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
