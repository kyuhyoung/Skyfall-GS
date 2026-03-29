#!/bin/bash

# Coarse grid + Pass sweep + Extra fill for FlowEdit score map
# GPU 4,5,6,7 parallel
#
# Usage:
#   ./grid_search_coarse_pass.sh
#   ./grid_search_coarse_pass.sh --gpus 4,5,6,7

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_coarse_pass_$(date '+%Y%m%d_%H%M%S')"
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

PYTHON="/data/kevin_workspace/envs/skyfall/bin/python"
INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"

# Fixed params
T_STEPS=28
GAMMA=1.0
TILE_SIZE=1024
OVERLAP=128

echo ""
echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  Coarse Grid + Pass Sweep + Extra Fill${NC}"
echo -e "${CYAN}  FlowEdit + FLUX.1-dev Score Map${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPUs: ${GPUS} (${NUM_GPUS} total)"
echo "Input: ${INPUT}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

T_START=$(date +%s)

run_single() {
    local gpu_id=$1
    local n_min=$2
    local n_max=$3
    local tg=$4
    local sg=$5
    local job_num=$6
    local total=$7

    local TAG=$(printf "nmin%02d_nmax%02d_tg%.1f_sg%.1f" "$n_min" "$n_max" "$tg" "$sg")
    local OUTFILE="${OUTPUT_DIR}/${TAG}.tif"

    echo -e "${YELLOW}[${job_num}/${total}] GPU ${gpu_id}: nmin=${n_min}, nmax=${n_max}, tg=${tg}, sg=${sg}${NC}"

    $PYTHON "${SCRIPT_DIR}/refine_geotiff.py" \
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
        --tar_guidance "$tg" \
        --src_guidance "$sg" \
        --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution" \
        --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors" \
        > "${OUTPUT_DIR}/${TAG}.stdout.log" 2>&1

    echo -e "${GREEN}[${job_num}/${total}] GPU ${gpu_id}: ${TAG} — DONE${NC}"
}

run_multipass() {
    local gpu_id=$1
    local n_min=$2
    local n_max=$3
    local tg=$4
    local sg=$5
    local max_pass=$6
    local job_num=$7
    local total=$8

    local TAG_BASE=$(printf "nmin%02d_nmax%02d_tg%.1f_sg%.1f" "$n_min" "$n_max" "$tg" "$sg")
    local CURRENT_INPUT="$INPUT"

    for P in $(seq 1 "$max_pass"); do
        local TAG="${TAG_BASE}_pass${P}"
        local OUTFILE="${OUTPUT_DIR}/${TAG}.tif"

        echo -e "${YELLOW}[${job_num}/${total}] GPU ${gpu_id}: ${TAG}${NC}"

        $PYTHON "${SCRIPT_DIR}/refine_geotiff.py" \
            --input "$CURRENT_INPUT" \
            --output "$OUTFILE" \
            --tile_size "$TILE_SIZE" \
            --overlap "$OVERLAP" \
            --n_min "$n_min" \
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

        CURRENT_INPUT="$OUTFILE"
    done

    echo -e "${GREEN}[${job_num}/${total}] GPU ${gpu_id}: ${TAG_BASE} pass1-${max_pass} — DONE${NC}"
}

##############################################################################
# Phase 1: Coarse grid (25 single-pass) + Extra fill (8 single-pass) = 33
##############################################################################
echo -e "${CYAN}=== Phase 1: Coarse grid + Extra fill (33 single-pass) ===${NC}"

SINGLE_JOBS=(
    # Coarse grid (25)
    "0,3,3.0,1.0"
    "0,3,5.5,1.0"
    "0,3,8.0,1.0"
    "0,5,3.0,1.0"
    "0,5,8.0,1.0"
    "0,5,10.0,1.0"
    "0,7,3.0,1.0"
    "0,7,8.0,1.0"
    "0,7,10.0,1.0"
    "0,10,3.0,1.0"
    "0,10,5.5,1.0"
    "0,10,8.0,1.0"
    "0,10,10.0,1.0"
    "0,12,3.0,1.0"
    "0,12,5.5,1.0"
    "0,12,8.0,1.0"
    "0,15,3.0,1.0"
    "0,15,5.5,1.0"
    "0,15,8.0,1.0"
    "0,20,5.5,1.0"
    "2,7,5.5,1.0"
    "2,9,5.5,1.0"
    "2,12,5.5,1.0"
    "4,9,5.5,1.0"
    "4,12,5.5,1.0"
    # Extra fill (8)
    "0,20,3.0,1.0"
    "0,20,8.0,1.0"
    "0,15,10.0,1.0"
    "0,12,10.0,1.0"
    "0,5,5.5,1.0"
    "0,10,6.0,1.0"
    "0,7,5.5,0.5"
    "0,7,5.5,1.5"
)

TOTAL_SINGLE=${#SINGLE_JOBS[@]}
TOTAL_ALL=$((TOTAL_SINGLE + 5))  # 5 multi-pass combos

JOB_IDX=0
while [ "$JOB_IDX" -lt "$TOTAL_SINGLE" ]; do
    PIDS=()

    for ((g=0; g<NUM_GPUS && JOB_IDX<TOTAL_SINGLE; g++)); do
        IFS=',' read -r NMIN NMAX TG SG <<< "${SINGLE_JOBS[$JOB_IDX]}"
        GPU_ID=${GPU_IDS[$g]}
        JOB_IDX=$((JOB_IDX + 1))

        run_single "$GPU_ID" "$NMIN" "$NMAX" "$TG" "$SG" "$JOB_IDX" "$TOTAL_ALL" &
        PIDS+=($!)
    done

    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done

    ELAPSED=$(( $(date +%s) - T_START ))
    echo -e "${CYAN}--- Phase 1: ${JOB_IDX}/${TOTAL_SINGLE} done | Elapsed: ${ELAPSED}s ---${NC}"
    echo ""
done

##############################################################################
# Phase 2: Pass sweep (5 combos × 4 passes each)
##############################################################################
echo -e "${CYAN}=== Phase 2: Pass sweep (5 combos × pass 1-4) ===${NC}"

# nmin,nmax,tg,sg,max_pass
MULTI_JOBS=(
    "0,9,5.0,1.0,4"
    "0,9,6.0,1.0,4"
    "0,8,6.5,1.0,4"
    "0,8,6.0,1.0,4"
    "0,7,5.5,1.0,4"
)

MULTI_TOTAL=${#MULTI_JOBS[@]}
MULTI_IDX=0
while [ "$MULTI_IDX" -lt "$MULTI_TOTAL" ]; do
    PIDS=()

    for ((g=0; g<NUM_GPUS && MULTI_IDX<MULTI_TOTAL; g++)); do
        IFS=',' read -r NMIN NMAX TG SG MAXP <<< "${MULTI_JOBS[$MULTI_IDX]}"
        GPU_ID=${GPU_IDS[$g]}
        MULTI_IDX=$((MULTI_IDX + 1))
        JOB_NUM=$((TOTAL_SINGLE + MULTI_IDX))

        run_multipass "$GPU_ID" "$NMIN" "$NMAX" "$TG" "$SG" "$MAXP" "$JOB_NUM" "$TOTAL_ALL" &
        PIDS+=($!)
    done

    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done

    ELAPSED=$(( $(date +%s) - T_START ))
    echo -e "${CYAN}--- Phase 2: ${MULTI_IDX}/${MULTI_TOTAL} combos done | Elapsed: ${ELAPSED}s ---${NC}"
    echo ""
done

##############################################################################
# Summary
##############################################################################
T_END=$(date +%s)
T_ELAPSED=$(( T_END - T_START ))

echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  Grid Search Complete${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Total time: ${T_ELAPSED}s ($(echo "scale=1; ${T_ELAPSED}/60" | bc)min)"
echo ""
echo "Output files:"
ls -lhS "${OUTPUT_DIR}"/*.tif 2>/dev/null | head -20
echo "..."
echo "Total: $(ls "${OUTPUT_DIR}"/*.tif 2>/dev/null | wc -l) files"
echo ""
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
