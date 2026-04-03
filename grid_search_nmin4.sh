#!/bin/bash

# nmin=4 grid search: 31 combos × 4 passes = 124 TIF
# GPU 0,1,2,3,4,5 × 2 concurrent per GPU = 12 parallel
#
# Usage:
#   ./grid_search_nmin4.sh
#   ./grid_search_nmin4.sh --gpus 0,1,2,3,4,5

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_nmin4_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$OUTPUT_DIR"

LOGFILE="${OUTPUT_DIR}/grid_search.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

GPUS="0,1,2,3,4,5"
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
CONCURRENT=$((NUM_GPUS * 2))

PYTHON="/data/kevin_workspace/envs/skyfall/bin/python"
INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"

T_STEPS=28
GAMMA=0.7
TILE_SIZE=1024
OVERLAP=128
MAX_PASS=4

echo ""
echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  nmin=4 Grid Search (31 combos × 4 passes = 124 TIF)${NC}"
echo -e "${CYAN}  GPU: ${GPUS} × 2 concurrent = ${CONCURRENT} parallel${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

COMBOS=(
    "4,6,5.0,1.0"
    "4,6,9.0,0.5"
    "4,8,3.0,1.5"
    "4,8,7.0,1.0"
    "4,10,3.0,0.5"
    "4,10,5.0,1.5"
    "4,10,9.0,1.0"
    "4,6,5.0,1.5"
    "4,6,9.0,1.0"
    "4,8,5.0,0.5"
    "4,8,7.0,1.5"
    "4,10,3.0,1.0"
    "4,10,7.0,0.5"
    "4,10,9.0,1.5"
    "4,6,3.0,1.0"
    "4,6,7.0,0.5"
    "4,6,9.0,1.5"
    "4,8,5.0,1.0"
    "4,8,9.0,0.5"
    "4,10,3.0,1.5"
    "4,10,7.0,1.0"
    "4,8,3.0,0.5"
    "4,8,5.0,1.5"
    "4,8,9.0,1.0"
    "4,10,5.0,0.5"
    "4,10,7.0,1.5"
    "4,8,3.0,1.0"
    "4,8,7.0,0.5"
    "4,8,9.0,1.5"
    "4,10,5.0,1.0"
    "4,10,9.0,0.5"
)

TOTAL_COMBOS=${#COMBOS[@]}
TOTAL_TIFS=$((TOTAL_COMBOS * MAX_PASS))
echo "Total combos: ${TOTAL_COMBOS} (× ${MAX_PASS} passes = ${TOTAL_TIFS} TIFs)"
echo ""

T_START=$(date +%s)

run_combo() {
    local gpu_id=$1
    local n_min=$2
    local n_max=$3
    local tg=$4
    local sg=$5
    local combo_num=$6

    local TAG_BASE=$(printf "nmin%02d_nmax%02d_tg%.1f_sg%.1f" "$n_min" "$n_max" "$tg" "$sg")
    local CURRENT_INPUT="$INPUT"

    for P in $(seq 1 "$MAX_PASS"); do
        if [ "$P" -eq 1 ]; then
            NO_STRETCH=""
        else
            NO_STRETCH="--no_stretch"
        fi

        local TAG="${TAG_BASE}_pass${P}"
        local OUTFILE="${OUTPUT_DIR}/${TAG}.tif"

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
            $NO_STRETCH \
            > "${OUTPUT_DIR}/${TAG}.stdout.log" 2>&1

        CURRENT_INPUT="$OUTFILE"
    done

    echo -e "${GREEN}[${combo_num}/${TOTAL_COMBOS}] GPU ${gpu_id}: ${TAG_BASE} pass1-${MAX_PASS} — DONE${NC}"
}

# Dispatch: 2 combos per GPU, round-robin
COMBO_IDX=0
while [ "$COMBO_IDX" -lt "$TOTAL_COMBOS" ]; do
    PIDS=()

    for ((slot=0; slot<CONCURRENT && COMBO_IDX<TOTAL_COMBOS; slot++)); do
        g=$((slot % NUM_GPUS))
        IFS=',' read -r NMIN NMAX TG SG <<< "${COMBOS[$COMBO_IDX]}"
        GPU_ID=${GPU_IDS[$g]}
        COMBO_IDX=$((COMBO_IDX + 1))

        TAG_BASE=$(printf "nmin%02d_nmax%02d_tg%.1f_sg%.1f" "$NMIN" "$NMAX" "$TG" "$SG")
        echo -e "${YELLOW}[${COMBO_IDX}/${TOTAL_COMBOS}] GPU ${GPU_ID}: ${TAG_BASE}${NC}"

        run_combo "$GPU_ID" "$NMIN" "$NMAX" "$TG" "$SG" "$COMBO_IDX" &
        PIDS+=($!)
    done

    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done

    ELAPSED=$(( $(date +%s) - T_START ))
    DONE=$COMBO_IDX
    if [ "$DONE" -gt 0 ]; then
        AVG=$(echo "scale=1; ${ELAPSED} / ${DONE}" | bc)
        REMAINING=$(echo "scale=0; (${TOTAL_COMBOS} - ${DONE}) * ${AVG} / ${CONCURRENT}" | bc)
        echo -e "${CYAN}--- Batch done: ${DONE}/${TOTAL_COMBOS} | Elapsed: ${ELAPSED}s ($(echo "scale=1; ${ELAPSED}/60" | bc)min) | Est. remaining: ${REMAINING}s ---${NC}"
    fi
    echo ""
done

T_END=$(date +%s)
T_ELAPSED=$(( T_END - T_START ))

echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  Grid Search Complete${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Total time: ${T_ELAPSED}s ($(echo "scale=1; ${T_ELAPSED}/60" | bc)min)"
echo "Combos: ${TOTAL_COMBOS}, TIFs: $(ls "${OUTPUT_DIR}"/*.tif 2>/dev/null | wc -l) / ${TOTAL_TIFS}"
echo ""
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
