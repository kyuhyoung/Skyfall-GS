#!/bin/bash

# nmin=2 grid search Part 1: combo 1-12 on GPU 5
# nmax=5,6,7 × tg=4.5,5.0,5.5,6.0 × pass 1-4
#
# Usage:
#   ./grid_nmin2_part1.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_nmin2_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$OUTPUT_DIR"

LOGFILE="${OUTPUT_DIR}/grid_search.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

GPU_ID=5
PYTHON="/data/kevin_workspace/envs/skyfall/bin/python"
INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"

N_MIN=2
SG=1.0
T_STEPS=28
GAMMA=0.7
TILE_SIZE=1024
OVERLAP=128
MAX_PASS=4

echo ""
echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  nmin=2 Grid Search Part 1 (GPU ${GPU_ID})${NC}"
echo -e "${CYAN}  nmax=5,6,7 × tg=4.5,5.0,5.5,6.0 × pass 1-4${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# 12 combos: nmax=5,6,7 × tg=4.5,5.0,5.5,6.0
COMBOS=(
    "5,4.5"
    "5,5.0"
    "5,5.5"
    "5,6.0"
    "6,4.5"
    "6,5.0"
    "6,5.5"
    "6,6.0"
    "7,4.5"
    "7,5.0"
    "7,5.5"
    "7,6.0"
)

TOTAL_COMBOS=${#COMBOS[@]}
TOTAL_TIFS=$((TOTAL_COMBOS * MAX_PASS))

T_START=$(date +%s)
COMBO_DONE=0

for combo in "${COMBOS[@]}"; do
    IFS=',' read -r NMAX TG <<< "$combo"
    TAG_BASE=$(printf "nmin%02d_nmax%02d_tg%.1f_sg%.1f" "$N_MIN" "$NMAX" "$TG" "$SG")
    COMBO_DONE=$((COMBO_DONE + 1))

    echo -e "${YELLOW}[${COMBO_DONE}/${TOTAL_COMBOS}] ${TAG_BASE} pass 1-${MAX_PASS}${NC}"

    CURRENT_INPUT="$INPUT"
    for P in $(seq 1 "$MAX_PASS"); do
        if [ "$P" -eq 1 ]; then
            NO_STRETCH=""
        else
            NO_STRETCH="--no_stretch"
        fi

        TAG="${TAG_BASE}_pass${P}"
        OUTFILE="${OUTPUT_DIR}/${TAG}.tif"

        $PYTHON "${SCRIPT_DIR}/refine_geotiff.py" \
            --input "$CURRENT_INPUT" \
            --output "$OUTFILE" \
            --tile_size "$TILE_SIZE" \
            --overlap "$OVERLAP" \
            --n_min "$N_MIN" \
            --n_max "$NMAX" \
            --n_avg 1 \
            --T_steps "$T_STEPS" \
            --gamma "$GAMMA" \
            --device "cuda:${GPU_ID}" \
            --tar_guidance "$TG" \
            --src_guidance "$SG" \
            --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution" \
            --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors" \
            $NO_STRETCH \
            > "${OUTPUT_DIR}/${TAG}.stdout.log" 2>&1

        CURRENT_INPUT="$OUTFILE"
    done

    ELAPSED=$(( $(date +%s) - T_START ))
    AVG=$(echo "scale=1; ${ELAPSED} / ${COMBO_DONE}" | bc)
    REMAINING=$(echo "scale=0; (${TOTAL_COMBOS} - ${COMBO_DONE}) * ${AVG}" | bc)
    echo -e "${GREEN}[${COMBO_DONE}/${TOTAL_COMBOS}] ${TAG_BASE} — DONE | Elapsed: ${ELAPSED}s | Est. remaining: ${REMAINING}s${NC}"
    echo ""
done

T_END=$(date +%s)
T_ELAPSED=$(( T_END - T_START ))

echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  Part 1 Complete${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Total time: ${T_ELAPSED}s ($(echo "scale=1; ${T_ELAPSED}/60" | bc)min)"
echo "Combos: ${TOTAL_COMBOS}, TIFs: $(ls "${OUTPUT_DIR}"/*.tif 2>/dev/null | wc -l)"
echo ""
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
