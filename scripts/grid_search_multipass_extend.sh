#!/bin/bash

# Multi-pass extension: extend existing results to higher passes
# GPU 0,6,7 parallel (3 GPUs, each combo runs sequentially on one GPU)
#
# Usage:
#   ./grid_search_multipass_extend.sh

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_multipass_extend_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$OUTPUT_DIR"

LOGFILE="${OUTPUT_DIR}/grid_search.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

GPUS="5"
IFS=',' read -ra GPU_IDS <<< "$GPUS"
NUM_GPUS=${#GPU_IDS[@]}

PYTHON="/data/kevin_workspace/envs/skyfall/bin/python"
INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"
COARSE_DIR="${SCRIPT_DIR}/output/grid_coarse_pass_20260327_183751"
FINE_DIR="${SCRIPT_DIR}/output/grid_flowedit_fine_20260327_114126"

T_STEPS=28
GAMMA=0.7
TILE_SIZE=1024
OVERLAP=128

echo ""
echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  Multi-pass Extension${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPUs: ${GPUS} (${NUM_GPUS} total)"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

T_START=$(date +%s)

run_multipass() {
    local gpu_id=$1
    local n_min=$2
    local n_max=$3
    local tg=$4
    local sg=$5
    local start_pass=$6
    local end_pass=$7
    local start_input=$8
    local job_label=$9

    local TAG_BASE=$(printf "nmin%02d_nmax%02d_tg%.1f_sg%.1f" "$n_min" "$n_max" "$tg" "$sg")
    local CURRENT_INPUT="$start_input"

    for P in $(seq "$start_pass" "$end_pass"); do
        if [ "$P" -eq 1 ]; then
            NO_STRETCH=""
        else
            NO_STRETCH="--no_stretch"
        fi

        local TAG="${TAG_BASE}_pass${P}"
        local OUTFILE="${OUTPUT_DIR}/${TAG}.tif"

        echo -e "${YELLOW}[${job_label}] GPU ${gpu_id}: ${TAG}${NC}"

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

    echo -e "${GREEN}[${job_label}] GPU ${gpu_id}: ${TAG_BASE} pass${start_pass}-${end_pass} — DONE${NC}"
}

# ── Group A: start from pass 1 (need 5 passes each) ──
# nmax=5/tg=5.5: pass 1→6 (start from original)
# nmax=6/tg=6.0: pass 1→6 (start from original)
# nmax=10/tg=6.0: pass 1→6 (start from original)

echo -e "${CYAN}=== Group A: Full pass 1-6 (3 combos, sequential on GPU ${GPU_IDS[0]}) ===${NC}"
run_multipass "${GPU_IDS[0]}" 0 5 5.5 1.0 1 6 "$INPUT" "A1"
run_multipass "${GPU_IDS[0]}" 0 6 6.0 1.0 1 6 "$INPUT" "A2"
run_multipass "${GPU_IDS[0]}" 0 10 6.0 1.0 1 6 "$INPUT" "A3"

ELAPSED=$(( $(date +%s) - T_START ))
echo -e "${CYAN}--- Group A done | Elapsed: ${ELAPSED}s ---${NC}"
echo ""

# ── Group B: start from pass 4 (need 2 passes each) ──
echo -e "${CYAN}=== Group B: Extend pass 5-6 (3 combos, sequential on GPU ${GPU_IDS[0]}) ===${NC}"
run_multipass "${GPU_IDS[0]}" 0 8 6.0 1.0 5 6 "${COARSE_DIR}/nmin00_nmax08_tg6.0_sg1.0_pass4.tif" "B1"
run_multipass "${GPU_IDS[0]}" 0 9 5.0 1.0 5 6 "${COARSE_DIR}/nmin00_nmax09_tg5.0_sg1.0_pass4.tif" "B2"
run_multipass "${GPU_IDS[0]}" 0 9 6.0 1.0 5 6 "${COARSE_DIR}/nmin00_nmax09_tg6.0_sg1.0_pass4.tif" "B3"

ELAPSED=$(( $(date +%s) - T_START ))
echo -e "${CYAN}--- Group B done | Elapsed: ${ELAPSED}s ---${NC}"
echo ""

# ── Summary ──
T_END=$(date +%s)
T_ELAPSED=$(( T_END - T_START ))

echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  Grid Search Complete${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Total time: ${T_ELAPSED}s ($(echo "scale=1; ${T_ELAPSED}/60" | bc)min)"
echo ""
echo "Output files:"
ls -lhS "${OUTPUT_DIR}"/*.tif 2>/dev/null | head -25
echo "..."
echo "Total: $(ls "${OUTPUT_DIR}"/*.tif 2>/dev/null | wc -l) files"
echo ""
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
