#!/bin/bash

# FlowEdit fine grid: 24 jobs = 68 TIFs
# 5 GPU 병렬, start_pass/end_pass 지원
#
# Usage:
#   ./grid_search_fine_68.sh
#   ./grid_search_fine_68.sh --gpus 0,1,2,3,4

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_fine_68"
mkdir -p "$OUTPUT_DIR"

LOGFILE="${OUTPUT_DIR}/grid_search.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

GPUS="0,1,2,3,4"
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

export HF_HOME=/media2/4tb/kevin/.cache/huggingface
export HF_HUB_OFFLINE=1

PYTHON="/media2/4tb/kevin/envs/skyfall/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="$(which python)"
fi
INPUT="/media2/data/dataset_stereo/satelite/korea/seoul/gangnam/samsung/fused_top_naive.tif"

T_STEPS=28
GAMMA=0.7
TILE_SIZE=1024
OVERLAP=128
SRC_PROMPT="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR_PROMPT="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"

JOB_FILE="${OUTPUT_DIR}/jobs.json"
TOTAL_JOBS=$($PYTHON -c "import json; print(len(json.load(open('${JOB_FILE}'))))")
TOTAL_TIFS=$($PYTHON -c "import json; print(sum(j.get('end_pass',4)-j.get('start_pass',1)+1 for j in json.load(open('${JOB_FILE}'))))")

echo ""
echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  FlowEdit Fine Grid (${TOTAL_JOBS} jobs, ${TOTAL_TIFS} TIFs)${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPUs: ${GPUS} (${NUM_GPUS} total)"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Split jobs into per-GPU job files (round-robin)
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_ID=${GPU_IDS[$g]}
    GPU_JOB_FILE="${OUTPUT_DIR}/jobs_gpu${GPU_ID}.json"
    cat "$JOB_FILE" | $PYTHON -c "
import sys, json
jobs = json.load(sys.stdin)
gpu_idx = ${g}
num_gpus = ${NUM_GPUS}
my_jobs = [jobs[i] for i in range(gpu_idx, len(jobs), num_gpus)]
json.dump(my_jobs, sys.stdout, indent=2)
" > "$GPU_JOB_FILE"
    N_JOBS=$($PYTHON -c "import json; print(len(json.load(open('${GPU_JOB_FILE}'))))")
    echo -e "${YELLOW}GPU ${GPU_ID}: ${N_JOBS} jobs → ${GPU_JOB_FILE}${NC}"
done
echo ""

T_START=$(date +%s)
FAILED=0

PIDS=()
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_ID=${GPU_IDS[$g]}
    GPU_JOB_FILE="${OUTPUT_DIR}/jobs_gpu${GPU_ID}.json"
    WORKER_LOG="${OUTPUT_DIR}/worker_gpu${GPU_ID}.log"

    echo -e "${GREEN}Launching worker on GPU ${GPU_ID}...${NC}"

    $PYTHON "${SCRIPT_DIR}/grid_worker.py" \
        --job_file "$GPU_JOB_FILE" \
        --input "$INPUT" \
        --output_dir "$OUTPUT_DIR" \
        --tile_size "$TILE_SIZE" \
        --overlap "$OVERLAP" \
        --T_steps "$T_STEPS" \
        --gamma "$GAMMA" \
        --max_pass 5 \
        --device "cuda:${GPU_ID}" \
        --src_prompt "$SRC_PROMPT" \
        --tar_prompt "$TAR_PROMPT" \
        > "$WORKER_LOG" 2>&1 &

    PIDS+=($!)
done

echo ""
echo -e "${CYAN}All workers launched. Waiting for completion...${NC}"
echo -e "${CYAN}Monitor with: tail -f ${OUTPUT_DIR}/worker_gpu*.log${NC}"
echo ""

for ((g=0; g<NUM_GPUS; g++)); do
    GPU_ID=${GPU_IDS[$g]}
    if ! wait "${PIDS[$g]}"; then
        echo -e "\033[0;31m[ERROR] Worker GPU ${GPU_ID} failed. Check ${OUTPUT_DIR}/worker_gpu${GPU_ID}.log\033[0m"
        FAILED=$((FAILED + 1))
    else
        echo -e "${GREEN}Worker GPU ${GPU_ID} finished.${NC}"
    fi
done

T_END=$(date +%s)
T_ELAPSED=$(( T_END - T_START ))

echo ""
echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  Grid Search Complete${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Total time: ${T_ELAPSED}s ($(echo "scale=1; ${T_ELAPSED}/60" | bc)min)"
echo "Workers failed: ${FAILED}/${NUM_GPUS}"
echo "TIFs: $(ls "${OUTPUT_DIR}"/*.tif 2>/dev/null | wc -l)"
echo ""
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
