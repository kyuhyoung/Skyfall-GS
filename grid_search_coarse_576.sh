#!/bin/bash

# FlowEdit 1단계 coarse grid: 132 combos × 4 passes = 528 TIFs
# nmin=0,2,4 × nmax=4,6,8,10 (nmin<nmax) × tg=3,5,7,9 × sg=0.5,1.0,1.5
# 8 GPU 병렬: GPU당 모델 1회 로드 → 할당된 combos 순차 처리
#
# Usage:
#   ./grid_search_coarse_576.sh
#   ./grid_search_coarse_576.sh --gpus 0,1,2,3,4,5,6,7
#   ./grid_search_coarse_576.sh --max_concurrent 4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_coarse_576_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$OUTPUT_DIR"

LOGFILE="${OUTPUT_DIR}/grid_search.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse --gpus and --max_concurrent
GPUS="0,1,2,3,4"
MAX_CONCURRENT=5
for arg in "$@"; do
    if [ "$prev_arg" = "--gpus" ]; then
        GPUS="$arg"
        prev_arg=""
        continue
    fi
    if [ "$prev_arg" = "--max_concurrent" ]; then
        MAX_CONCURRENT="$arg"
        prev_arg=""
        continue
    fi
    if [ "$arg" = "--gpus" ] || [ "$arg" = "--max_concurrent" ]; then
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
MAX_PASS=4
SRC_PROMPT="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR_PROMPT="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"

echo ""
echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  FlowEdit Coarse Grid (model loaded once per GPU)${NC}"
echo -e "${CYAN}  nmin=0,2,4 × nmax=4,6,8,10 × tg=3,5,7,9 × sg=0.5,1.0,1.5${NC}"
echo -e "${CYAN}  pass=1,2,3,4 (chain dependency)${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPUs: ${GPUS} (${NUM_GPUS} total)"
echo "Python: ${PYTHON}"
echo "Input: ${INPUT}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Build combo list as JSON arrays, one per GPU (round-robin split)
COMBOS_JSON="["
FIRST=true
for NMIN in 0 2 4; do
    for NMAX in 4 6 8 10; do
        if [ "$NMIN" -ge "$NMAX" ]; then
            continue
        fi
        for TG in 3 5 7 9; do
            for SG in 0.5 1.0 1.5; do
                if [ "$FIRST" = true ]; then
                    FIRST=false
                else
                    COMBOS_JSON+=","
                fi
                COMBOS_JSON+="{\"n_min\":${NMIN},\"n_max\":${NMAX},\"tar_guidance\":${TG},\"src_guidance\":${SG}}"
            done
        done
    done
done
COMBOS_JSON+="]"

# Count total
TOTAL_COMBOS=$(echo "$COMBOS_JSON" | $PYTHON -c "import sys,json; print(len(json.load(sys.stdin)))")
TOTAL_TIFS=$((TOTAL_COMBOS * MAX_PASS))
echo "Total combos: ${TOTAL_COMBOS} (× ${MAX_PASS} passes = ${TOTAL_TIFS} TIFs)"
echo "Strategy: ${NUM_GPUS} workers (max ${MAX_CONCURRENT} concurrent), model loaded once per GPU"
echo ""

# Split combos into per-GPU job files (round-robin)
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_ID=${GPU_IDS[$g]}
    JOB_FILE="${OUTPUT_DIR}/jobs_gpu${GPU_ID}.json"
    echo "$COMBOS_JSON" | $PYTHON -c "
import sys, json
combos = json.load(sys.stdin)
gpu_idx = ${g}
num_gpus = ${NUM_GPUS}
my_jobs = [combos[i] for i in range(gpu_idx, len(combos), num_gpus)]
json.dump(my_jobs, sys.stdout, indent=2)
" > "$JOB_FILE"
    N_JOBS=$($PYTHON -c "import json; print(len(json.load(open('${JOB_FILE}'))))")
    echo -e "${YELLOW}GPU ${GPU_ID}: ${N_JOBS} combos → ${JOB_FILE}${NC}"
done
echo ""

T_START=$(date +%s)

# Launch workers in batches to avoid RAM OOM
# Each FLUX model uses ~30GB CPU RAM with enable_model_cpu_offload
# 251GB RAM → max ~4 concurrent workers safely
FAILED=0
NUM_BATCHES=$(( (NUM_GPUS + MAX_CONCURRENT - 1) / MAX_CONCURRENT ))

echo -e "${CYAN}Running ${NUM_GPUS} workers in ${NUM_BATCHES} batch(es) (max ${MAX_CONCURRENT} concurrent)${NC}"
echo ""

for ((batch=0; batch<NUM_BATCHES; batch++)); do
    BATCH_START=$((batch * MAX_CONCURRENT))
    BATCH_END=$((BATCH_START + MAX_CONCURRENT))
    if [ "$BATCH_END" -gt "$NUM_GPUS" ]; then
        BATCH_END=$NUM_GPUS
    fi
    BATCH_SIZE=$((BATCH_END - BATCH_START))

    echo -e "${CYAN}--- Batch $((batch+1))/${NUM_BATCHES}: GPUs ${GPU_IDS[@]:$BATCH_START:$BATCH_SIZE} ---${NC}"

    PIDS=()
    for ((g=BATCH_START; g<BATCH_END; g++)); do
        GPU_ID=${GPU_IDS[$g]}
        JOB_FILE="${OUTPUT_DIR}/jobs_gpu${GPU_ID}.json"
        WORKER_LOG="${OUTPUT_DIR}/worker_gpu${GPU_ID}.log"

        echo -e "${GREEN}Launching worker on GPU ${GPU_ID}...${NC}"

        # Stagger model loading within batch
        if [ "$g" -gt "$BATCH_START" ]; then
            sleep 60
        fi

        $PYTHON "${SCRIPT_DIR}/grid_worker.py" \
            --job_file "$JOB_FILE" \
            --input "$INPUT" \
            --output_dir "$OUTPUT_DIR" \
            --tile_size "$TILE_SIZE" \
            --overlap "$OVERLAP" \
            --T_steps "$T_STEPS" \
            --gamma "$GAMMA" \
            --max_pass "$MAX_PASS" \
            --device "cuda:${GPU_ID}" \
            --src_prompt "$SRC_PROMPT" \
            --tar_prompt "$TAR_PROMPT" \
            > "$WORKER_LOG" 2>&1 &

        PIDS+=($!)
    done

    echo ""
    echo -e "${CYAN}Batch $((batch+1)) launched. Waiting for completion...${NC}"
    echo -e "${CYAN}Monitor with: tail -f ${OUTPUT_DIR}/worker_gpu*.log${NC}"
    echo ""

    # Wait for batch to complete
    for ((i=0; i<BATCH_SIZE; i++)); do
        g=$((BATCH_START + i))
        GPU_ID=${GPU_IDS[$g]}
        if ! wait "${PIDS[$i]}"; then
            echo -e "\033[0;31m[ERROR] Worker GPU ${GPU_ID} failed. Check ${OUTPUT_DIR}/worker_gpu${GPU_ID}.log\033[0m"
            FAILED=$((FAILED + 1))
        else
            echo -e "${GREEN}Worker GPU ${GPU_ID} finished.${NC}"
        fi
    done

    if [ "$((batch+1))" -lt "$NUM_BATCHES" ]; then
        echo -e "${YELLOW}Batch $((batch+1)) done. Starting next batch in 10s...${NC}"
        sleep 10
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
echo "Combos: ${TOTAL_COMBOS}, TIFs: $(ls "${OUTPUT_DIR}"/*.tif 2>/dev/null | wc -l)"
echo ""
echo "Output files (first 20):"
ls -lhS "${OUTPUT_DIR}"/*.tif 2>/dev/null | head -20
echo "..."
echo "Total: $(ls "${OUTPUT_DIR}"/*.tif 2>/dev/null | wc -l) files"
echo ""
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
