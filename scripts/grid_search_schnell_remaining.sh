#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

export HF_HOME=/media2/4tb/kevin/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

PYTHON="/media2/4tb/kevin/envs/skyfall/bin/python"
OUTPUT_DIR="output/grid_schnell_20260401_154140"
JOB_FILE="${OUTPUT_DIR}/jobs_remaining6.json"

GPUS=(0 1 3 4)
NUM_GPUS=${#GPUS[@]}

# Split into per-GPU job files
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_ID=${GPUS[$g]}
    GPU_JOB="${OUTPUT_DIR}/jobs_remaining_gpu${GPU_ID}.json"
    cat "$JOB_FILE" | $PYTHON -c "
import sys, json
jobs = json.load(sys.stdin)
my_jobs = [jobs[i] for i in range(${g}, len(jobs), ${NUM_GPUS})]
json.dump(my_jobs, sys.stdout, indent=2)
" > "$GPU_JOB"
    echo "GPU ${GPU_ID}: $($PYTHON -c "import json; print(len(json.load(open('${GPU_JOB}'))))")  jobs"
done

PIDS=()
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_ID=${GPUS[$g]}
    GPU_JOB="${OUTPUT_DIR}/jobs_remaining_gpu${GPU_ID}.json"
    LOG="${OUTPUT_DIR}/worker_remaining_gpu${GPU_ID}.log"

    $PYTHON -u grid_worker.py \
        --job_file "$GPU_JOB" \
        --input /media2/data/dataset_stereo/satelite/korea/seoul/gangnam/samsung/fused_top_naive.tif \
        --output_dir "$OUTPUT_DIR" \
        --tile_size 1024 --overlap 128 --gamma 0.7 --max_pass 1 \
        --model black-forest-labs/FLUX.1-schnell \
        --device "cuda:${GPU_ID}" \
        --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution" \
        --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors" \
        > "$LOG" 2>&1 &

    PIDS+=($!)
    echo "Launched GPU ${GPU_ID} (PID ${PIDS[-1]})"
done

echo "Waiting..."
FAILED=0
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_ID=${GPUS[$g]}
    if ! wait "${PIDS[$g]}"; then
        echo "GPU ${GPU_ID} FAILED"
        FAILED=$((FAILED + 1))
    else
        echo "GPU ${GPU_ID} done"
    fi
done

echo "Complete. Failed: ${FAILED}/${NUM_GPUS}"
echo "TIFs: $(ls ${OUTPUT_DIR}/*.tif 2>/dev/null | wc -l) / 112"
