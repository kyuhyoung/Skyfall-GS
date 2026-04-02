#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

export HF_HOME=/media2/4tb/kevin/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

PYTHON="/media2/4tb/kevin/envs/skyfall/bin/python"
OUTPUT_DIR="output/grid_schnell_20260401_154140"

echo '[{"n_min":0,"n_max":4,"tar_guidance":4.5,"src_guidance":0.5,"T_steps":8}]' > "$OUTPUT_DIR/jobs_last_gpu5.json"
echo '[{"n_min":1,"n_max":4,"tar_guidance":3.0,"src_guidance":0.5,"T_steps":8}]' > "$OUTPUT_DIR/jobs_last_gpu6.json"
echo '[{"n_min":1,"n_max":3,"tar_guidance":4.0,"src_guidance":1.0,"T_steps":8}]' > "$OUTPUT_DIR/jobs_last_gpu7.json"

PIDS=()
for GPU_ID in 5 6 7; do
    JOB="$OUTPUT_DIR/jobs_last_gpu${GPU_ID}.json"
    LOG="$OUTPUT_DIR/worker_last_gpu${GPU_ID}.log"

    $PYTHON -u grid_worker.py \
        --job_file "$JOB" \
        --input /media2/data/dataset_stereo/satelite/korea/seoul/gangnam/samsung/fused_top_naive.tif \
        --output_dir "$OUTPUT_DIR" \
        --tile_size 1024 --overlap 128 --gamma 0.7 --max_pass 1 \
        --model black-forest-labs/FLUX.1-schnell \
        --device "cuda:${GPU_ID}" \
        --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution" \
        --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors" \
        > "$LOG" 2>&1 &

    PIDS+=($!)
    echo "Launched GPU ${GPU_ID}"
done

echo "Waiting..."
for i in 0 1 2; do
    GPU_ID=$((5 + i))
    if ! wait "${PIDS[$i]}"; then
        echo "GPU ${GPU_ID} FAILED"
    else
        echo "GPU ${GPU_ID} done"
    fi
done

echo "TIFs: $(ls ${OUTPUT_DIR}/*.tif 2>/dev/null | wc -l) / 112"
