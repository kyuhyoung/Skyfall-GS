#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

export HF_HOME=/media2/4tb/kevin/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

PYTHON="/media2/4tb/kevin/envs/skyfall/bin/python"
OUTPUT_DIR="output/grid_schnell_20260401_154140"

echo '[{"n_min":1,"n_max":4,"tar_guidance":3.0,"src_guidance":0.5,"T_steps":8}]' > "$OUTPUT_DIR/jobs_last_gpu1.json"
echo '[{"n_min":1,"n_max":3,"tar_guidance":4.0,"src_guidance":1.0,"T_steps":8}]' > "$OUTPUT_DIR/jobs_last_gpu2.json"

$PYTHON -u grid_worker.py \
    --job_file "$OUTPUT_DIR/jobs_last_gpu1.json" \
    --input /media2/data/dataset_stereo/satelite/korea/seoul/gangnam/samsung/fused_top_naive.tif \
    --output_dir "$OUTPUT_DIR" \
    --tile_size 1024 --overlap 128 --gamma 0.7 --max_pass 1 \
    --model black-forest-labs/FLUX.1-schnell \
    --device cuda:1 \
    --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution" \
    --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors" \
    > "$OUTPUT_DIR/worker_last_gpu1.log" 2>&1 &
PID1=$!
echo "Launched GPU 1"

$PYTHON -u grid_worker.py \
    --job_file "$OUTPUT_DIR/jobs_last_gpu2.json" \
    --input /media2/data/dataset_stereo/satelite/korea/seoul/gangnam/samsung/fused_top_naive.tif \
    --output_dir "$OUTPUT_DIR" \
    --tile_size 1024 --overlap 128 --gamma 0.7 --max_pass 1 \
    --model black-forest-labs/FLUX.1-schnell \
    --device cuda:2 \
    --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution" \
    --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors" \
    > "$OUTPUT_DIR/worker_last_gpu2.log" 2>&1 &
PID2=$!
echo "Launched GPU 2"

echo "Waiting..."
wait $PID1 && echo "GPU 1 done" || echo "GPU 1 FAILED"
wait $PID2 && echo "GPU 2 done" || echo "GPU 2 FAILED"

echo "TIFs: $(ls ${OUTPUT_DIR}/*.tif 2>/dev/null | wc -l) / 112"
