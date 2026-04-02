#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export HF_HOME=/media2/4tb/kevin/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

PYTHON="/media2/4tb/kevin/envs/skyfall/bin/python"

$PYTHON -u grid_worker.py \
    --job_file output/grid_schnell_20260401_154140/jobs_gpu4.json \
    --input /media2/data/dataset_stereo/satelite/korea/seoul/gangnam/samsung/fused_top_naive.tif \
    --output_dir output/grid_schnell_20260401_154140 \
    --tile_size 1024 --overlap 128 --gamma 0.7 --max_pass 1 \
    --model black-forest-labs/FLUX.1-schnell \
    --device cuda:7 \
    --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution" \
    --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors" \
    2>&1 | stdbuf -oL tee output/grid_schnell_20260401_154140/worker_gpu7.log
