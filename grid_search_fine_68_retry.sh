#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export HF_HOME=/media2/4tb/kevin/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

PYTHON="/media2/4tb/kevin/envs/skyfall/bin/python"
OUTPUT_DIR="output/grid_fine_68"

# Job 1: nmin01_nmax07_tg4.5_sg1.0 pass 4-5 → GPU 0
echo '[{"n_min":1,"n_max":7,"tar_guidance":4.5,"src_guidance":1.0,"start_pass":4,"end_pass":5}]' > "$OUTPUT_DIR/jobs_retry_gpu0.json"

# Job 2: nmin02_nmax06_tg4.5_sg1.0 pass 1-4 → GPU 1
echo '[{"n_min":2,"n_max":6,"tar_guidance":4.5,"src_guidance":1.0,"start_pass":1,"end_pass":4}]' > "$OUTPUT_DIR/jobs_retry_gpu1.json"

$PYTHON -u grid_worker.py \
    --job_file "$OUTPUT_DIR/jobs_retry_gpu0.json" \
    --input /media2/data/dataset_stereo/satelite/korea/seoul/gangnam/samsung/fused_top_naive.tif \
    --output_dir "$OUTPUT_DIR" \
    --tile_size 1024 --overlap 128 --T_steps 28 --gamma 0.7 --max_pass 5 \
    --device cuda:0 \
    --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution" \
    --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors" \
    2>&1 | stdbuf -oL tee "$OUTPUT_DIR/worker_retry_gpu0.log" &

$PYTHON -u grid_worker.py \
    --job_file "$OUTPUT_DIR/jobs_retry_gpu1.json" \
    --input /media2/data/dataset_stereo/satelite/korea/seoul/gangnam/samsung/fused_top_naive.tif \
    --output_dir "$OUTPUT_DIR" \
    --tile_size 1024 --overlap 128 --T_steps 28 --gamma 0.7 --max_pass 5 \
    --device cuda:1 \
    --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution" \
    --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors" \
    2>&1 | stdbuf -oL tee "$OUTPUT_DIR/worker_retry_gpu1.log" &

wait
echo "Done."
