#!/bin/bash

# Run FlowEdit dev + schnell on input/samsung TIF, GPU 5 only
# dev:     ts28_nmin01_nmax06_tg4.5_sg1.00_pass3
# schnell: ts08_nmin01_nmax03_pass2

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output/demo_input_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$OUTPUT_DIR"

LOGFILE="${OUTPUT_DIR}/run.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1

PYTHON="/media2/4tb/kevin/envs/skyfall/bin/python"
export HF_HOME=/media2/4tb/kevin/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

INPUT="${SCRIPT_DIR}/input/samsung_8_dense_0331_fix11_ldn02_s1c02_s301_albedo0_shadow0_noeval_i3000.tif"
DEVICE="cuda:5"
TILE_SIZE=1024
OVERLAP=128
GAMMA=0.7

SRC_PROMPT="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR_PROMPT="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"

T_START=$(date +%s)

# ---- 1) Dev: ts28_nmin01_nmax06_tg4.5_sg1.00_pass1-3 ----
echo "============================================"
echo "  [1/2] FlowEdit Dev (FLUX.1-dev)"
echo "  ts28_nmin01_nmax06_tg4.5_sg1.00_pass1-3"
echo "============================================"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

DEV_DIR="${OUTPUT_DIR}/dev"
mkdir -p "$DEV_DIR"

$PYTHON -u "${SCRIPT_DIR}/grid_worker.py" \
    --job_file "${SCRIPT_DIR}/input/jobs_demo_dev.json" \
    --input "$INPUT" \
    --output_dir "$DEV_DIR" \
    --tile_size "$TILE_SIZE" \
    --overlap "$OVERLAP" \
    --gamma "$GAMMA" \
    --max_pass 3 \
    --model "black-forest-labs/FLUX.1-dev" \
    --device "$DEVICE" \
    --src_prompt "$SRC_PROMPT" \
    --tar_prompt "$TAR_PROMPT"

echo ""
echo "Dev finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ---- 2) Schnell: ts08_nmin01_nmax03_pass1-2 ----
echo "============================================"
echo "  [2/2] FlowEdit Schnell (FLUX.1-schnell)"
echo "  ts08_nmin01_nmax03_pass1-2"
echo "============================================"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

SCHNELL_DIR="${OUTPUT_DIR}/schnell"
mkdir -p "$SCHNELL_DIR"

$PYTHON -u "${SCRIPT_DIR}/grid_worker.py" \
    --job_file "${SCRIPT_DIR}/input/jobs_demo_schnell.json" \
    --input "$INPUT" \
    --output_dir "$SCHNELL_DIR" \
    --tile_size "$TILE_SIZE" \
    --overlap "$OVERLAP" \
    --gamma "$GAMMA" \
    --max_pass 2 \
    --model "black-forest-labs/FLUX.1-schnell" \
    --device "$DEVICE" \
    --src_prompt "$SRC_PROMPT" \
    --tar_prompt "$TAR_PROMPT"

echo ""
echo "Schnell finished: $(date '+%Y-%m-%d %H:%M:%S')"

T_END=$(date +%s)
T_ELAPSED=$(( T_END - T_START ))
echo ""
echo "============================================"
echo "  All done! Total: ${T_ELAPSED}s"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"
