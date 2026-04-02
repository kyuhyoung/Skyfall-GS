#!/bin/bash

# Refine GeoTIFF (multi-pass, multi-GPU support)
#
# Methods:
#   flowedit  — FlowEdit + FLUX.1-dev  (skyfall env, diffusers 0.30.1)
#   kontext   — FLUX Kontext           (skyfall-kontext env, diffusers 0.35+)
#
# Usage:
#   ./run_refine.sh                              # flowedit (default)
#   ./run_refine.sh --method kontext             # kontext
#   ./run_refine.sh --gpus 0,1,2,3              # specify GPUs
#   ./run_refine.sh --method kontext --gpus 4,5,6,7 --passes 1

# Log file
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${SCRIPT_DIR}/output"
LOGFILE="${SCRIPT_DIR}/output/run_refine.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1
echo ""
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

OUTPUT_DIR="${SCRIPT_DIR}/output/flux_grid_search"
mkdir -p "$OUTPUT_DIR"

# Parse arguments
PASSES=1
START_PASS=1
INPUT_OVERRIDE=""
GPUS="4,5,6,7"
METHOD="flux2"
EXTRA_ARGS=()
for arg in "$@"; do
    if [ "$prev_arg" = "--passes" ]; then
        PASSES="$arg"
        prev_arg=""
        continue
    fi
    if [ "$prev_arg" = "--start-pass" ]; then
        START_PASS="$arg"
        prev_arg=""
        continue
    fi
    if [ "$prev_arg" = "--input" ]; then
        INPUT_OVERRIDE="$arg"
        prev_arg=""
        continue
    fi
    if [ "$prev_arg" = "--gpus" ]; then
        GPUS="$arg"
        prev_arg=""
        continue
    fi
    if [ "$prev_arg" = "--method" ]; then
        METHOD="$arg"
        prev_arg=""
        continue
    fi
    if [ "$arg" = "--passes" ] || [ "$arg" = "--start-pass" ] || [ "$arg" = "--input" ] || [ "$arg" = "--gpus" ] || [ "$arg" = "--method" ]; then
        prev_arg="$arg"
        continue
    fi
    EXTRA_ARGS+=("$arg")
done

# Select python env based on method
if [ "$METHOD" = "kontext" ] || [ "$METHOD" = "flux2" ]; then
    PYTHON="/data/kevin_workspace/envs/skyfall-kontext/bin/python"
else
    PYTHON="/data/kevin_workspace/envs/skyfall/bin/python"
fi

# 241 서버 용 tif
# INPUT="${INPUT_OVERRIDE:-/data/satellite/seoul/gangnam/samsung/gwarp_out_ps_ba/fused_top_naive.tif}"
INPUT="${INPUT_OVERRIDE:-/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif}"
# Always use original stem for output naming
INPUT_STEM="fused_top_naive"
END_PASS=$(( START_PASS + PASSES - 1 ))
# GPU args for launch_refine.py
GPU_ARGS=""
if [ -n "$GPUS" ]; then
    GPU_ARGS="--gpus $GPUS"
fi

echo "Method: ${METHOD}"
echo "Python: ${PYTHON}"
echo "Passes: ${START_PASS} to ${END_PASS}"
echo "Input: ${INPUT}"
echo "GPUs: ${GPUS:-auto-detect}"
echo ""

CURRENT_INPUT="$INPUT"
for P in $(seq "$START_PASS" "$END_PASS"); do
    echo "===== Pass ${P}/${PASSES} ====="

    if [ "$METHOD" = "flux2" ]; then
        # FLUX.2 FlowEdit params
        N_MIN=0
        N_MAX=28
        SRC_GUIDANCE=1.0
        TAR_GUIDANCE=5.5
        TAG=$(printf "flux2fe_nmin%02d_nmax%02d_sg%.1f_tg%.1f" "$N_MIN" "$N_MAX" "$SRC_GUIDANCE" "$TAR_GUIDANCE")

        if [ "$END_PASS" -eq 1 ] && [ "$START_PASS" -eq 1 ]; then
            OUTPUT="${OUTPUT_DIR}/${INPUT_STEM}_${TAG}.tif"
        else
            OUTPUT="${OUTPUT_DIR}/${INPUT_STEM}_${TAG}_pass${P}.tif"
        fi

        CUDA_VISIBLE_DEVICES=4,5,6,7 $PYTHON "${SCRIPT_DIR}/refine_geotiff_flux2.py" \
            --input "$CURRENT_INPUT" \
            --output "$OUTPUT" \
            --flowedit \
            --n_min "$N_MIN" \
            --n_max "$N_MAX" \
            --src_guidance "$SRC_GUIDANCE" \
            --tar_guidance "$TAR_GUIDANCE" \
            --num_steps 28 \
            --save_preview \
            --device cuda:0 \
            --src_prompt "Satellite image with noise, blurring, and low resolution" \
            --prompt "High resolution clean satellite orthographic top-down image with sharp details, vivid colors, buildings, roads, and vegetation" \
            "${EXTRA_ARGS[@]}"

    elif [ "$METHOD" = "kontext" ]; then
        # Kontext params
        GUIDANCE=2.5
        TRUE_CFG=3.0
        G_TAG=$(printf "g%05.1f_tc%04.1f" "$GUIDANCE" "$TRUE_CFG")

        if [ "$END_PASS" -eq 1 ] && [ "$START_PASS" -eq 1 ]; then
            OUTPUT="${OUTPUT_DIR}/${INPUT_STEM}_kontext_${G_TAG}.tif"
        else
            OUTPUT="${OUTPUT_DIR}/${INPUT_STEM}_kontext_${G_TAG}_pass${P}.tif"
        fi

        $PYTHON "${SCRIPT_DIR}/refine_geotiff_kontext.py" \
            --input "$CURRENT_INPUT" \
            --output "$OUTPUT" \
            --whole \
            --guidance_scale "$GUIDANCE" \
            --true_cfg "$TRUE_CFG" \
            --num_steps 28 \
            --gamma 1.0 \
            --save_preview \
            --device cuda:7 \
            --prompt "Enhance this satellite image: sharpen details, fill missing regions naturally with buildings roads and vegetation, improve resolution and colors" \
            "${EXTRA_ARGS[@]}"
    else
        # FlowEdit output naming
        if [ "$END_PASS" -eq 1 ] && [ "$START_PASS" -eq 1 ]; then
            OUTPUT="${OUTPUT_DIR}/${INPUT_STEM}_flux_nmin00_nmax07_tg5.5_sg1.0.tif"
        else
            OUTPUT="${OUTPUT_DIR}/${INPUT_STEM}_flux_nmin00_nmax07_tg5.5_sg1.0_pass${P}.tif"
        fi

        $PYTHON "${SCRIPT_DIR}/launch_refine.py" \
            --method flowedit \
            $GPU_ARGS \
            --input "$CURRENT_INPUT" \
            --output "$OUTPUT" \
            --overlap 128 \
            --n_min 0 \
            --n_max 7 \
            --n_avg 1 \
            --T_steps 28 \
            --gamma 1.0 \
            --save_preview \
            --tar_guidance 5.5 \
            --src_guidance 1.0 \
            --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution" \
            --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors" \
            "${EXTRA_ARGS[@]}"
    fi

    CURRENT_INPUT="$OUTPUT"
    echo ""
done

echo "========== All ${PASSES} passes complete =========="
echo "$(date '+%Y-%m-%d %H:%M:%S')"
