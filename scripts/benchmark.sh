#!/bin/bash

# Benchmark: single GPU (original) vs multi GPU parallel vs multi GPU + auto tile
#
# Run 1: Single GPU, tile 1024         — git clone 원본 그대로
# Run 2: Multi GPU,  tile 1024         — 순수 GPU 병렬 효과
# Run 3: Multi GPU,  auto tile (VRAM)  — 큰 타일 + 병렬 합산 효과
#
# Usage:
#   ./benchmark.sh                    # auto-detect GPUs
#   ./benchmark.sh --gpus 0,1,2,3     # specify GPUs

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="${SCRIPT_DIR}/output/benchmark_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$BENCH_DIR"

LOGFILE="${BENCH_DIR}/benchmark.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse --gpus
GPUS=""
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

INPUT="/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif"

# Shared parameters (identical across all runs)
COMMON_ARGS=(
    --input "$INPUT"
    --overlap 128
    --n_min 0
    --n_max 7
    --n_avg 1
    --T_steps 28
    --gamma 1.0
    --tar_guidance 5.5
    --src_guidance 1.0
    --src_prompt "Satellite image with black missing regions, noise, blurring, and low resolution"
    --tar_prompt "Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"
)

GPU_ARGS=""
if [ -n "$GPUS" ]; then
    GPU_ARGS="--gpus $GPUS"
fi

echo ""
echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  Benchmark: Single GPU vs Multi GPU (parallel) vs Auto Tile${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Input: ${INPUT}"
echo "GPUs: ${GPUS:-auto-detect}"
echo "Output dir: ${BENCH_DIR}"
echo ""

clear_gpu() {
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
    sleep 3
}

##############################################################################
# Run 1: Single GPU, tile_size=1024 (original git clone behavior)
##############################################################################
echo -e "${YELLOW}[Run 1/3] Single GPU — tile_size=1024, cuda:0${NC}"
echo "  Original behavior (refine_geotiff.py)"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
T1_START=$(date +%s)

python "${SCRIPT_DIR}/refine_geotiff.py" \
    --output "${BENCH_DIR}/run1_single_gpu_tile1024.tif" \
    --tile_size 1024 \
    --device cuda:0 \
    --save_preview \
    "${COMMON_ARGS[@]}"

T1_END=$(date +%s)
T1_ELAPSED=$(( T1_END - T1_START ))
echo -e "${GREEN}[Run 1/3] Done. Elapsed: ${T1_ELAPSED}s ($(echo "scale=1; ${T1_ELAPSED}/60" | bc)min)${NC}"
echo ""

clear_gpu

##############################################################################
# Run 2: Multi GPU, tile_size=1024 (pure GPU parallelism)
##############################################################################
echo -e "${YELLOW}[Run 2/3] Multi GPU — tile_size=1024 (pure parallelism)${NC}"
echo "  Same tile size as Run 1, but tiles distributed across GPUs"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
T2_START=$(date +%s)

python "${SCRIPT_DIR}/launch_refine.py" \
    $GPU_ARGS \
    --output "${BENCH_DIR}/run2_multi_gpu_tile1024.tif" \
    --tile_size 1024 \
    --save_preview \
    "${COMMON_ARGS[@]}"

T2_END=$(date +%s)
T2_ELAPSED=$(( T2_END - T2_START ))
echo -e "${GREEN}[Run 2/3] Done. Elapsed: ${T2_ELAPSED}s ($(echo "scale=1; ${T2_ELAPSED}/60" | bc)min)${NC}"
echo ""

clear_gpu

##############################################################################
# Run 3: Multi GPU, auto tile_size (VRAM-based)
##############################################################################
echo -e "${YELLOW}[Run 3/3] Multi GPU — auto tile_size (VRAM-based)${NC}"
echo "  Larger tiles + GPU parallelism"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
T3_START=$(date +%s)

python "${SCRIPT_DIR}/launch_refine.py" \
    $GPU_ARGS \
    --output "${BENCH_DIR}/run3_multi_gpu_auto_tile.tif" \
    --save_preview \
    "${COMMON_ARGS[@]}"

T3_END=$(date +%s)
T3_ELAPSED=$(( T3_END - T3_START ))
echo -e "${GREEN}[Run 3/3] Done. Elapsed: ${T3_ELAPSED}s ($(echo "scale=1; ${T3_ELAPSED}/60" | bc)min)${NC}"
echo ""

##############################################################################
# Summary
##############################################################################
speedup() {
    if [ "$1" -gt 0 ] && [ "$2" -gt 0 ]; then
        echo "$(echo "scale=2; $1 / $2" | bc)x"
    else
        echo "N/A"
    fi
}

S_2vs1=$(speedup $T1_ELAPSED $T2_ELAPSED)
S_3vs1=$(speedup $T1_ELAPSED $T3_ELAPSED)

echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  Benchmark Results${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo ""
echo -e "  Run 1  1 GPU,  tile 1024 (baseline):  ${T1_ELAPSED}s  ($(echo "scale=1; ${T1_ELAPSED}/60" | bc)min)"
echo -e "  Run 2  N GPU,  tile 1024 (parallel):  ${T2_ELAPSED}s  ($(echo "scale=1; ${T2_ELAPSED}/60" | bc)min)  → ${S_2vs1} speedup"
echo -e "  Run 3  N GPU,  auto tile (full):      ${T3_ELAPSED}s  ($(echo "scale=1; ${T3_ELAPSED}/60" | bc)min)  → ${S_3vs1} speedup"
echo ""
echo -e "${CYAN}==============================================================${NC}"
echo ""
echo "Output files:"
ls -lh "${BENCH_DIR}"/*.tif 2>/dev/null
echo ""

##############################################################################
# Comparison PNGs
##############################################################################
echo -e "${YELLOW}Generating comparison images...${NC}"

python - "${BENCH_DIR}" <<'PYEOF'
import sys, os
import numpy as np
import rasterio
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bench_dir = sys.argv[1]
run1_path = os.path.join(bench_dir, "run1_single_gpu_tile1024.tif")
run2_path = os.path.join(bench_dir, "run2_multi_gpu_tile1024.tif")
run3_path = os.path.join(bench_dir, "run3_multi_gpu_auto_tile.tif")

def read_tif_rgb(path):
    with rasterio.open(path) as src:
        data = src.read()  # (C, H, W)
    return data.transpose(1, 2, 0).astype(np.float32)  # (H, W, C)

imgs = {}
for name, path in [("run1", run1_path), ("run2", run2_path), ("run3", run3_path)]:
    if os.path.exists(path):
        imgs[name] = read_tif_rgb(path)

if len(imgs) < 2:
    print("Not enough TIF files for comparison.")
    sys.exit(0)

# --- 1. Side-by-side RGB ---
fig, axes = plt.subplots(1, len(imgs), figsize=(6 * len(imgs), 8))
if len(imgs) == 1:
    axes = [axes]
titles = {
    "run1": "Run1: 1GPU tile1024",
    "run2": "Run2: NGPU tile1024",
    "run3": "Run3: NGPU auto tile",
}
for ax, (name, img) in zip(axes, imgs.items()):
    ax.imshow(np.clip(img / 255.0, 0, 1))
    ax.set_title(titles.get(name, name), fontsize=12)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(bench_dir, "comparison_rgb.png"), dpi=150)
plt.close()
print("  comparison_rgb.png")

# --- 2. Per-band histograms ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
band_names = ["Red", "Green", "Blue"]
for b, (ax, bname) in enumerate(zip(axes, band_names)):
    for name, img in imgs.items():
        vals = img[:, :, b].ravel()
        ax.hist(vals, bins=256, range=(0, 255), alpha=0.4, label=titles.get(name, name))
    ax.set_title(bname)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 255)
plt.tight_layout()
plt.savefig(os.path.join(bench_dir, "comparison_histograms.png"), dpi=150)
plt.close()
print("  comparison_histograms.png")

# --- 3. Difference heatmaps (run1 vs run2, run1 vs run3) ---
pairs = []
if "run1" in imgs and "run2" in imgs:
    pairs.append(("run1", "run2"))
if "run1" in imgs and "run3" in imgs:
    pairs.append(("run1", "run3"))

if pairs:
    fig, axes = plt.subplots(len(pairs), 2, figsize=(12, 5 * len(pairs)))
    if len(pairs) == 1:
        axes = [axes]
    for row, (a, b) in enumerate(pairs):
        diff = imgs[a].astype(np.float32) - imgs[b].astype(np.float32)
        abs_diff = np.abs(diff).mean(axis=2)
        signed_diff = diff.mean(axis=2)

        ax1, ax2 = axes[row]
        im1 = ax1.imshow(signed_diff, cmap="RdBu_r", vmin=-50, vmax=50)
        ax1.set_title(f"Signed diff: {titles[a]} - {titles[b]}")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        im2 = ax2.imshow(abs_diff, cmap="hot", vmin=0, vmax=50)
        ax2.set_title(f"Abs diff: {titles[a]} vs {titles[b]}")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(bench_dir, "comparison_diff.png"), dpi=150)
    plt.close()
    print("  comparison_diff.png")

# --- 4. Diff histogram ---
if pairs:
    fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 4))
    if len(pairs) == 1:
        axes = [axes]
    for ax, (a, b) in zip(axes, pairs):
        diff = (imgs[a].astype(np.float32) - imgs[b].astype(np.float32)).ravel()
        ax.hist(diff, bins=256, range=(-128, 128), alpha=0.7)
        ax.set_title(f"Pixel diff: {titles[a]} - {titles[b]}")
        ax.set_xlabel("Difference")
        mean_abs = np.abs(diff).mean()
        ax.axvline(0, color="r", linestyle="--", alpha=0.5)
        ax.text(0.95, 0.95, f"mean|diff|={mean_abs:.1f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(bench_dir, "comparison_diff_histogram.png"), dpi=150)
    plt.close()
    print("  comparison_diff_histogram.png")

# --- 5. Stats summary ---
print("\n  Per-run band stats:")
for name, img in imgs.items():
    for b, bname in enumerate(band_names):
        vals = img[:, :, b]
        print(f"    {titles.get(name, name)} [{bname}] mean={vals.mean():.1f} std={vals.std():.1f} min={vals.min():.0f} max={vals.max():.0f}")

if pairs:
    print("\n  Pairwise comparison:")
    for a, b in pairs:
        diff = np.abs(imgs[a].astype(np.float32) - imgs[b].astype(np.float32))
        pct_gt20 = (diff > 20).mean() * 100
        pct_gt50 = (diff > 50).mean() * 100
        print(f"    {titles[a]} vs {titles[b]}: mean_abs_diff={diff.mean():.1f}, >20: {pct_gt20:.1f}%, >50: {pct_gt50:.1f}%")

print("\nDone.")
PYEOF

echo ""
echo -e "${GREEN}Comparison images saved to ${BENCH_DIR}/${NC}"
echo ""
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
