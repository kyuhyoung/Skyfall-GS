#!/bin/bash

# Skyfall-GS (FlowEdit) Conda Environment Setup Script
#
# Docker-like behavior:
#   - Script unchanged → reuse existing env (instant)
#   - Script changed   → recreate env from scratch
#   - No env yet       → create env from scratch
#
# Usage:
#   ./using_conda.sh            # auto-detect: reuse or rebuild
#   ./using_conda.sh --force    # force rebuild even if unchanged

set -e

# Log file (timestamped per run)
LOGDIR="$(dirname "$0")/logs"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/using_conda_$(date '+%Y%m%d_%H%M%S').log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging
exec > >(tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' >> "$LOGFILE")) 2>&1
echo ""
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

ENV_NAME=skyfall
ENV_PATH=/media2/4tb/kevin/envs/skyfall
HASH_FILE="${ENV_PATH}/.setup_hash"

# Options
FORCE=false
for arg in "$@"; do
    case $arg in
        --force) FORCE=true ;;
    esac
done

# Check conda is available
if ! command -v conda &>/dev/null; then
    echo -e "${RED}conda not found. Install Miniconda/Anaconda first.${NC}"
    exit 1
fi

# Source conda for subshell
eval "$(conda shell.bash hook)"

# Compute hash of this script
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
CURRENT_HASH=$(sha256sum "$SCRIPT_PATH" | cut -d' ' -f1)

# Check if rebuild is needed
NEED_BUILD=true
if [ "$FORCE" = true ]; then
    echo -e "${YELLOW}--force: rebuilding environment${NC}"
elif [ ! -d "${ENV_PATH}" ]; then
    echo -e "${YELLOW}Environment not found. Building...${NC}"
elif [ ! -f "${HASH_FILE}" ]; then
    echo -e "${YELLOW}No hash record found. Rebuilding...${NC}"
else
    SAVED_HASH=$(cat "${HASH_FILE}")
    if [ "$CURRENT_HASH" = "$SAVED_HASH" ]; then
        NEED_BUILD=false
        echo -e "${GREEN}Script unchanged (hash match). Reusing existing environment.${NC}"
    else
        echo -e "${YELLOW}Script changed (hash mismatch). Rebuilding...${NC}"
    fi
fi

# If no build needed, just activate and exit
if [ "$NEED_BUILD" = false ]; then
    exec 1>/dev/tty 2>/dev/tty
    SHELL_NAME=$(basename "$SHELL")
    if [ "$SHELL_NAME" = "fish" ]; then
        exec fish -C "conda activate ${ENV_PATH}; echo -e '(${ENV_NAME}) environment active.'"
    else
        exec bash --rcfile <(echo "source ~/.bashrc; conda activate ${ENV_PATH}; echo -e '${GREEN}(${ENV_NAME}) environment active.${NC}'")
    fi
fi

####################################################################################
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  Skyfall-GS (FlowEdit) Conda Environment Setup${NC}"
echo -e "${GREEN}==================================================${NC}"

# Remove old env if exists
if [ -d "${ENV_PATH}" ]; then
    echo -e "${YELLOW}Removing old environment...${NC}"
    conda env remove -p ${ENV_PATH} -y
fi

echo -e "${YELLOW}Creating conda environment: ${ENV_PATH} (python 3.10)${NC}"
mkdir -p "$(dirname ${ENV_PATH})"
conda create -p ${ENV_PATH} python=3.10 -y
conda activate ${ENV_PATH}

# HuggingFace cache → 4TB disk
export HF_HOME=/media2/4tb/kevin/.cache/huggingface
mkdir -p "$HF_HOME"

####################################################################################
#   Install PyTorch (CUDA 12.1 — compatible with H100)
echo -e "${YELLOW}Installing PyTorch (CUDA 12.1)...${NC}"
pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

####################################################################################
#   Install CLIP (needs --no-build-isolation)
echo -e "${YELLOW}Installing CLIP...${NC}"
pip install --no-cache-dir --no-build-isolation \
    "clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"

####################################################################################
#   Install FlowEdit dependencies (diffusers, transformers, etc.)
echo -e "${YELLOW}Installing FlowEdit dependencies...${NC}"
pip install --no-cache-dir \
    accelerate==1.0.1 \
    diffusers==0.30.1 \
    huggingface-hub==0.33.4 \
    transformers==4.46.3 \
    tokenizers==0.20.3 \
    sentencepiece

####################################################################################
#   Install remaining requirements
echo -e "${YELLOW}Installing remaining dependencies...${NC}"
pip install --no-cache-dir \
    clean-fid \
    torchmetrics \
    open3d \
    plyfile \
    ninja \
    GPUtil \
    opencv-python \
    lpips \
    OpenEXR \
    mediapy \
    absl-py \
    pyiqa \
    rasterio \
    scipy \
    tqdm \
    matplotlib

####################################################################################
#   Build CUDA submodules
echo -e "${YELLOW}Building CUDA submodules...${NC}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export TORCH_CUDA_ARCH_LIST="7.5"
export FORCE_CUDA=1

pip install --no-cache-dir --no-build-isolation "${SCRIPT_DIR}/submodules/diff-gaussian-rasterization-depth"
pip install --no-cache-dir --no-build-isolation "${SCRIPT_DIR}/submodules/simple-knn"
pip install --no-cache-dir --no-build-isolation "${SCRIPT_DIR}/submodules/fused-ssim"

####################################################################################
#   System dependencies check
echo -e "${YELLOW}Checking system libraries...${NC}"
MISSING=""
dpkg -s gdal-bin &>/dev/null || MISSING="$MISSING gdal-bin"
dpkg -s libgdal-dev &>/dev/null || MISSING="$MISSING libgdal-dev"
dpkg -s libgl1-mesa-glx &>/dev/null || MISSING="$MISSING libgl1-mesa-glx"

if [ -n "$MISSING" ]; then
    echo -e "${YELLOW}Missing system packages:${MISSING}${NC}"
    echo -e "${YELLOW}Install with: sudo apt-get install -y${MISSING}${NC}"
else
    echo -e "${GREEN}All system libraries present${NC}"
fi

####################################################################################
#   Save hash for future runs
echo "$CURRENT_HASH" > "${HASH_FILE}"

echo ""
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "${YELLOW}To activate:  conda activate ${ENV_PATH}${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

# Drop into activated shell
exec 1>/dev/tty 2>/dev/tty
SHELL_NAME=$(basename "$SHELL")
if [ "$SHELL_NAME" = "fish" ]; then
    exec fish -C "conda activate ${ENV_PATH}; echo -e '(${ENV_NAME}) environment active.'"
else
    exec bash --rcfile <(echo "source ~/.bashrc; conda activate ${ENV_PATH}; echo -e '${GREEN}(${ENV_NAME}) environment active.${NC}'")
fi
