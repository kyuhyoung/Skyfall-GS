#!/bin/bash

# Skyfall-GS (FlowEdit) Conda Environment Setup Script
#
# Usage:
#   ./using_conda.sh            # create env + activate
#   ./using_conda.sh -r         # activate existing env (fast)

set -e

# Log file
LOGFILE="$(dirname "$0")/using_conda.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging
exec > >(tee >(sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1
echo ""
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

ENV_NAME=skyfall
ENV_PATH=/data/kevin_workspace/envs/skyfall

# Check for options
REUSE=false
for arg in "$@"; do
    case $arg in
        -r)
            REUSE=true
            ;;
    esac
done

# Check conda is available
if ! command -v conda &>/dev/null; then
    echo -e "${RED}conda not found. Install Miniconda/Anaconda first.${NC}"
    exit 1
fi

# Source conda for subshell
eval "$(conda shell.bash hook)"

if [ "$REUSE" = true ]; then
    if [ -d "${ENV_PATH}" ]; then
        echo -e "${GREEN}Activating existing environment: ${ENV_PATH}${NC}"
        exec 1>/dev/tty 2>/dev/tty
        SHELL_NAME=$(basename "$SHELL")
        if [ "$SHELL_NAME" = "fish" ]; then
            exec fish -C "conda activate ${ENV_PATH}; echo -e '(${ENV_NAME}) environment active.'"
        else
            exec bash --rcfile <(echo "source ~/.bashrc; conda activate ${ENV_PATH}; echo -e '${GREEN}(${ENV_NAME}) environment active.${NC}'")
        fi
    else
        echo -e "${RED}Environment not found at ${ENV_PATH}. Run without -r first.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  Skyfall-GS (FlowEdit) Conda Environment Setup${NC}"
echo -e "${GREEN}==================================================${NC}"

####################################################################################
#   Create conda environment
if [ -d "${ENV_PATH}" ]; then
    echo -e "${YELLOW}Environment already exists at ${ENV_PATH}. Removing...${NC}"
    conda env remove -p ${ENV_PATH} -y
fi

echo -e "${YELLOW}Creating conda environment: ${ENV_PATH} (python 3.10)${NC}"
mkdir -p "$(dirname ${ENV_PATH})"
conda create -p ${ENV_PATH} python=3.10 -y

conda activate ${ENV_PATH}

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

export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
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
