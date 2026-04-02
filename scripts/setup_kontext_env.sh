#!/bin/bash

# Create conda environment for FLUX Kontext
# Separate from skyfall env to avoid breaking FlowEdit (diffusers 0.30.1)

set -e

LOGDIR="$(cd "$(dirname "$0")/.." && pwd)/logs"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/setup_kontext_$(date '+%Y%m%d_%H%M%S').log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' >> "$LOGFILE")) 2>&1

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ENV_PATH=/data/kevin_workspace/envs/skyfall-kontext

echo ""
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
echo -e "${GREEN}Setting up skyfall-kontext environment${NC}"
echo ""

# Check conda
if ! command -v conda &>/dev/null; then
    echo -e "${RED}conda not found.${NC}"
    exit 1
fi
eval "$(conda shell.bash hook)"

# Create env
if [ -d "${ENV_PATH}" ]; then
    echo -e "${YELLOW}Removing existing env at ${ENV_PATH}...${NC}"
    conda env remove -p ${ENV_PATH} -y
fi

echo -e "${YELLOW}Creating conda env: ${ENV_PATH} (python 3.10)${NC}"
conda create -p ${ENV_PATH} python=3.10 -y
conda activate ${ENV_PATH}

# PyTorch (CUDA 12.1)
echo -e "${YELLOW}Installing PyTorch (CUDA 12.1)...${NC}"
pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# diffusers >= 0.35.0 for FluxKontextPipeline
echo -e "${YELLOW}Installing diffusers (latest) + dependencies...${NC}"
pip install --no-cache-dir \
    diffusers>=0.35.0 \
    transformers \
    accelerate \
    huggingface-hub \
    sentencepiece \
    tokenizers

# GeoTIFF dependencies
echo -e "${YELLOW}Installing GeoTIFF dependencies...${NC}"
pip install --no-cache-dir \
    rasterio \
    scipy \
    tqdm \
    numpy \
    Pillow

echo ""
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  skyfall-kontext setup complete!${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "  Python: ${ENV_PATH}/bin/python"
echo -e "  To activate: conda activate ${ENV_PATH}"
echo -e "${GREEN}==================================================${NC}"
echo ""
