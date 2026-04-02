#!/bin/bash

export HF_HOME=/media2/4tb/kevin/.cache/huggingface

PYTHON="/media2/4tb/kevin/envs/skyfall/bin/python"

echo "Downloading FLUX.1-schnell (~23GB)..."
$PYTHON -c "
from diffusers import FluxPipeline
import torch
print('Downloading...')
pipe = FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-schnell', torch_dtype=torch.float16)
print('Done. Model cached.')
"
