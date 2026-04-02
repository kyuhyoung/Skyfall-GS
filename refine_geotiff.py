"""
Refine a GeoTIFF satellite image using FlowEdit + FLUX.

Usage:
    python refine_geotiff.py --input /path/to/input.tif --output /path/to/output.tif
    python refine_geotiff.py --input /path/to/input.tif --output /path/to/output.tif \
        --tile_size 1024 --overlap 128 --n_max 15
"""

import sys
sys.path.append("submodules/FlowEdit")

import os
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from FlowEdit_utils import FlowEditFLUX
from diffusers import FluxPipeline
from geotiff_utils import read_geotiff, save_geotiff, compute_tiles, make_blend_weight


def parse_args():
    parser = argparse.ArgumentParser(description="Refine GeoTIFF with FlowEdit + FLUX")
    parser.add_argument("--input", type=str, required=True, help="Input GeoTIFF path")
    parser.add_argument("--output", type=str, default=None, help="Output GeoTIFF path (default: input_refined.tif)")
    parser.add_argument("--tile_size", type=int, default=1024, help="Tile size (must be divisible by 16)")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between tiles for blending")
    parser.add_argument("--n_min", type=int, default=0, help="FlowEdit n_min parameter")
    parser.add_argument("--n_max", type=int, default=15, help="FlowEdit n_max parameter")
    parser.add_argument("--n_avg", type=int, default=1, help="FlowEdit n_avg parameter")
    parser.add_argument("--T_steps", type=int, default=28, help="Number of ODE steps")
    parser.add_argument("--src_guidance", type=float, default=1.5, help="Source guidance scale")
    parser.add_argument("--tar_guidance", type=float, default=5.5, help="Target guidance scale")
    parser.add_argument("--src_prompt", type=str,
                        default="Low resolution satellite image with noise, blurring, and artifacts",
                        help="Source prompt describing input image")
    parser.add_argument("--tar_prompt", type=str,
                        default="High resolution clean satellite image with sharp details and natural colors",
                        help="Target prompt describing desired output")
    parser.add_argument("--gamma", type=float, default=0.7,
                        help="Gamma correction before FLUX (0=disable, default: 0.7)")
    parser.add_argument("--save_preview", action="store_true",
                        help="Save pre-FLUX preview PNG (after stretch+gamma)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no_stretch", action="store_true",
                        help="Skip preprocessing (for multi-pass: input already processed)")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev",
                        help="HuggingFace model ID (default: black-forest-labs/FLUX.1-dev)")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def refine_tile(pipe, scheduler, tile_img, args):
    """Run FlowEdit on a single tile. Input/output: numpy HxWx3 float32 0-1."""
    # Ensure dimensions are divisible by 16
    h, w = tile_img.shape[:2]
    h_pad = (16 - h % 16) % 16
    w_pad = (16 - w % 16) % 16

    if h_pad > 0 or w_pad > 0:
        tile_img = np.pad(tile_img, ((0, h_pad), (0, w_pad), (0, 0)), mode="reflect")

    # To PIL -> VAE encode
    pil_img = Image.fromarray((tile_img * 255).clip(0, 255).astype(np.uint8))
    image_src = pipe.image_processor.preprocess(pil_img)
    image_src = image_src.to(args.device).half()

    with torch.autocast("cuda"), torch.inference_mode():
        x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
    x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    x0_src = x0_src.to(args.device)

    # FlowEdit
    x0_tar = FlowEditFLUX(
        pipe, scheduler, x0_src,
        args.src_prompt, args.tar_prompt, "",
        T_steps=args.T_steps, n_avg=args.n_avg,
        src_guidance_scale=args.src_guidance,
        tar_guidance_scale=args.tar_guidance,
        n_min=args.n_min, n_max=args.n_max
    )

    # Decode
    x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.autocast("cuda"), torch.inference_mode():
        image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
    image_tar = pipe.image_processor.postprocess(image_tar)[0]

    # PIL -> numpy
    result = np.array(image_tar).astype(np.float32) / 255.0

    # Remove padding
    if h_pad > 0 or w_pad > 0:
        result = result[:h, :w]

    return result


def main():
    args = parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_refined{ext}"

    assert args.tile_size % 16 == 0, "tile_size must be divisible by 16"

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed: {args.seed}")

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    # Read GeoTIFF
    print("Reading GeoTIFF...")
    img, profile, applied_gamma = read_geotiff(args.input, no_stretch=args.no_stretch, gamma=args.gamma)
    h, w, c = img.shape
    print(f"Image size: {w}x{h}, {c} bands")

    # Save pre-FLUX preview PNG
    if args.save_preview:
        output_dir = os.path.dirname(args.output)
        input_stem = os.path.splitext(os.path.basename(args.input))[0]
        preview_path = os.path.join(output_dir, input_stem + "_flux_input.png")
        preview = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(preview).save(preview_path)
        print(f"Saved pre-FLUX preview: {preview_path}")

    # Compute tiles
    tiles = compute_tiles(h, w, args.tile_size, args.overlap)
    print(f"Processing {len(tiles)} tiles (tile_size={args.tile_size}, overlap={args.overlap})")

    # Load FLUX
    model_id = args.model
    print(f"Loading FLUX model ({model_id})...")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # Auto: direct GPU if VRAM >= 30GB, else cpu_offload
    gpu_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
    vram_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 3)
    if vram_gb >= 30:
        pipe = pipe.to(args.device)
        print(f"VRAM {vram_gb:.0f}GB → direct GPU load")
    else:
        pipe.enable_model_cpu_offload(device=args.device)
        print(f"VRAM {vram_gb:.0f}GB → cpu_offload")
    scheduler = pipe.scheduler
    print("FLUX model loaded.")

    # Process tiles
    output = np.zeros((h, w, c), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    for idx, (y1, x1, y2, x2) in enumerate(tqdm(tiles, desc="Refining tiles")):
        # Per-tile deterministic seed
        torch.manual_seed(args.seed + idx)
        torch.cuda.manual_seed_all(args.seed + idx)
        tile = img[y1:y2, x1:x2].copy()
        tile_h, tile_w = tile.shape[:2]

        refined = refine_tile(pipe, scheduler, tile, args)
        weight = make_blend_weight(tile_h, tile_w, args.overlap)

        output[y1:y2, x1:x2] += refined * weight[:, :, None]
        weight_sum[y1:y2, x1:x2] += weight

    # Normalize by weights
    mask = weight_sum > 0
    for c_idx in range(c):
        output[:, :, c_idx][mask] /= weight_sum[mask]

    # Save
    print(f"Saving to {args.output}...")
    save_geotiff(args.output, output, profile)
    print("Done.")


if __name__ == "__main__":
    main()
