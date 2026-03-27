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
import rasterio
from PIL import Image
from tqdm import tqdm
from FlowEdit_utils import FlowEditFLUX
from diffusers import FluxPipeline


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
    parser.add_argument("--inverse_stretch", action="store_true",
                        help="Inverse p2/p98 stretch before saving (restore original value range)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def read_geotiff(path):
    """Read GeoTIFF, return (numpy HxWx3 float32 0-1, profile, stretch_params)."""
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        nodata = src.nodata
        data = src.read()  # (C, H, W)

    data = data.transpose(1, 2, 0).astype(np.float64)  # (H, W, C)

    # Mask nodata and all-zero pixels
    zero_mask = np.all(data == 0, axis=-1)
    if nodata is not None:
        nodata_mask = np.any(data == nodata, axis=-1)
        mask = nodata_mask | zero_mask
    else:
        mask = zero_mask if zero_mask.any() else None

    if mask is not None:
        data[mask] = 0

    # Percentile stretch to 0-1 (same approach as top_converter.py)
    valid = data[~mask] if mask is not None else data.reshape(-1, 3)
    p2 = np.percentile(valid, 2, axis=0)
    p98 = np.percentile(valid, 98, axis=0)
    print(f"[read_geotiff] percentile stretch: p2={p2}, p98={p98}")

    stretch_params = {"p2": p2, "p98": p98, "mask": mask}

    for c in range(data.shape[2]):
        rng = p98[c] - p2[c]
        if rng < 1:
            rng = 1
        data[:, :, c] = (data[:, :, c] - p2[c]) / rng

    data_f = np.clip(data, 0, 1).astype(np.float32)

    # Fill nodata regions with nearest valid pixel (so FLUX sees natural context, not black)
    if mask is not None and mask.any():
        from scipy.ndimage import distance_transform_edt
        _, nearest_idx = distance_transform_edt(mask, return_distances=True, return_indices=True)
        for ch in range(data_f.shape[2]):
            data_f[:, :, ch][mask] = data_f[:, :, ch][nearest_idx[0][mask], nearest_idx[1][mask]]
        print(f"[read_geotiff] filled {mask.sum()} nodata pixels with nearest valid pixels")

    return data_f, profile, stretch_params


def save_geotiff(path, data_float, profile, stretch_params, inverse_stretch=False):
    """Save float32 0-1 image back to GeoTIFF as uint8.

    If inverse_stretch=True, restore original value range using p2/p98
    before saving (preserves original satellite image distribution).
    """
    if inverse_stretch:
        p2 = stretch_params["p2"]
        p98 = stretch_params["p98"]
        data_out = np.zeros_like(data_float)
        for c in range(data_float.shape[2]):
            data_out[:, :, c] = data_float[:, :, c] * (p98[c] - p2[c]) + p2[c]
        data_out = np.clip(data_out, 0, 255).astype(np.uint8)
        print(f"[save_geotiff] inverse stretch: p2={p2}, p98={p98}")
    else:
        data_out = np.clip(data_float * 255, 0, 255).astype(np.uint8)

    # Restore nodata regions to 0
    mask = stretch_params.get("mask")
    if mask is not None and mask.any():
        data_out[mask] = 0

    # (H, W, C) -> (C, H, W)
    data_out = data_out.transpose(2, 0, 1)

    out_profile = profile.copy()
    out_profile["dtype"] = "uint8"
    out_profile["count"] = 3
    out_profile["nodata"] = None
    out_profile["compress"] = "lzw"

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data_out)


def compute_tiles(h, w, tile_size, overlap):
    """Compute tile positions (top, left) with overlap."""
    step = tile_size - overlap
    tiles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            # Adjust start if tile would be too small
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            tiles.append((y_start, x_start, y_end, x_end))
    # Deduplicate
    tiles = list(set(tiles))
    tiles.sort()
    return tiles


def make_blend_weight(tile_h, tile_w, overlap):
    """Create smooth blending weights that taper at edges."""
    weight = np.ones((tile_h, tile_w), dtype=np.float32)
    if overlap <= 0:
        return weight

    ramp = np.linspace(0, 1, overlap)
    # Top/bottom ramps
    for i in range(min(overlap, tile_h)):
        weight[i, :] *= ramp[i]
        weight[tile_h - 1 - i, :] *= ramp[i]
    # Left/right ramps
    for i in range(min(overlap, tile_w)):
        weight[:, i] *= ramp[i]
        weight[:, tile_w - 1 - i] *= ramp[i]

    return weight


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
    img, profile, stretch_params = read_geotiff(args.input)
    h, w, c = img.shape
    print(f"Image size: {w}x{h}, {c} bands")

    # Gamma correction (brighten dark satellite imagery before FLUX)
    gamma = args.gamma
    if gamma > 0 and gamma != 1.0:
        print(f"Applying gamma correction: {gamma}")
        mask = stretch_params["mask"]
        img = np.power(np.clip(img, 0, 1), gamma)
        if mask is not None:
            img[mask] = 0

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
    print("Loading FLUX model...")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload(device=args.device)
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

    # Inverse gamma correction (reverse the pre-processing)
    if gamma > 0 and gamma != 1.0:
        inv_gamma = 1.0 / gamma
        print(f"Applying inverse gamma: {inv_gamma:.4f}")
        output = np.power(np.clip(output, 0, 1), inv_gamma)
        nodata_mask = stretch_params["mask"]
        if nodata_mask is not None:
            output[nodata_mask] = 0

    # Save
    print(f"Saving to {args.output}...")
    save_geotiff(args.output, output, profile, stretch_params,
                 inverse_stretch=args.inverse_stretch)
    print("Done.")


if __name__ == "__main__":
    main()
