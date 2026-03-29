"""
Refine a GeoTIFF satellite image using FLUX Kontext (native image-to-image).

Requires: diffusers >= 0.35.0 (skyfall-kontext env)

Usage:
    python refine_geotiff_kontext.py --input /path/to/input.tif --output /path/to/output.tif
    python refine_geotiff_kontext.py --input /path/to/input.tif --output /path/to/output.tif \
        --tile_size 1024 --guidance_scale 2.5
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from geotiff_utils import read_geotiff, save_geotiff, compute_tiles, make_blend_weight


def parse_args():
    parser = argparse.ArgumentParser(description="Refine GeoTIFF with FLUX Kontext")
    parser.add_argument("--input", type=str, required=True, help="Input GeoTIFF path")
    parser.add_argument("--output", type=str, default=None, help="Output GeoTIFF path")
    parser.add_argument("--tile_size", type=int, default=1024, help="Tile size (must be divisible by 16)")
    parser.add_argument("--whole", action="store_true",
                        help="Process entire image at once (resize to fit max_area, then upscale back)")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between tiles for blending")
    parser.add_argument("--guidance_scale", type=float, default=2.5,
                        help="Guidance scale embedding (default: 2.5)")
    parser.add_argument("--true_cfg", type=float, default=1.0,
                        help="True CFG scale — actual editing strength. >1.0 enables real CFG (default: 1.0 = off)")
    parser.add_argument("--negative_prompt", type=str,
                        default="Blurry, low resolution, noisy, artifacts, black regions, missing data",
                        help="Negative prompt (used when true_cfg > 1.0)")
    parser.add_argument("--num_steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--prompt", type=str,
                        default="Enhance this satellite image: sharpen details, fill missing regions naturally with buildings roads and vegetation, improve resolution and colors",
                        help="Edit instruction prompt")
    parser.add_argument("--gamma", type=float, default=0.7,
                        help="Gamma correction before processing (0=disable, default: 0.7)")
    parser.add_argument("--save_preview", action="store_true",
                        help="Save pre-processing preview PNG")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def refine_tile(pipe, tile_img, args):
    """Run Kontext on a single tile. Input/output: numpy HxWx3 float32 0-1."""
    h, w = tile_img.shape[:2]

    # Convert to PIL
    pil_img = Image.fromarray((tile_img * 255).clip(0, 255).astype(np.uint8))

    # Run Kontext
    pipe_kwargs = dict(
        image=pil_img,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        height=h,
        width=w,
    )
    if args.true_cfg > 1.0:
        pipe_kwargs["true_cfg_scale"] = args.true_cfg
        pipe_kwargs["negative_prompt"] = args.negative_prompt

    with torch.inference_mode():
        result_img = pipe(**pipe_kwargs).images[0]

    # PIL -> numpy
    result = np.array(result_img).astype(np.float32) / 255.0
    return result


def main():
    args = parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_kontext{ext}"

    assert args.tile_size % 16 == 0, "tile_size must be divisible by 16"

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Prompt: {args.prompt}")
    print(f"Guidance scale: {args.guidance_scale}")

    # Read GeoTIFF
    print("Reading GeoTIFF...")
    img, profile, stretch_params = read_geotiff(args.input)
    h, w, c = img.shape
    print(f"Image size: {w}x{h}, {c} bands")

    # Gamma correction
    gamma = args.gamma
    if gamma > 0 and gamma != 1.0:
        print(f"Applying gamma correction: {gamma}")
        mask = stretch_params["mask"]
        img = np.power(np.clip(img, 0, 1), gamma)
        if mask is not None:
            img[mask] = 0

    # Save preview
    if args.save_preview:
        output_dir = os.path.dirname(args.output)
        input_stem = os.path.splitext(os.path.basename(args.input))[0]
        preview_path = os.path.join(output_dir, input_stem + "_kontext_input.png")
        preview = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(preview).save(preview_path)
        print(f"Saved preview: {preview_path}")

    # Load Kontext
    print("Loading FLUX Kontext model...")
    from diffusers import FluxKontextPipeline
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload(device=args.device)
    print("Kontext model loaded.")

    if args.whole:
        # ── Whole image mode: resize → Kontext → upscale back ──
        MAX_AREA = 1024 * 1024  # 1MP
        orig_area = h * w
        if orig_area > MAX_AREA:
            scale_factor = (MAX_AREA / orig_area) ** 0.5
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            # Round to nearest 16
            new_h = (new_h // 16) * 16
            new_w = (new_w // 16) * 16
        else:
            new_h, new_w = h, w

        print(f"Whole image mode: {w}x{h} → {new_w}x{new_h} ({new_w*new_h/1e6:.2f}MP)")

        # Resize down
        pil_input = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
        pil_small = pil_input.resize((new_w, new_h), Image.LANCZOS)

        # Run Kontext
        pipe_kwargs = dict(
            image=pil_small,
            prompt=args.prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            height=new_h,
            width=new_w,
        )
        if args.true_cfg > 1.0:
            pipe_kwargs["true_cfg_scale"] = args.true_cfg
            pipe_kwargs["negative_prompt"] = args.negative_prompt

        print("Running Kontext...")
        with torch.inference_mode():
            result_img = pipe(**pipe_kwargs).images[0]

        # Upscale back to original size
        result_img = result_img.resize((w, h), Image.LANCZOS)
        output = np.array(result_img).astype(np.float32) / 255.0
        print(f"Upscaled back to {w}x{h}")

    else:
        # ── Tile mode ──
        tiles = compute_tiles(h, w, args.tile_size, args.overlap)
        print(f"Processing {len(tiles)} tiles (tile_size={args.tile_size}, overlap={args.overlap})")

        output = np.zeros((h, w, c), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)

        for idx, (y1, x1, y2, x2) in enumerate(tqdm(tiles, desc="Refining tiles")):
            tile = img[y1:y2, x1:x2].copy()
            tile_h, tile_w = tile.shape[:2]

            refined = refine_tile(pipe, tile, args)
            weight = make_blend_weight(tile_h, tile_w, args.overlap)

            output[y1:y2, x1:x2] += refined * weight[:, :, None]
            weight_sum[y1:y2, x1:x2] += weight

        norm_mask = weight_sum > 0
        for c_idx in range(c):
            output[:, :, c_idx][norm_mask] /= weight_sum[norm_mask]

    # Inverse gamma
    if gamma > 0 and gamma != 1.0:
        inv_gamma = 1.0 / gamma
        print(f"Applying inverse gamma: {inv_gamma:.4f}")
        output = np.power(np.clip(output, 0, 1), inv_gamma)
        nodata_mask = stretch_params["mask"]
        if nodata_mask is not None:
            output[nodata_mask] = 0

    # Save
    print(f"Saving to {args.output}...")
    save_geotiff(args.output, output, profile, stretch_params)
    print("Done.")


if __name__ == "__main__":
    main()
