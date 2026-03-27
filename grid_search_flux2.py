"""
Grid search for FLUX.2 SDEdit parameters.
Loads model once, iterates over (strength, guidance_scale, detail_sigma) combinations.

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python grid_search_flux2.py \
        --input /path/to/input.tif --output_dir /path/to/output
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, distance_transform_edt
import rasterio

from refine_geotiff_flux2 import (
    read_geotiff, save_geotiff, pad_to_16, run_flux2_sdedit,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search FLUX.2 SDEdit")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str,
                        default="High resolution clean satellite image with sharp details, vivid colors, buildings, roads, and vegetation")
    parser.add_argument("--turbo", action="store_true")
    # Grid values
    parser.add_argument("--strengths", type=str, default="0.04,0.06,0.08,0.10,0.12")
    parser.add_argument("--guidances", type=str, default="2.0,3.0,4.0,5.0,6.0")
    parser.add_argument("--sigmas", type=str, default="2.0,4.0,6.0")
    parser.add_argument("--alphas", type=str, default="0.3,0.5,0.7")
    return parser.parse_args()


def main():
    args = parse_args()

    strengths = [float(x) for x in args.strengths.split(",")]
    guidances = [float(x) for x in args.guidances.split(",")]
    detail_sigmas = [float(x) for x in args.sigmas.split(",")]
    detail_alphas = [float(x) for x in args.alphas.split(",")]

    total = len(strengths) * len(guidances) * len(detail_sigmas) * len(detail_alphas)

    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, "output", f"grid_flux2_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Grid search: {len(strengths)} strengths × {len(guidances)} guidances × {len(detail_sigmas)} sigmas × {len(detail_alphas)} alphas = {total} combinations")
    print(f"  strengths: {strengths}")
    print(f"  guidances: {guidances}")
    print(f"  detail_sigmas: {detail_sigmas}")
    print(f"  detail_alphas: {detail_alphas}")
    print(f"  output_dir: {args.output_dir}")
    print()

    # Read image once
    print("Reading GeoTIFF...")
    img, profile, stretch_params = read_geotiff(args.input)
    h, w, c = img.shape
    print(f"Image size: {w}x{h}, {c} bands ({w*h/1e6:.2f}MP)")

    # Load model once
    print("Loading FLUX.2-dev model...")
    from diffusers import Flux2Pipeline
    pipe = Flux2Pipeline.from_pretrained(
        "black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16,
        device_map="balanced",
    )

    if args.turbo:
        print("Loading Turbo LoRA...")
        pipe.load_lora_weights(
            "fal/FLUX.2-dev-Turbo",
            weight_name="flux.2-turbo-lora.safetensors"
        )

    print("Model loaded.")
    print()

    # Prepare PIL input once
    pil_input = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))

    t_start = time.time()
    done = 0

    post_combos = len(detail_sigmas) * len(detail_alphas)

    for strength in strengths:
        for guidance in guidances:
            # Run SDEdit once per (strength, guidance) — sigma/alpha are post-processing only
            print(f"[{done+1}-{done+post_combos}/{total}] strength={strength}, guidance={guidance}")

            sdedit_args = argparse.Namespace(
                strength=strength,
                guidance_scale=guidance,
                num_steps=8 if args.turbo else 50,
                turbo=args.turbo,
                prompt=args.prompt,
                seed=args.seed,
                device=args.device,
            )

            try:
                result_pil = run_flux2_sdedit(pipe, pil_input, sdedit_args)
                sdedit_result = np.array(result_pil).astype(np.float32) / 255.0
            except Exception as e:
                print(f"  ERROR: {e}")
                done += post_combos
                continue

            # Apply detail restoration with different sigma/alpha combos
            for detail_sigma in detail_sigmas:
                # Pre-compute high_freq for this sigma
                high_freq = np.zeros_like(img)
                for ch in range(c):
                    blur_orig = gaussian_filter(img[:, :, ch], sigma=detail_sigma)
                    high_freq[:, :, ch] = img[:, :, ch] - blur_orig

                for detail_alpha in detail_alphas:
                    tag = f"s{strength:.2f}_g{guidance:.1f}_d{detail_sigma:.1f}_a{detail_alpha:.1f}"
                    out_path = os.path.join(args.output_dir, f"flux2_{tag}.tif")

                    output = np.clip(sdedit_result + detail_alpha * high_freq, 0, 1)

                    save_geotiff(out_path, output, profile, stretch_params)
                    done += 1

                    elapsed = time.time() - t_start
                    avg = elapsed / done
                    remaining = avg * (total - done)
                    print(f"  [{done}/{total}] {tag} → saved ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t_start
    print()
    print(f"Grid search complete: {total} combinations in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
