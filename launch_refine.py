"""
Multi-GPU launcher for GeoTIFF refinement (FlowEdit + FLUX.1-dev / Kontext).

- Auto-detects available GPUs when --gpus is not given
- Single GPU: identical behavior to refine_geotiff.py / refine_geotiff_kontext.py
- Multi GPU: distributes tiles across GPUs via multiprocessing
- Seed fixed across all tiles for consistent style

Usage:
    python launch_refine.py --method flowedit --input in.tif --output out.tif
    python launch_refine.py --method kontext --gpus 4,5,6,7 --input in.tif --output out.tif
"""

import sys
import os
import argparse
import tempfile


def auto_load_pipe(pipe, device):
    """Auto: direct GPU if VRAM >= 30GB, else cpu_offload."""
    gpu_idx = int(device.split(":")[-1]) if ":" in device else 0
    vram_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 3)
    if vram_gb >= 30:
        pipe = pipe.to(device)
        print(f"VRAM {vram_gb:.0f}GB → direct GPU load")
    else:
        pipe.enable_model_cpu_offload(device=device)
        print(f"VRAM {vram_gb:.0f}GB → cpu_offload")
    return pipe
import shutil
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from PIL import Image


def estimate_max_tile_size(gpu_id):
    """Estimate maximum tile_size based on GPU VRAM and model constraints."""
    FLUX_MAX_DIM = 1344

    props = torch.cuda.get_device_properties(gpu_id)
    vram_gb = props.total_memory / (1024 ** 3)

    MODEL_OVERHEAD_GB = 14.0
    BASE_TILE_VRAM_GB = 8.0
    SAFETY_FACTOR = 0.85

    available = (vram_gb * SAFETY_FACTOR) - MODEL_OVERHEAD_GB
    if available <= 0:
        return 512

    scale = available / BASE_TILE_VRAM_GB
    vram_tile = int(1024 * (scale ** 0.5))

    max_tile = min(vram_tile, FLUX_MAX_DIM)
    max_tile = (max_tile // 16) * 16
    max_tile = max(512, max_tile)

    print(f"[GPU {gpu_id}] {props.name}, VRAM: {vram_gb:.1f}GB → max tile_size: {max_tile}")
    return max_tile


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU GeoTIFF refinement")
    parser.add_argument("--method", type=str, default="flowedit",
                        choices=["flowedit", "kontext"],
                        help="Refinement method (default: flowedit)")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev",
                        help="HuggingFace model ID (default: black-forest-labs/FLUX.1-dev)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs. Auto-detect if omitted.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tile_size", type=int, default=0,
                        help="Tile size. 0 = auto-detect from VRAM (default).")
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility across tiles (default: 42)")
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--save_preview", action="store_true")

    # FlowEdit params
    parser.add_argument("--n_min", type=int, default=0)
    parser.add_argument("--n_max", type=int, default=15)
    parser.add_argument("--n_avg", type=int, default=1)
    parser.add_argument("--T_steps", type=int, default=28)
    parser.add_argument("--src_guidance", type=float, default=1.5)
    parser.add_argument("--tar_guidance", type=float, default=5.5)
    parser.add_argument("--src_prompt", type=str,
                        default="Low resolution satellite image with noise, blurring, and artifacts")
    parser.add_argument("--tar_prompt", type=str,
                        default="High resolution clean satellite image with sharp details and natural colors")

    # Kontext params
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--true_cfg", type=float, default=1.0)
    parser.add_argument("--negative_prompt", type=str,
                        default="Blurry, low resolution, noisy, artifacts, black regions, missing data")
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--prompt", type=str,
                        default="Enhance this satellite image: sharpen details, fill missing regions naturally with buildings roads and vegetation, improve resolution and colors")

    return parser.parse_args()


def get_gpu_ids(gpus_arg):
    if gpus_arg is not None:
        return [int(x.strip()) for x in gpus_arg.split(",")]
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("No CUDA GPUs available")
    return list(range(count))


# ── FlowEdit worker ──

def flowedit_worker_fn(gpu_id, tile_list, input_path, gamma, args, tmp_dir):
    sys.path.append("submodules/FlowEdit")
    from geotiff_utils import read_geotiff
    from refine_geotiff import refine_tile
    from diffusers import FluxPipeline

    device = f"cuda:{gpu_id}"

    img, _, stretch_params = read_geotiff(input_path)
    if gamma > 0 and gamma != 1.0:
        mask = stretch_params["mask"]
        img = np.power(np.clip(img, 0, 1), gamma)
        if mask is not None:
            img[mask] = 0

    model_id = args.model
    print(f"[GPU {gpu_id}] Loading FLUX model ({model_id})...")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = auto_load_pipe(pipe, device)
    scheduler = pipe.scheduler
    print(f"[GPU {gpu_id}] Model loaded. Processing {len(tile_list)} tiles.")

    worker_args = argparse.Namespace(**vars(args))
    worker_args.device = device

    for tile_idx, (y1, x1, y2, x2) in tile_list:
        tile = img[y1:y2, x1:x2].copy()
        refined = refine_tile(pipe, scheduler, tile, worker_args)
        np.save(os.path.join(tmp_dir, f"tile_{tile_idx}.npy"), refined)
        print(f"[GPU {gpu_id}] Tile {tile_idx} done ({y1},{x1})-({y2},{x2})")

    del pipe
    torch.cuda.empty_cache()


# ── Kontext worker ──

def kontext_worker_fn(gpu_id, tile_list, input_path, gamma, args, tmp_dir):
    from geotiff_utils import read_geotiff
    from diffusers import FluxKontextPipeline

    device = f"cuda:{gpu_id}"

    img, _, stretch_params = read_geotiff(input_path)
    if gamma > 0 and gamma != 1.0:
        mask = stretch_params["mask"]
        img = np.power(np.clip(img, 0, 1), gamma)
        if mask is not None:
            img[mask] = 0

    print(f"[GPU {gpu_id}] Loading Kontext model...")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
    )
    pipe = auto_load_pipe(pipe, device)
    print(f"[GPU {gpu_id}] Model loaded. Processing {len(tile_list)} tiles.")

    for tile_idx, (y1, x1, y2, x2) in tile_list:
        tile = img[y1:y2, x1:x2].copy()
        h, w = tile.shape[:2]

        pil_img = Image.fromarray((tile * 255).clip(0, 255).astype(np.uint8))

        # Fixed seed per tile for consistency
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

        pipe_kwargs = dict(
            image=pil_img,
            prompt=args.prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            height=h,
            width=w,
            generator=generator,
        )
        if args.true_cfg > 1.0:
            pipe_kwargs["true_cfg_scale"] = args.true_cfg
            pipe_kwargs["negative_prompt"] = args.negative_prompt

        with torch.inference_mode():
            result_img = pipe(**pipe_kwargs).images[0]

        refined = np.array(result_img).astype(np.float32) / 255.0
        np.save(os.path.join(tmp_dir, f"tile_{tile_idx}.npy"), refined)
        print(f"[GPU {gpu_id}] Tile {tile_idx} done ({y1},{x1})-({y2},{x2})")

    del pipe
    torch.cuda.empty_cache()


# ── Main ──

def main():
    t_start = time.time()
    args = parse_args()

    method = args.method
    print(f"Method: {method}")

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_refined{ext}"

    gpu_ids = get_gpu_ids(args.gpus)
    num_gpus = len(gpu_ids)

    # Auto tile_size
    if args.tile_size == 0:
        auto_tiles = [estimate_max_tile_size(gid) for gid in gpu_ids]
        args.tile_size = min(auto_tiles)
        print(f"Auto tile_size: {args.tile_size} (based on min VRAM across GPUs)")
    else:
        assert args.tile_size % 16 == 0, "tile_size must be divisible by 16"

    print(f"GPUs: {gpu_ids} ({num_gpus} total)")
    print(f"Tile size: {args.tile_size}")
    print(f"Seed: {args.seed}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    # ── Read image ──
    print("Reading GeoTIFF...")
    if method != "kontext":
        sys.path.append("submodules/FlowEdit")
    from geotiff_utils import read_geotiff, save_geotiff, compute_tiles, make_blend_weight

    img, profile, stretch_params = read_geotiff(args.input)
    h, w, c = img.shape
    print(f"Image size: {w}x{h}, {c} bands")

    # Gamma
    if args.gamma > 0 and args.gamma != 1.0:
        print(f"Applying gamma correction: {args.gamma}")
        mask = stretch_params["mask"]
        img = np.power(np.clip(img, 0, 1), args.gamma)
        if mask is not None:
            img[mask] = 0

    # Preview
    if args.save_preview:
        output_dir = os.path.dirname(args.output)
        input_stem = os.path.splitext(os.path.basename(args.input))[0]
        tag = "kontext" if method == "kontext" else "flux"
        preview_path = os.path.join(output_dir, f"{input_stem}_{tag}_input.png")
        preview = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(preview).save(preview_path)
        print(f"Saved preview: {preview_path}")

    # ── Compute tiles ──
    tiles = compute_tiles(h, w, args.tile_size, args.overlap)
    print(f"Total tiles: {len(tiles)} (tile_size={args.tile_size}, overlap={args.overlap})")

    # Select worker function
    worker_fn = kontext_worker_fn if method == "kontext" else flowedit_worker_fn

    # ── Single GPU path ──
    if num_gpus == 1:
        device = f"cuda:{gpu_ids[0]}"
        args.device = device

        if method == "kontext":
            from diffusers import FluxKontextPipeline
            print(f"Loading Kontext model on {device}...")
            pipe = FluxKontextPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
            )
            pipe = auto_load_pipe(pipe, device)
            print("Kontext model loaded.")

            from refine_geotiff_kontext import refine_tile as _refine_tile_kontext

            output = np.zeros((h, w, c), dtype=np.float32)
            weight_sum = np.zeros((h, w), dtype=np.float32)

            for idx, (y1, x1, y2, x2) in enumerate(tqdm(tiles, desc="Refining tiles")):
                tile = img[y1:y2, x1:x2].copy()

                # Patch args with seed generator
                args._generator = torch.Generator(device="cpu").manual_seed(args.seed)
                refined = _refine_tile_kontext(pipe, tile, args)
                weight = make_blend_weight(tile.shape[0], tile.shape[1], args.overlap)
                output[y1:y2, x1:x2] += refined * weight[:, :, None]
                weight_sum[y1:y2, x1:x2] += weight
        else:
            from diffusers import FluxPipeline
            from refine_geotiff import refine_tile

            model_id = args.model
            print(f"Loading FLUX model ({model_id}) on {device}...")
            pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = auto_load_pipe(pipe, device)
            scheduler = pipe.scheduler
            print("FLUX model loaded.")

            output = np.zeros((h, w, c), dtype=np.float32)
            weight_sum = np.zeros((h, w), dtype=np.float32)

            for idx, (y1, x1, y2, x2) in enumerate(tqdm(tiles, desc="Refining tiles")):
                tile = img[y1:y2, x1:x2].copy()
                refined = refine_tile(pipe, scheduler, tile, args)
                weight = make_blend_weight(tile.shape[0], tile.shape[1], args.overlap)
                output[y1:y2, x1:x2] += refined * weight[:, :, None]
                weight_sum[y1:y2, x1:x2] += weight

    # ── Multi GPU path ──
    else:
        tile_assignments = [[] for _ in range(num_gpus)]
        for idx, tile_coords in enumerate(tiles):
            rank = idx % num_gpus
            tile_assignments[rank].append((idx, tile_coords))

        for rank in range(num_gpus):
            print(f"  GPU {gpu_ids[rank]}: {len(tile_assignments[rank])} tiles")

        tmp_dir = tempfile.mkdtemp(prefix="refine_tiles_")
        print(f"Temp dir: {tmp_dir}")

        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=worker_fn,
                args=(
                    gpu_ids[rank],
                    tile_assignments[rank],
                    args.input,
                    args.gamma,
                    args,
                    tmp_dir,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        missing = [i for i in range(len(tiles))
                    if not os.path.exists(os.path.join(tmp_dir, f"tile_{i}.npy"))]
        if missing:
            raise RuntimeError(f"Missing tile results: {missing}")

        print("Merging tile results...")
        output = np.zeros((h, w, c), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)

        for idx, (y1, x1, y2, x2) in enumerate(tiles):
            refined = np.load(os.path.join(tmp_dir, f"tile_{idx}.npy"))
            tile_h, tile_w = refined.shape[:2]
            weight = make_blend_weight(tile_h, tile_w, args.overlap)
            output[y1:y2, x1:x2] += refined * weight[:, :, None]
            weight_sum[y1:y2, x1:x2] += weight

        shutil.rmtree(tmp_dir)

    # ── Normalize ──
    norm_mask = weight_sum > 0
    for c_idx in range(c):
        output[:, :, c_idx][norm_mask] /= weight_sum[norm_mask]

    # Inverse gamma
    if args.gamma > 0 and args.gamma != 1.0:
        inv_gamma = 1.0 / args.gamma
        print(f"Applying inverse gamma: {inv_gamma:.4f}")
        output = np.power(np.clip(output, 0, 1), inv_gamma)
        nodata_mask = stretch_params["mask"]
        if nodata_mask is not None:
            output[nodata_mask] = 0

    # ── Save ──
    print(f"Saving to {args.output}...")
    save_geotiff(args.output, output, profile, stretch_params)

    elapsed = time.time() - t_start
    print(f"Done. Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
