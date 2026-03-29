"""
Refine a GeoTIFF satellite image using FLUX.2-dev + Turbo LoRA.

- Native image-to-image editing (no FlowEdit hack needed)
- Supports up to 4MP → can process most satellite tiles without tiling
- 8-step Turbo inference for speed

Requires: diffusers >= 0.37.0 (skyfall-kontext env or newer)

Usage:
    python refine_geotiff_flux2.py --input /path/to/input.tif --output /path/to/output.tif
    python refine_geotiff_flux2.py --input in.tif --output out.tif --guidance_scale 3.5 --turbo
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from geotiff_utils import read_geotiff, save_geotiff, compute_tiles, make_blend_weight


# Turbo LoRA 8-step sigma schedule
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]


def parse_args():
    parser = argparse.ArgumentParser(description="Refine GeoTIFF with FLUX.2-dev (SDEdit)")
    parser.add_argument("--input", type=str, required=True, help="Input GeoTIFF path")
    parser.add_argument("--output", type=str, default=None, help="Output GeoTIFF path")
    parser.add_argument("--tile_size", type=int, default=0,
                        help="Tile size. 0 = whole image (default). Set >0 for tiling.")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between tiles for blending")
    parser.add_argument("--strength", type=float, default=0.4,
                        help="Edit strength: 0.0=no change, 1.0=full regeneration (default: 0.4)")
    parser.add_argument("--guidance_scale", type=float, default=4.0,
                        help="Guidance scale (default: 4.0)")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Total inference steps before strength truncation (default: 50, turbo: 8)")
    parser.add_argument("--turbo", action="store_true",
                        help="Use FLUX.2-dev-Turbo LoRA (8 steps)")
    parser.add_argument("--prompt", type=str,
                        default="High resolution clean satellite image with sharp details, vivid colors, buildings, roads, and vegetation",
                        help="Target prompt describing desired output")
    parser.add_argument("--gamma", type=float, default=0.0,
                        help="Gamma correction (0=disable, default: 0)")
    parser.add_argument("--save_preview", action="store_true",
                        help="Save pre-processing preview PNG")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--restore_detail", action="store_true",
                        help="Restore high-freq detail from original after processing (combats VAE blur)")
    parser.add_argument("--detail_sigma", type=float, default=2.0,
                        help="Gaussian blur sigma for detail extraction (default: 2.0)")
    parser.add_argument("--detail_alpha", type=float, default=1.0,
                        help="Detail restoration intensity: 0.0=none, 0.5=half, 1.0=full (default: 1.0)")
    # FlowEdit mode
    parser.add_argument("--flowedit", action="store_true",
                        help="Use FlowEdit instead of SDEdit (better source fidelity control)")
    parser.add_argument("--n_min", type=int, default=0, help="FlowEdit: pure generation steps at end")
    parser.add_argument("--n_max", type=int, default=15, help="FlowEdit: editing steps from end")
    parser.add_argument("--n_avg", type=int, default=1, help="FlowEdit: averaging iterations")
    parser.add_argument("--src_guidance", type=float, default=1.5, help="FlowEdit: source guidance")
    parser.add_argument("--tar_guidance", type=float, default=5.5, help="FlowEdit: target guidance")
    parser.add_argument("--src_prompt", type=str,
                        default="Low resolution satellite image with noise, blurring, and artifacts",
                        help="FlowEdit: source prompt")
    return parser.parse_args()


def pad_to_16(pil_img):
    """Pad PIL image to dimensions divisible by 16. Returns (padded_pil, orig_h, orig_w)."""
    w, h = pil_img.size
    new_h = ((h + 15) // 16) * 16
    new_w = ((w + 15) // 16) * 16
    if new_h == h and new_w == w:
        return pil_img, h, w
    # Reflect-pad via numpy
    arr = np.array(pil_img)
    padded = np.pad(arr, ((0, new_h - h), (0, new_w - w), (0, 0)), mode="reflect")
    return Image.fromarray(padded), h, w


def run_flux2_sdedit(pipe, pil_img, args):
    """SDEdit: encode image → add noise (strength) → denoise with prompt.
    Input/output: PIL Image (original dimensions preserved).
    """
    padded_img, orig_h, orig_w = pad_to_16(pil_img)
    pad_w, pad_h = padded_img.size  # PIL is (w, h)

    device = args.device
    dtype = torch.bfloat16

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    # 1. Encode image to latents
    image_tensor = torch.from_numpy(
        np.array(padded_img).astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    image_tensor = 2.0 * image_tensor - 1.0  # normalize to [-1, 1]

    with torch.inference_mode():
        # Move to target GPU
        image_tensor = image_tensor.to(device=device, dtype=dtype)
        image_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator=generator)
        image_latents = pipe._patchify_latents(image_latents)

        # Apply batch norm normalization (FLUX.2 specific)
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps)
        bn_std = bn_std.to(image_latents.device, image_latents.dtype)
        image_latents = (image_latents - bn_mean) / bn_std

    # 2. Set up scheduler and timesteps
    from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu
    packed_latents = pipe._pack_latents(image_latents)
    image_seq_len = packed_latents.shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=args.num_steps)

    sigmas = None
    if args.turbo:
        sigmas = TURBO_SIGMAS

    from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, args.num_steps, pipe.transformer.device,
        sigmas=sigmas, mu=mu,
    )

    # Compute start step based on strength
    start_step = max(int(len(timesteps) * (1.0 - args.strength)), 0)
    timesteps = timesteps[start_step:]
    start_sigma = pipe.scheduler.sigmas[start_step]

    print(f"  SDEdit: strength={args.strength}, start_step={start_step}/{args.num_steps}, "
          f"denoise_steps={len(timesteps)}, start_sigma={start_sigma:.4f}")

    # 3. Add noise at start_sigma (all on same device)
    compute_device = torch.device(device)
    image_latents = image_latents.to(compute_device)

    generator_gpu = torch.Generator(device=compute_device).manual_seed(args.seed)
    noise = torch.randn(
        image_latents.shape, generator=generator_gpu,
        device=compute_device, dtype=image_latents.dtype
    )
    # Flow matching: noisy = sigma * noise + (1 - sigma) * clean
    latents = start_sigma * noise + (1.0 - start_sigma) * image_latents

    # Pack latents and get latent IDs
    latent_ids = pipe._prepare_latent_ids(image_latents)
    latent_ids = latent_ids.to(compute_device)
    latents = pipe._pack_latents(latents)

    # 4. Encode prompt
    prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=args.prompt,
        device=pipe.transformer.device,
        num_images_per_prompt=1,
    )

    # Guidance embedding
    guidance = torch.full([1], args.guidance_scale,
                          device=pipe.transformer.device, dtype=torch.float32)

    # 5. Denoise loop
    with torch.inference_mode():
        for i, t in enumerate(tqdm(timesteps, desc="  Denoising", leave=False)):
            latent_model_input = latents.to(pipe.transformer.device, dtype=pipe.transformer.dtype)

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=(t / 1000).unsqueeze(0).to(pipe.transformer.device),
                guidance=guidance,
                encoder_hidden_states=prompt_embeds.to(pipe.transformer.device, dtype=pipe.transformer.dtype),
                txt_ids=text_ids.to(pipe.transformer.device, dtype=pipe.transformer.dtype),
                img_ids=latent_ids.to(pipe.transformer.device, dtype=pipe.transformer.dtype),
                return_dict=False,
            )[0]

            # Only keep the noise-latent portion (no image reference tokens)
            noise_pred = noise_pred[:, :latents.size(1)]

            latents = pipe.scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]

    # 6. Decode latents
    # Unpack and denormalize — ensure all on same device
    latents = latents.to(compute_device, dtype=dtype)
    latent_ids = latent_ids.to(compute_device)
    latents = pipe._unpack_latents_with_ids(latents, latent_ids)
    bn_std = bn_std.to(compute_device, dtype=dtype)
    bn_mean = bn_mean.to(compute_device, dtype=dtype)
    latents = latents * bn_std + bn_mean
    latents = pipe._unpatchify_latents(latents)

    with torch.inference_mode():
        latents = latents.to(pipe.vae.device, dtype=pipe.vae.dtype)
        image_out = pipe.vae.decode(latents, return_dict=False)[0]

    # Post-process: [-1,1] → [0,255] uint8
    image_out = (image_out / 2 + 0.5).clamp(0, 1)
    image_out = image_out[0].permute(1, 2, 0).cpu().float().numpy()
    image_out = (image_out * 255).clip(0, 255).astype(np.uint8)

    # Crop to original
    image_out = image_out[:orig_h, :orig_w]
    return Image.fromarray(image_out)


def main():
    args = parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        tag = "flux2turbo" if args.turbo else "flux2"
        args.output = f"{base}_{tag}{ext}"

    if args.turbo:
        args.num_steps = 8
        print("Turbo mode: 8 steps")

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Prompt: {args.prompt}")
    print(f"Guidance: {args.guidance_scale}, Steps: {args.num_steps}")
    print(f"Turbo: {args.turbo}, Seed: {args.seed}")

    # Read GeoTIFF
    print("Reading GeoTIFF...")
    img, profile, stretch_params = read_geotiff(args.input)
    h, w, c = img.shape
    print(f"Image size: {w}x{h}, {c} bands ({w*h/1e6:.2f}MP)")

    # Gamma
    gamma = args.gamma
    if gamma > 0 and gamma != 1.0:
        print(f"Applying gamma correction: {gamma}")
        mask = stretch_params["mask"]
        img = np.power(np.clip(img, 0, 1), gamma)
        if mask is not None:
            img[mask] = 0

    # Preview
    if args.save_preview:
        output_dir = os.path.dirname(args.output)
        input_stem = os.path.splitext(os.path.basename(args.input))[0]
        preview_path = os.path.join(output_dir, input_stem + "_flux2_input.png")
        preview = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(preview).save(preview_path)
        print(f"Saved preview: {preview_path}")

    # Load FLUX.2
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

    # Decide: whole image or tiled
    if args.tile_size == 0:
        # ── Whole image mode ──
        area = h * w
        max_area = 4 * 1024 * 1024  # 4MP
        if area > max_area:
            print(f"WARNING: Image {w}x{h} ({area/1e6:.1f}MP) exceeds 4MP. Using tiling with tile_size=1024.")
            args.tile_size = 1024
        else:
            print(f"Whole image mode: {w}x{h} ({area/1e6:.2f}MP)")

            if args.flowedit:
                # ── FlowEdit mode ──
                from flowedit_flux2 import FlowEditFLUX2
                print(f"FlowEdit: n_min={args.n_min}, n_max={args.n_max}, "
                      f"src_guidance={args.src_guidance}, tar_guidance={args.tar_guidance}")

                # Pad to 16
                pil_padded, orig_h, orig_w = pad_to_16(
                    Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
                )
                pad_h, pad_w = pil_padded.size[1], pil_padded.size[0]

                # VAE encode
                compute_device = torch.device(args.device)
                image_tensor = torch.from_numpy(
                    np.array(pil_padded).astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0)
                image_tensor = 2.0 * image_tensor - 1.0
                image_tensor = image_tensor.to(device=compute_device, dtype=torch.bfloat16)

                print("VAE encoding...")
                image_latents = pipe.vae.encode(image_tensor).latent_dist.sample()
                image_latents = pipe._patchify_latents(image_latents)
                bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
                bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps)
                bn_std = bn_std.to(image_latents.device, image_latents.dtype)
                image_latents = (image_latents - bn_mean) / bn_std

                # Run FlowEdit
                edited_packed, latent_ids = FlowEditFLUX2(
                    pipe, image_latents,
                    src_prompt=args.src_prompt,
                    tar_prompt=args.prompt,
                    T_steps=args.num_steps,
                    n_avg=args.n_avg,
                    src_guidance_scale=args.src_guidance,
                    tar_guidance_scale=args.tar_guidance,
                    n_min=args.n_min,
                    n_max=args.n_max,
                )

                # VAE decode
                print("VAE decoding...")
                edited_packed = edited_packed.to(compute_device, dtype=torch.bfloat16)
                latent_ids = latent_ids.to(compute_device)
                latents = pipe._unpack_latents_with_ids(edited_packed, latent_ids)
                bn_std = bn_std.to(compute_device, dtype=torch.bfloat16)
                bn_mean = bn_mean.to(compute_device, dtype=torch.bfloat16)
                latents = latents * bn_std + bn_mean
                latents = pipe._unpatchify_latents(latents)
                latents = latents.to(pipe.vae.device, dtype=pipe.vae.dtype)
                image_out = pipe.vae.decode(latents, return_dict=False)[0]

                image_out = (image_out / 2 + 0.5).clamp(0, 1)
                image_out = image_out[0].permute(1, 2, 0).detach().cpu().float().numpy()
                output = image_out[:orig_h, :orig_w]

            else:
                # ── SDEdit mode ──
                pil_input = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))

                result_pil = run_flux2_sdedit(pipe, pil_input, args)

                if result_pil.size != (w, h):
                    print(f"Resizing output {result_pil.size} → ({w}, {h})")
                    result_pil = result_pil.resize((w, h), Image.LANCZOS)

                output = np.array(result_pil).astype(np.float32) / 255.0

    if args.tile_size > 0:
        # ── Tile mode ──
        assert args.tile_size % 16 == 0, "tile_size must be divisible by 16"
        tiles = compute_tiles(h, w, args.tile_size, args.overlap)
        print(f"Processing {len(tiles)} tiles (tile_size={args.tile_size}, overlap={args.overlap})")

        output = np.zeros((h, w, c), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)

        for idx, (y1, x1, y2, x2) in enumerate(tqdm(tiles, desc="Refining tiles")):
            tile = img[y1:y2, x1:x2].copy()
            tile_h, tile_w = tile.shape[:2]

            pil_tile = Image.fromarray((tile * 255).clip(0, 255).astype(np.uint8))
            result_pil = run_flux2_sdedit(pipe, pil_tile, args)
            refined = np.array(result_pil).astype(np.float32) / 255.0

            # Handle size mismatch
            if refined.shape[:2] != (tile_h, tile_w):
                result_pil = result_pil.resize((tile_w, tile_h), Image.LANCZOS)
                refined = np.array(result_pil).astype(np.float32) / 255.0

            weight = make_blend_weight(tile_h, tile_w, args.overlap)
            output[y1:y2, x1:x2] += refined * weight[:, :, None]
            weight_sum[y1:y2, x1:x2] += weight

        norm_mask = weight_sum > 0
        for c_idx in range(c):
            output[:, :, c_idx][norm_mask] /= weight_sum[norm_mask]

    # Restore high-frequency detail from original
    if args.restore_detail:
        from scipy.ndimage import gaussian_filter
        sigma = args.detail_sigma
        alpha = args.detail_alpha
        print(f"Restoring detail from original (sigma={sigma}, alpha={alpha})...")
        for ch in range(c):
            orig_ch = img[:, :, ch]
            blur_orig = gaussian_filter(orig_ch, sigma=sigma)
            high_freq = orig_ch - blur_orig
            output[:, :, ch] = np.clip(output[:, :, ch] + alpha * high_freq, 0, 1)

    # Inverse gamma
    if gamma > 0 and gamma != 1.0:
        inv_gamma = 1.0 / gamma
        print(f"Applying inverse gamma: {inv_gamma:.4f}")
        output = np.power(np.clip(output, 0, 1), inv_gamma)

    # Save
    print(f"Saving to {args.output}...")
    save_geotiff(args.output, output, profile, stretch_params)
    print("Done.")


if __name__ == "__main__":
    main()
