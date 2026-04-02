"""
Grid search worker: loads FLUX model once, processes multiple combos on a single GPU.

Supports two modes:
  1) Legacy:  --job_file jobs.json   (processes all jobs in file sequentially)
  2) Queue:   --queue_file queue.json --queue_lock queue.lock
              (claims jobs one-by-one from a shared queue with file locking)

Usage:
    python grid_worker.py --job_file jobs.json --device cuda:0 --input in.tif --output_dir out/ ...
    python grid_worker.py --queue_file queue.json --queue_lock queue.lock --device cuda:0 --input in.tif --output_dir out/ ...
"""

import sys
sys.path.append("submodules/FlowEdit")

import os
import json
import fcntl
import argparse
import threading
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from FlowEdit_utils import FlowEditFLUX
from diffusers import FluxPipeline
from geotiff_utils import read_geotiff, save_geotiff, compute_tiles, make_blend_weight


class PeakRAMMonitor:
    """Background thread that samples available RAM at ~100ms intervals to find the peak usage."""

    def __init__(self, baseline_bytes=None):
        import psutil
        self._psutil = psutil
        self._baseline = baseline_bytes if baseline_bytes is not None else psutil.virtual_memory().available
        self._min_available = self._baseline  # track lowest available = highest usage
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._min_available = self._psutil.virtual_memory().available
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        while not self._stop.is_set():
            avail = self._psutil.virtual_memory().available
            if avail < self._min_available:
                self._min_available = avail
            self._stop.wait(0.1)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def peak_usage_gb(self):
        """Peak RAM consumed = baseline - min_available during monitoring."""
        return (self._baseline - self._min_available) / (1024 ** 3)

    def min_available_gb(self):
        """Lowest available RAM observed during monitoring (worst-case available)."""
        return self._min_available / (1024 ** 3)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_file", type=str, default=None,
                        help="JSON file with list of combos (legacy per-GPU mode)")
    parser.add_argument("--queue_file", type=str, default=None,
                        help="Shared job queue JSON (queue mode)")
    parser.add_argument("--queue_lock", type=str, default=None,
                        help="Lock file for shared queue")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tile_size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--T_steps", type=int, default=28)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--max_pass", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--src_prompt", type=str,
                        default="Satellite image with black missing regions, noise, blurring, and low resolution")
    parser.add_argument("--tar_prompt", type=str,
                        default="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev",
                        help="HuggingFace model ID (default: black-forest-labs/FLUX.1-dev)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    if not args.job_file and not args.queue_file:
        parser.error("Either --job_file or --queue_file is required")
    if args.queue_file and not args.queue_lock:
        parser.error("--queue_lock is required when using --queue_file")
    return args


# --------------- shared queue helpers ---------------

def claim_next_job(queue_file, lock_file, device):
    """Atomically claim next pending job. Returns job dict or None."""
    with open(lock_file, 'r+') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with open(queue_file) as f:
                jobs = json.load(f)
            for job in jobs:
                if job["status"] == "pending":
                    job["status"] = "in_progress"
                    job["device"] = device
                    with open(queue_file, 'w') as f:
                        json.dump(jobs, f, indent=2)
                    return job
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return None


def update_job_status(queue_file, lock_file, job_id, status):
    """Update a job's status in the shared queue."""
    with open(lock_file, 'r+') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with open(queue_file) as f:
                jobs = json.load(f)
            for job in jobs:
                if job["job_id"] == job_id:
                    job["status"] = status
                    break
            with open(queue_file, 'w') as f:
                json.dump(jobs, f, indent=2)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# --------------- tile / image processing ---------------

def refine_tile(pipe, scheduler, tile_img, device, src_prompt, tar_prompt,
                T_steps, n_avg, src_guidance, tar_guidance, n_min, n_max, seed, tile_idx):
    """Run FlowEdit on a single tile. src_guidance/tar_guidance can be None for schnell."""
    h, w = tile_img.shape[:2]
    h_pad = (16 - h % 16) % 16
    w_pad = (16 - w % 16) % 16

    if h_pad > 0 or w_pad > 0:
        tile_img = np.pad(tile_img, ((0, h_pad), (0, w_pad), (0, 0)), mode="reflect")

    pil_img = Image.fromarray((tile_img * 255).clip(0, 255).astype(np.uint8))
    image_src = pipe.image_processor.preprocess(pil_img)
    image_src = image_src.to(device).half()

    with torch.autocast("cuda"), torch.inference_mode():
        x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
    x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    x0_src = x0_src.to(device)

    torch.manual_seed(seed + tile_idx)
    torch.cuda.manual_seed_all(seed + tile_idx)

    fe_kwargs = dict(
        T_steps=T_steps, n_avg=n_avg,
        n_min=n_min, n_max=n_max
    )
    if src_guidance is not None:
        fe_kwargs["src_guidance_scale"] = src_guidance
    if tar_guidance is not None:
        fe_kwargs["tar_guidance_scale"] = tar_guidance

    x0_tar = FlowEditFLUX(
        pipe, scheduler, x0_src,
        src_prompt, tar_prompt, "",
        **fe_kwargs
    )

    x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.autocast("cuda"), torch.inference_mode():
        image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
    image_tar = pipe.image_processor.postprocess(image_tar)[0]

    result = np.array(image_tar).astype(np.float32) / 255.0
    if h_pad > 0 or w_pad > 0:
        result = result[:h, :w]
    return result


def process_image(pipe, scheduler, img, tiles, args, n_min, n_max, src_guidance, tar_guidance,
                   T_steps=None, ram_monitor=None):
    """Process all tiles for one image."""
    if T_steps is None:
        T_steps = args.T_steps
    h, w, c = img.shape
    output = np.zeros((h, w, c), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    for idx, (y1, x1, y2, x2) in enumerate(tqdm(tiles, desc="  tiles", leave=False)):
        tile = img[y1:y2, x1:x2].copy()
        tile_h, tile_w = tile.shape[:2]

        # Start peak RAM monitoring on first tile
        if idx == 0 and ram_monitor is not None:
            ram_monitor.start()

        refined = refine_tile(
            pipe, scheduler, tile, args.device,
            args.src_prompt, args.tar_prompt,
            T_steps, 1, src_guidance, tar_guidance,
            n_min, n_max, args.seed, idx
        )

        # Stop monitoring after first tile and update cost file
        if idx == 0 and ram_monitor is not None:
            ram_monitor.stop()
            _update_peak_ram(args, ram_monitor.peak_usage_gb(), ram_monitor.min_available_gb())
            ram_monitor = None  # don't monitor again

        weight = make_blend_weight(tile_h, tile_w, args.overlap)
        output[y1:y2, x1:x2] += refined * weight[:, :, None]
        weight_sum[y1:y2, x1:x2] += weight

    mask = weight_sum > 0
    for c_idx in range(c):
        output[:, :, c_idx][mask] /= weight_sum[mask]
    return output


def _update_peak_ram(args, peak_gb, min_available_gb):
    """Update ram_per_model.txt and worst_available.txt after first tile peak measurement."""
    ram_cost_file = os.path.join(args.output_dir, "ram_per_model.txt")
    worst_avail_file = os.path.join(args.output_dir, "worst_available.txt")
    lock_path = ram_cost_file + ".lock"
    with open(lock_path, 'a+') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            prev_max = 0.0
            if os.path.exists(ram_cost_file):
                prev_max = float(open(ram_cost_file).read().strip())
            print(f"[{args.device}] Peak RAM during first tile: {peak_gb:.1f}GB (prev max: {prev_max:.1f}GB)")
            print(f"[{args.device}] Worst-case available during first tile: {min_available_gb:.1f}GB")
            if peak_gb > prev_max:
                with open(ram_cost_file, 'w') as f:
                    f.write(f"{peak_gb:.1f}")
                print(f"[{args.device}] Updated ram_per_model.txt: {prev_max:.1f} → {peak_gb:.1f}GB")
            # Always write worst available (first worker's measurement)
            with open(worst_avail_file, 'w') as f:
                f.write(f"{min_available_gb:.1f}")
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def process_job(pipe, scheduler, src_img, tiles, profile, args, job, job_label,
                ram_monitor=None):
    """Process one job (all passes for one parameter combo)."""
    n_min = job["n_min"]
    n_max = job["n_max"]
    tg = job.get("tar_guidance", None)
    sg = job.get("src_guidance", None)
    job_T_steps = job.get("T_steps", args.T_steps)
    if tg is not None and sg is not None:
        tag_base = f"nmin{n_min:02d}_nmax{n_max:02d}_tg{tg:.1f}_sg{sg:.2f}"
    else:
        tag_base = f"nmin{n_min:02d}_nmax{n_max:02d}"
    tag_base = f"ts{job_T_steps:02d}_{tag_base}"

    start_pass = job.get("start_pass", 1)
    end_pass = job.get("end_pass", args.max_pass)

    print(f"[{args.device}] {job_label} {tag_base} pass {start_pass}-{end_pass}")

    current_img = src_img

    for p in range(start_pass, end_pass + 1):
        tag = f"{tag_base}_pass{p}"
        outfile = os.path.join(args.output_dir, f"{tag}.tif")

        if p > 1:
            current_img, profile_p, _ = read_geotiff(
                os.path.join(args.output_dir, f"{tag_base}_pass{p-1}.tif"),
                no_stretch=True, gamma=args.gamma
            )
            tiles_p = compute_tiles(current_img.shape[0], current_img.shape[1],
                                    args.tile_size, args.overlap)
        else:
            tiles_p = tiles

        # Pass ram_monitor only on first pass of first job, then clear it
        output = process_image(pipe, scheduler, current_img, tiles_p, args,
                               n_min, n_max, sg, tg, T_steps=job_T_steps,
                               ram_monitor=ram_monitor)
        ram_monitor = None  # only measure once per worker
        save_geotiff(outfile, output, profile)
        print(f"[{args.device}]   pass {p} saved: {outfile}")

    print(f"[{args.device}] {job_label} {tag_base} — DONE")


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # RAM check before loading
    import psutil
    ram_cost_file = os.path.join(args.output_dir, "ram_per_model.txt")
    ram_before_bytes = psutil.virtual_memory().available
    ram_before = ram_before_bytes / (1024 ** 3)

    # Determine minimum RAM needed: use measured value if available, else fallback 30GB
    if os.path.exists(ram_cost_file):
        measured_gb = float(open(ram_cost_file).read().strip())
        min_ram_gb = max(measured_gb * 1.2, measured_gb + 40)  # 20% headroom + 40GB system reserve
    else:
        min_ram_gb = 50.0  # conservative fallback for first worker

    if ram_before < min_ram_gb:
        print(f"[{args.device}] ERROR: Not enough RAM. "
              f"Available: {ram_before:.1f}GB, required: {min_ram_gb:.1f}GB. Aborting.")
        sys.exit(1)
    print(f"[{args.device}] RAM check OK: {ram_before:.1f}GB available (need {min_ram_gb:.1f}GB)")

    # Load model ONCE
    model_id = args.model
    print(f"[{args.device}] Loading FLUX model ({model_id})...")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # Auto: direct GPU if VRAM >= 30GB, else cpu_offload
    gpu_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
    vram_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 3)
    if vram_gb >= 30:
        pipe = pipe.to(args.device)
        print(f"[{args.device}] VRAM {vram_gb:.0f}GB → direct GPU load")
    else:
        pipe.enable_model_cpu_offload(device=args.device)
        print(f"[{args.device}] VRAM {vram_gb:.0f}GB → cpu_offload")
    scheduler = pipe.scheduler

    # Measure actual RAM consumed and update max
    ram_after = psutil.virtual_memory().available / (1024 ** 3)
    ram_used = ram_before - ram_after
    print(f"[{args.device}] FLUX model loaded. (RAM used: {ram_used:.1f}GB)")

    # Update ram_per_model.txt with max observed value (file-locked)
    lock_path = ram_cost_file + ".lock"
    with open(lock_path, 'a+') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            prev_max = 0.0
            if os.path.exists(ram_cost_file):
                prev_max = float(open(ram_cost_file).read().strip())
            new_max = max(prev_max, ram_used)
            if new_max > prev_max:
                with open(ram_cost_file, 'w') as f:
                    f.write(f"{new_max:.1f}")
                print(f"[{args.device}] Updated RAM cost: {prev_max:.1f} → {new_max:.1f}GB")
            elif prev_max == 0.0:
                with open(ram_cost_file, 'w') as f:
                    f.write(f"{new_max:.1f}")
                print(f"[{args.device}] Saved RAM cost: {new_max:.1f}GB")
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

    # Read source image and compute tiles ONCE
    print(f"[{args.device}] Reading input: {args.input}")
    src_img, profile, _ = read_geotiff(args.input, no_stretch=False, gamma=args.gamma)
    h, w, c = src_img.shape
    tiles = compute_tiles(h, w, args.tile_size, args.overlap)
    print(f"[{args.device}] Image {w}x{h}, {len(tiles)} tiles per pass")

    if args.queue_file:
        # ---- Shared queue mode ----
        job_count = 0
        while True:
            job = claim_next_job(args.queue_file, args.queue_lock, args.device)
            if job is None:
                break
            job_count += 1
            job_id = job["job_id"]
            try:
                process_job(pipe, scheduler, src_img, tiles, profile, args, job,
                            f"[job {job_id}]", ram_monitor=PeakRAMMonitor(baseline_bytes=ram_before_bytes))
                update_job_status(args.queue_file, args.queue_lock, job_id, "done")
            except Exception as e:
                print(f"[{args.device}] ERROR job {job_id}: {e}")
                import traceback; traceback.print_exc()
                update_job_status(args.queue_file, args.queue_lock, job_id, "failed")
        print(f"[{args.device}] Processed {job_count} jobs. Queue empty.")
    else:
        # ---- Legacy per-GPU job file mode ----
        with open(args.job_file) as f:
            jobs = json.load(f)
        total_jobs = len(jobs)
        print(f"[{args.device}] {total_jobs} combos to process (× {args.max_pass} passes each)")
        for job_idx, job in enumerate(jobs):
            process_job(pipe, scheduler, src_img, tiles, profile, args, job,
                        f"[{job_idx+1}/{total_jobs}]", ram_monitor=PeakRAMMonitor(baseline_bytes=ram_before_bytes))
        print(f"[{args.device}] All {total_jobs} combos complete.")


if __name__ == "__main__":
    main()
