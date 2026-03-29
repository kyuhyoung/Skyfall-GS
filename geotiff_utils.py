"""
Common GeoTIFF utilities for all refinement pipelines.

- read_geotiff: nodata/zero fill → stretch if needed → /255 → auto gamma
  - no_stretch=True for multi-pass (pass 2+): just /255, skip everything else
- save_geotiff: 0-1 float × 255 → uint8
- compute_tiles / make_blend_weight: tiling helpers
"""

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt


def read_geotiff(path, no_stretch=False, gamma=0.7):
    """Read GeoTIFF.

    pass 1 (no_stretch=False):
      1. nodata/zero → nearest valid pixel fill
      2. max > 255 → percentile stretch to 0-255, else keep as-is
      3. /255 → 0-1 float
      4. median < 0.4 → gamma correction

    pass 2+ (no_stretch=True):
      1. /255 → 0-1 float (skip everything else)

    Returns: (data_f, profile, applied_gamma)
      - data_f: HxWx3 float32 0-1
      - profile: rasterio profile
      - applied_gamma: gamma value actually applied (1.0 if not applied)
    """
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        nodata = src.nodata
        data = src.read()  # (C, H, W)

    data = data.transpose(1, 2, 0).astype(np.float64)  # (H, W, C)

    if no_stretch:
        # Multi-pass: input is already processed uint8, just normalize
        data_f = np.clip(data / 255.0, 0, 1).astype(np.float32) if data.max() > 1 else data.astype(np.float32)
        print(f"[read_geotiff] no_stretch mode: simple normalization to 0-1")
        return data_f, profile, 1.0

    # 1. Mask nodata and all-zero pixels
    zero_mask = np.all(data == 0, axis=-1)
    if nodata is not None:
        nodata_mask = np.any(data == nodata, axis=-1)
        mask = nodata_mask | zero_mask
    else:
        mask = zero_mask if zero_mask.any() else None

    # 2. Fill nodata/zero regions with nearest valid pixel
    if mask is not None and mask.any():
        _, nearest_idx = distance_transform_edt(mask, return_distances=True, return_indices=True)
        for ch in range(data.shape[2]):
            data[:, :, ch][mask] = data[:, :, ch][nearest_idx[0][mask], nearest_idx[1][mask]]
        print(f"[read_geotiff] filled {mask.sum()} nodata pixels with nearest valid pixels")

    # 3. Range check and normalize to 0-255
    valid = data[~mask] if mask is not None else data.reshape(-1, data.shape[2])
    if valid.max() > 255:
        p2 = np.percentile(valid, 2, axis=0)
        p98 = np.percentile(valid, 98, axis=0)
        print(f"[read_geotiff] percentile stretch: p2={p2}, p98={p98}")
        for c in range(data.shape[2]):
            rng = p98[c] - p2[c]
            if rng < 1:
                rng = 1
            data[:, :, c] = (data[:, :, c] - p2[c]) / rng * 255
        data = np.clip(data, 0, 255)
    else:
        print(f"[read_geotiff] no stretch needed (max={valid.max():.0f}, within 0-255)")

    # 4. /255 -> 0-1 float
    data_f = np.clip(data / 255.0, 0, 1).astype(np.float32)

    # 5. Auto gamma: check median brightness
    median_brightness = np.median(data_f[~mask] if mask is not None else data_f)
    applied_gamma = 1.0
    if median_brightness < 0.4 and gamma > 0 and gamma != 1.0:
        print(f"[read_geotiff] dark image (median={median_brightness:.3f}), applying gamma={gamma}")
        data_f = np.power(data_f, gamma)
        applied_gamma = gamma
    else:
        print(f"[read_geotiff] brightness OK (median={median_brightness:.3f}), no gamma needed")

    return data_f, profile, applied_gamma


def save_geotiff(path, data_float, profile, stretch_params=None):
    """Save float32 0-1 image back to GeoTIFF as uint8 (0-255)."""
    data_out = np.clip(data_float * 255, 0, 255).astype(np.uint8)

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
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            tiles.append((y_start, x_start, y_end, x_end))
    tiles = list(set(tiles))
    tiles.sort()
    return tiles


def make_blend_weight(tile_h, tile_w, overlap):
    """Create smooth blending weights that taper at edges."""
    weight = np.ones((tile_h, tile_w), dtype=np.float32)
    if overlap <= 0:
        return weight

    ramp = np.linspace(0, 1, overlap)
    for i in range(min(overlap, tile_h)):
        weight[i, :] *= ramp[i]
        weight[tile_h - 1 - i, :] *= ramp[i]
    for i in range(min(overlap, tile_w)):
        weight[:, i] *= ramp[i]
        weight[:, tile_w - 1 - i] *= ramp[i]

    return weight
