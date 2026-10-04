"""Quy luật "ngoài augmentation lúc train" + POOL (quy tắc khóa: progress_log 2026-10-04).
NOISE: chỉ nhiễu σ; FLIP: lật ngang xác suất p mỗi view; POOL: mỗi view (a) PIA bỏ tịnh tiến hoặc (b) phối cảnh thuần
căn tâm (yaw, pitch ~ U(±θ)), xác suất 0.5. RNG biến đổi: `random` của Python; nhiễu σ (torch) như pia_views."""
import math
import random

import torch

from attack.methods.camgeo import camera_grid, mean_disp
from attack.methods.pia import (DELTA_PAPER, PI_PAPER, drop_factor, homography, sample_source_corners, warp,
                                warp_grid)
from attack.methods.rrb import gaussian_blur


def flip_views(adv_pixels, n, p=0.5, sigma=6.0, companion=None, log=None):
    out = []
    for _ in range(n):
        f = random.random() < p
        if log is not None:
            log.append(float(f))
        view = torch.flip(adv_pixels, dims=[-1]) if f else adv_pixels
        comp = (torch.flip(companion, dims=[-1]) if f else companion) if companion is not None else None
        out.append((gaussian_blur(view, sigma=sigma) if sigma > 0 else view, comp, None))
    return out


def pool_grid(h, w, device, theta_deg=30.0, delta_max=DELTA_PAPER, pi=PI_PAPER):
    """-> (lưới, nhánh 'a' | 'b')."""
    if random.random() < 0.5:
        dst = torch.tensor([(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)], dtype=torch.float64)
        _, src = sample_source_corners(h, w, delta_max, pi)
        H, _ = drop_factor(homography(dst, src), "T", h, w)
        return warp_grid(H, h, w, device), "a"
    th = math.radians(theta_deg)
    return camera_grid(h, w, device, random.uniform(-th, th), random.uniform(-th, th)), "b"


def pool_views(adv_pixels, n, sigma=6.0, companion=None, log=None, theta_deg=30.0):
    h, w = adv_pixels.shape[-2:]
    out = []
    for _ in range(n):
        grid, br = pool_grid(h, w, adv_pixels.device, theta_deg)
        if log is not None:
            log.append(mean_disp(grid, h, w))
        view = warp(adv_pixels, grid)
        out.append((gaussian_blur(view, sigma=sigma) if sigma > 0 else view,
                    warp(companion, grid) if companion is not None else None, None))
    return out
