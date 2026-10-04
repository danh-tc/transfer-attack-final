"""Giai đoạn B (quy tắc khóa: progress_log 2026-10-04 "đa dạng mô hình camera"): xoay camera ảo thuần phối cảnh
(H = K·R_y(yaw)·R_x(pitch)·K⁻¹, f = max(h, w), không tịnh tiến, không roll) và méo ống kính xuyên tâm
(nguồn = c + d·(1 + k·ρ²), ρ = ‖d‖ / nửa đường chéo) — biến đổi ngoài họ homography. Ánh xạ ngược pixel ra -> nguồn,
ngoài ảnh nguồn lấp 0. RNG tham số: `random` của Python mỗi view; nhiễu σ (torch) cộng lên view adv như pia_views.
"""
import math
import random

import torch

from attack.methods.box_transforms import get_gt
from attack.methods.morgd import warp_boxes_by_grid
from attack.methods.pia import gt_with_keep, warp
from attack.methods.rrb import gaussian_blur


def rotation_homography(yaw: float, pitch: float, h: int, w: int) -> torch.Tensor:
    """K·R_y·R_x·K⁻¹, CĂN LẠI TÂM (sửa sau unit test, trước khi chạy): ghép tịnh tiến đưa ảnh của tâm về tâm —
    bậc nhất xoay camera = dịch f·tan θ (thành phần Giai đoạn A cho là vô ích); chỉ giữ méo phối cảnh."""
    f, cx, cy = float(max(h, w)), (w - 1) / 2, (h - 1) / 2
    K = torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=torch.float64)
    cyw, syw, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Ry = torch.tensor([[cyw, 0, syw], [0, 1, 0], [-syw, 0, cyw]], dtype=torch.float64)
    Rx = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=torch.float64)
    H = K @ Ry @ Rx @ torch.linalg.inv(K)
    p = H @ torch.tensor([cx, cy, 1.0], dtype=torch.float64)
    T = torch.tensor([[1, 0, cx - p[0] / p[2]], [0, 1, cy - p[1] / p[2]], [0, 0, 1]], dtype=torch.float64)
    return T @ H


def camera_grid(h: int, w: int, device, yaw: float = 0.0, pitch: float = 0.0, k: float = 0.0) -> torch.Tensor:
    """Lưới grid_sample [1,h,w,2] (align_corners=True): q -> lens(q) -> H·lens(q)."""
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    cx, cy = (w - 1) / 2, (h - 1) / 2
    if k != 0.0:
        R = math.hypot(cx, cy)
        dx, dy = xs - cx, ys - cy
        g = 1 + k * (dx ** 2 + dy ** 2) / R ** 2
        xs, ys = cx + dx * g, cy + dy * g
    if yaw != 0.0 or pitch != 0.0:
        H = rotation_homography(yaw, pitch, h, w)
        p = torch.stack([xs, ys, torch.ones_like(xs)], -1) @ H.T
        xs, ys = p[..., 0] / p[..., 2], p[..., 1] / p[..., 2]
    grid = torch.stack([xs / (w - 1) * 2 - 1, ys / (h - 1) * 2 - 1], dim=-1)
    return grid.float().unsqueeze(0).to(device)


def mean_disp(grid: torch.Tensor, h: int, w: int) -> float:
    g = grid[0].double()
    return float(torch.sqrt(((g[..., 0] + 1) / 2 * (w - 1) - torch.arange(w, device=g.device)) ** 2
                            + ((g[..., 1] + 1) / 2 * (h - 1) - torch.arange(h, device=g.device)[:, None]) ** 2).mean())


def cam_views(adv_pixels: torch.Tensor, n: int, theta_deg: float = 15.0, kappa: float = 0.0, sigma: float = 6.0,
              companion=None, data_sample=None, log=None, gt_log=None):
    """n view; theta_deg = 0 -> không xoay; kappa = 0 -> không méo. Trả về [(view_nhiễu, companion_warp, ds_view)]."""
    h, w = adv_pixels.shape[-2:]
    gt = get_gt(data_sample, adv_pixels.device) if data_sample is not None else None
    th = math.radians(theta_deg)
    out = []
    for _ in range(n):
        yaw = random.uniform(-th, th) if th > 0 else 0.0
        pitch = random.uniform(-th, th) if th > 0 else 0.0
        k = random.uniform(-kappa, kappa) if kappa > 0 else 0.0
        grid = camera_grid(h, w, adv_pixels.device, yaw, pitch, k)
        if log is not None:
            log.append(mean_disp(grid, h, w))
        view = warp(adv_pixels, grid)
        view = gaussian_blur(view, sigma=sigma) if sigma > 0 else view
        dsv = None
        if gt is not None:
            boxes, masks = gt
            nb, keep = warp_boxes_by_grid(boxes, grid, h, w)
            m = warp(masks, grid) if masks is not None and len(masks) else masks
            dsv = gt_with_keep(data_sample, nb, m, keep)
            if gt_log is not None:
                gt_log.append((int(keep.sum()), len(keep)))
        out.append((view, warp(companion, grid) if companion is not None else None, dsv))
    return out
