"""Pilot MORGD @ dev30 (quy tắc khóa: docs/progress_log.md entry 2026-10-04 "Pilot MORGD").

Trường dịch chuyển mượt toàn ảnh, neo tại từng GT object (MORGD) hoặc tại vị trí ngẫu nhiên với
kích thước GT hoán vị (SMOOTH, đối chứng cùng biên độ). Mỗi neo i (tâm c, cỡ w × h ở khung model):
  Δ = (λ·w·r_x, λ·h·r_y), r ~ U(−1, 1);  s = exp(U(−ln S_MAX, ln S_MAX))
  u_i(q) = (q − c)(1/s − 1) − Δ/s          (ánh xạ ngược của phép đồng dạng: c -> c + Δ, co giãn s)
  K_i(q) = exp(−½[((q_x − c'_x)/w)² + ((q_y − c'_y)/h)²]),  c' = c + Δ
  u = Σ K_i u_i / (Σ K_i + ε);  ảnh ra(q) = x(q + u(q))  (grid_sample bilinear, ngoài = 0)
Tham số tự chọn, khóa trước khi chạy, không tune.
"""
import math
import random

import torch

from attack.methods.box_transforms import get_gt
from attack.methods.pia import gt_with_keep, warp
from attack.methods.rrb import gaussian_blur

LAMBDA = 0.25
S_MAX = 1.25
EPS = 0.05


def sample_anchors(boxes: torch.Tensor, h: int, w: int, anchors: str, lam: float = LAMBDA,
                   s_max: float = S_MAX):
    """boxes [N,4] xyxy (khung model) -> list neo (cx, cy, bw, bh, dx, dy, s). anchors: "object" | "random".
    "random": tâm đều trên ảnh, kích thước = kích thước GT hoán vị ngẫu nhiên."""
    sizes = [(max(float(x2 - x1), 1.0), max(float(y2 - y1), 1.0)) for x1, y1, x2, y2 in boxes.tolist()]
    if anchors == "object":
        centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes.tolist()]
    else:
        assert anchors == "random"
        centers = [(random.uniform(0, w - 1), random.uniform(0, h - 1)) for _ in sizes]
        random.shuffle(sizes)
    out = []
    for (cx, cy), (bw, bh) in zip(centers, sizes):
        dx = lam * bw * random.uniform(-1, 1)
        dy = lam * bh * random.uniform(-1, 1)
        s = math.exp(random.uniform(-math.log(s_max), math.log(s_max)))
        out.append((cx, cy, bw, bh, dx, dy, s))
    return out


def flow_field(anchors, xs: torch.Tensor, ys: torch.Tensor, eps: float = EPS):
    """Trường u = Σ K_i u_i / (Σ K_i + ε) tại tọa độ (xs, ys) bất kỳ. Trả về (ux, uy, Σ K_i)."""
    num_x = torch.zeros_like(xs); num_y = torch.zeros_like(xs); den = torch.zeros_like(xs)
    for cx, cy, bw, bh, dx, dy, s in anchors:
        ncx, ncy = cx + dx, cy + dy
        k = torch.exp(-0.5 * (((xs - ncx) / bw) ** 2 + ((ys - ncy) / bh) ** 2))
        num_x += k * ((xs - cx) * (1 / s - 1) - dx / s)
        num_y += k * ((ys - cy) * (1 / s - 1) - dy / s)
        den += k
    return num_x / (den + eps), num_y / (den + eps), den


def pixel_coords(h: int, w: int, device, dtype=torch.float32):
    return torch.meshgrid(torch.arange(h, dtype=dtype, device=device),
                          torch.arange(w, dtype=dtype, device=device), indexing="ij")


def to_grid(xs, ys, ux, uy, h: int, w: int) -> torch.Tensor:
    """Lưới grid_sample [1,h,w,2] (align_corners=True) lấy mẫu tại (xs + ux, ys + uy)."""
    return torch.stack([(xs + ux) / (w - 1) * 2 - 1, (ys + uy) / (h - 1) * 2 - 1], dim=-1).unsqueeze(0)


def flow_grid(anchors, h: int, w: int, device, eps: float = EPS):
    """Trả về (grid [1,h,w,2] cho grid_sample align_corners=True, |u| trung bình px)."""
    ys, xs = pixel_coords(h, w, device)
    ux, uy, _ = flow_field(anchors, xs, ys, eps)
    return to_grid(xs, ys, ux, uy, h, w), float(torch.sqrt(ux ** 2 + uy ** 2).mean())


def warp_boxes_by_grid(boxes: torch.Tensor, grid: torch.Tensor, h: int, w: int):
    """AABB của ảnh warp mặt nạ chữ nhật mỗi box (≥ 0.5). Trả về (box [M,4], keep [N] bool)."""
    if len(boxes) == 0:
        return boxes, torch.zeros(0, dtype=torch.bool, device=boxes.device)
    rect = torch.zeros(len(boxes), h, w, device=grid.device)
    for j, (x1, y1, x2, y2) in enumerate(boxes.tolist()):
        rect[j, int(y1):max(int(math.ceil(y2)), int(y1) + 1), int(x1):max(int(math.ceil(x2)), int(x1) + 1)] = 1
    wr = warp(rect, grid) >= 0.5
    out, keep = [], []
    for m in wr:
        ys, xs = torch.nonzero(m, as_tuple=True)
        ok = len(xs) > 0
        keep.append(ok)
        if ok:
            out.append(torch.stack([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]).float())
    keep = torch.tensor(keep, dtype=torch.bool, device=boxes.device)
    out = torch.stack(out) if out else boxes.new_zeros(0, 4)
    big = (out[:, 2] - out[:, 0] >= 1) & (out[:, 3] - out[:, 1] >= 1)
    keep[keep.clone()] = big
    return out[big], keep


def morgd_views(adv_pixels: torch.Tensor, n: int, data_sample, anchors: str = "object", sigma: float = 6.0,
                companion=None, with_gt: bool = False, log=None, gt_log=None):
    """n view; trả về [(view_nhiễu, companion_warp, data_sample_view | None)] như pia.pia_views."""
    h, w = adv_pixels.shape[-2:]
    boxes, masks = get_gt(data_sample, adv_pixels.device)
    out = []
    for _ in range(n):
        grid, disp = flow_grid(sample_anchors(boxes, h, w, anchors), h, w, adv_pixels.device)
        if log is not None:
            log.append(disp)
        view = warp(adv_pixels, grid)
        view = gaussian_blur(view, sigma=sigma) if sigma > 0 else view
        dsv = None
        if with_gt:
            nb, keep = warp_boxes_by_grid(boxes, grid, h, w)
            m = warp(masks, grid) if masks is not None and len(masks) else masks
            dsv = gt_with_keep(data_sample, nb, m, keep)
            if gt_log is not None:
                gt_log.append((int(keep.sum()), len(keep)))
        out.append((view, warp(companion, grid) if companion is not None else None, dsv))
    return out
