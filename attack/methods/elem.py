"""Quét liều phép biến đổi cơ bản (quy tắc khóa: progress_log 2026-10-05 "Phép biến đổi cơ bản").

Mỗi view: phép tuyến tính A (2×2) quanh TÂM ảnh, không tịnh tiến, không phối cảnh; H = C·[[A, 0], [0, 1]]·C⁻¹ ánh xạ
pixel ra -> nguồn (như pia.warp_grid), ngoài ảnh nguồn lấp 0. Tham số rút đều mỗi view từ `random` của Python:
  rot:   θ ~ U(−a, a) độ,        A = R(θ)
  shear: k ~ U(−a, a), hướng ngang/dọc xác suất 0.5,  A = [[1, k], [0, 1]] | [[1, 0], [k, 1]]
  aniso: ln k ~ U(−a, a),        A = diag(k, 1/k)
  scale: ln s ~ U(−a, a),        A = s·I
spec nhiều khóa (bước 2, ghép trong cùng view) -> A = tích theo thứ tự ELEM_ORDER, mỗi phép rút độc lập.
"""
import math
import random

import torch

from attack.methods.camgeo import mean_disp
from attack.methods.pia import warp, warp_grid
from attack.methods.rrb import gaussian_blur

ELEM_ORDER = ("rot", "shear", "aniso", "scale")


def elem_matrix(kind: str, a: float) -> torch.Tensor:
    if kind == "rot":
        t = math.radians(random.uniform(-a, a))
        return torch.tensor([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], dtype=torch.float64)
    if kind == "shear":
        k = random.uniform(-a, a)
        return torch.tensor([[1.0, k], [0.0, 1.0]] if random.random() < 0.5 else [[1.0, 0.0], [k, 1.0]],
                            dtype=torch.float64)
    if kind == "aniso":
        k = math.exp(random.uniform(-a, a))
        return torch.tensor([[k, 0.0], [0.0, 1 / k]], dtype=torch.float64)
    if kind == "scale":
        s = math.exp(random.uniform(-a, a))
        return torch.tensor([[s, 0.0], [0.0, s]], dtype=torch.float64)
    raise ValueError(kind)


def linear_homography(A: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """H (pixel ra -> nguồn) của phép tuyến tính A quanh tâm ảnh."""
    C = torch.tensor([[1, 0, (w - 1) / 2], [0, 1, (h - 1) / 2], [0, 0, 1]], dtype=torch.float64)
    Hc = torch.eye(3, dtype=torch.float64)
    Hc[:2, :2] = A
    return C @ Hc @ torch.linalg.inv(C)


def shift_offset(mode: str) -> float:
    """Kiểm "pha lưới" (progress_log 2026-10-07): 1 trục tịnh tiến (px, khung model).
    s<k> (k ∈ {2, 4, 8, 16, 32}): bội số k trong ±64, gồm 0 (giữ pha lưới mọi stride ≤ k; "Stride quyết định",
    progress_log 2026-10-08 — s32 trùng tuyệt đối bản cũ, cả RNG); odd: số nguyên ±64 KHÔNG chia hết 32;
    sub: dưới 1 px, U(−1, 1) (lệch pha ở mọi stride, nội suy bilinear)."""
    if mode in ("s2", "s4", "s8", "s16", "s32"):
        return float(random.choice(list(range(-64, 65, int(mode[1:])))))
    if mode == "odd":
        return float(random.choice([t for t in range(-64, 65) if t % 32 != 0]))
    if mode == "sub":
        return random.uniform(-1.0, 1.0)
    raise ValueError(mode)


def sample_elem_h(spec: dict, h: int, w: int) -> torch.Tensor:
    if "shift" in spec:  # chỉ tịnh tiến: pixel ra q lấy mẫu ở q − t (nội dung dời +t)
        H = torch.eye(3, dtype=torch.float64)
        H[0, 2], H[1, 2] = -shift_offset(spec["shift"]), -shift_offset(spec["shift"])
        return H
    A = torch.eye(2, dtype=torch.float64)
    for kind in ELEM_ORDER:
        if kind in spec:
            A = A @ elem_matrix(kind, spec[kind])
    return linear_homography(A, h, w)


def elem_views(adv_pixels: torch.Tensor, n: int, spec: dict, sigma: float = 6.0, companion=None, log=None):
    """n view; companion (ảnh sạch) nhận CÙNG lưới, không nhiễu. Trả về [(view_nhiễu, companion_warp, None)]."""
    h, w = adv_pixels.shape[-2:]
    out = []
    for _ in range(n):
        grid = warp_grid(sample_elem_h(spec, h, w), h, w, adv_pixels.device)
        if log is not None:
            log.append(mean_disp(grid, h, w))
        view = warp(adv_pixels, grid)
        out.append((gaussian_blur(view, sigma=sigma) if sigma > 0 else view,
                    warp(companion, grid) if companion is not None else None, None))
    return out
