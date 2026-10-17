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
from attack.methods.box_transforms import get_gt
from attack.methods.pia import gt_with_keep, warp, warp_boxes, warp_grid
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
    if mode == "ph4":  # pilot pha (progress_log 2026-10-08): chỉ pha mod 4, gần như không dời nội dung
        return float(random.randrange(4))
    if mode in ("ph8", "ph32"):  # độ phủ pha mod 8 / mod 32, dời ≤ 7 / 31 px
        return float(random.randrange(int(mode[2:])))
    if mode == "ph4l":  # đủ pha mod 4 + dời xa bội 32 (±64): tách "dời nội dung" khỏi "phủ pha stride thô"
        return float(32 * random.choice([-2, -1, 0, 1, 2]) + random.randrange(4))
    raise ValueError(mode)


def shift_h(tx: float, ty: float) -> torch.Tensor:
    """H (pixel ra -> nguồn) của phép tịnh tiến: pixel ra q lấy mẫu ở q − t (nội dung dời +t)."""
    H = torch.eye(3, dtype=torch.float64)
    H[0, 2], H[1, 2] = -tx, -ty
    return H


def sample_elem_h(spec: dict, h: int, w: int) -> torch.Tensor:
    if "shift" in spec and len(spec) == 1:  # chỉ tịnh tiến: pixel ra q lấy mẫu ở q − t (nội dung dời +t)
        return shift_h(shift_offset(spec["shift"]), shift_offset(spec["shift"]))
    if "shift" in spec:  # đa dạng tách cơ chế (progress_log 2026-10-08): tuyến tính quanh tâm ∘ dịch pha
        lin = sample_elem_h({k: v for k, v in spec.items() if k != "shift"}, h, w)
        return lin @ shift_h(shift_offset(spec["shift"]), shift_offset(spec["shift"]))
    A = torch.eye(2, dtype=torch.float64)
    for kind in ELEM_ORDER:
        if kind in spec:
            A = A @ elem_matrix(kind, spec[kind])
    return linear_homography(A, h, w)


FINEST_SCALE = 56.0  # Mask R-CNN SingleRoIExtractor: mức l khi 56·2^l ≤ √(wh) < 56·2^(l+1), clamp 0..3


def lvx_log_scale(boxes: torch.Tensor, max_oct: float = 1.0) -> float:
    """C2-LVX (progress_log 2026-10-08): ln s để 1 object GT ngẫu nhiên SANG phía kia ranh giới mức FPN gần nhất
    (x = log2(√(wh)/56), ranh giới x ∈ {1, 2, 3}), cách ranh giới U(0, 0.5) octave; |Δ| ≤ max_oct (clip -> có thể không qua).
    Không có box -> ln s ~ U(±max_oct/2) (như OCT)."""
    if len(boxes) == 0:
        return random.uniform(-max_oct / 2, max_oct / 2) * math.log(2)
    b = boxes[random.randrange(len(boxes))]
    sc = math.sqrt(max(float((b[2] - b[0]) * (b[3] - b[1])), 1e-6))
    x = math.log2(sc / FINEST_SCALE)
    bd = min(max(round(x), 1), 3)
    side = (1.0 if x < bd else -1.0) if x != bd else random.choice([-1.0, 1.0])
    target = bd + side * random.uniform(0.0, 0.5)
    return max(-max_oct, min(max_oct, target - x)) * math.log(2)


def strat_schedule(total: int, a: float) -> list:
    """OCT-1V-S (progress_log 2026-10-17): lịch PHỦ CÓ HỆ THỐNG thay cho rút độc lập, cho `total` view.
    Pha dịch (dx, dy) ∈ {0..7}²: nối các hoán vị ngẫu nhiên của đủ 64 tổ hợp (total ≥ 64 -> mỗi pha ≥ 1 lần).
    ln s: Latin hypercube trên [−a, a] — total ô đều, mỗi view 1 ô (vị trí ngẫu nhiên trong ô), thứ tự xáo.
    Dùng `random` (seed theo ảnh như mọi ô). Trả [(ln s, dx, dy)]."""
    grid = [(x, y) for x in range(8) for y in range(8)]
    phases = []
    while len(phases) < total:
        blk = grid[:]
        random.shuffle(blk)
        phases += blk
    cells = list(range(total))
    random.shuffle(cells)
    return [(-a + (c + random.random()) * 2 * a / total, *phases[t]) for t, c in enumerate(cells)]


def elem_views(adv_pixels: torch.Tensor, n: int, spec: dict, sigma: float = 6.0, companion=None, log=None,
               data_sample=None, gt_log=None, fixed=None):
    """n view; companion (ảnh sạch) nhận CÙNG lưới, không nhiễu. Trả về [(view_nhiễu, companion_warp, ds_view)];
    ds_view = None nếu không truyền data_sample, ngược lại GT co-transform như pia_views (C2-PH8, progress_log 2026-10-08).
    fixed = [(ln s, dx, dy)] × n (strat_schedule): dùng thay cho rút ngẫu nhiên co giãn ∘ dịch pha."""
    h, w = adv_pixels.shape[-2:]
    gt = get_gt(data_sample, adv_pixels.device) if data_sample is not None else None
    out = []
    prev = None
    for k in range(n):
        if fixed is not None:  # cùng thứ tự hợp thành như sample_elem_h: tuyến tính quanh tâm ∘ dịch pha
            u, dx, dy = fixed[k]
            sc = math.exp(u)
            H = linear_homography(torch.tensor([[sc, 0.0], [0.0, sc]], dtype=torch.float64), h, w) @ shift_h(dx, dy)
        elif "lvx" in spec:  # co giãn vượt mức FPN theo 1 object ∘ dịch pha
            sc = math.exp(lvx_log_scale(gt[0] if gt is not None else torch.zeros(0, 4), spec["lvx"]))
            H = linear_homography(torch.tensor([[sc, 0.0], [0.0, sc]], dtype=torch.float64), h, w)
            if "shift" in spec:
                H = H @ shift_h(shift_offset(spec["shift"]), shift_offset(spec["shift"]))
        elif "scale_strat" in spec:  # co giãn phân tầng: view lẻ lệch nửa khoảng (a) so với view chẵn trước, + dịch pha
            a = spec["scale_strat"]
            if k % 2 == 0:
                u = random.uniform(-a, a)
            else:
                u = u + a if u < 0 else u - a
            sc = math.exp(u)
            H = linear_homography(torch.tensor([[sc, 0.0], [0.0, sc]], dtype=torch.float64), h, w)
            if "shift" in spec:
                H = H @ shift_h(shift_offset(spec["shift"]), shift_offset(spec["shift"]))
        elif spec.get("shift") == "ph4a":  # đối pha: view lẻ = view chẵn trước nó + (2, 2) mod 4 (nửa chu kỳ lưới 4)
            if k % 2 == 0:
                prev = (random.randrange(4), random.randrange(4))
                H = shift_h(*prev)
            else:
                H = shift_h((prev[0] + 2) % 4, (prev[1] + 2) % 4)
        else:
            H = sample_elem_h(spec, h, w)
        grid = warp_grid(H, h, w, adv_pixels.device)
        if log is not None:
            log.append(mean_disp(grid, h, w))
        view = warp(adv_pixels, grid)
        dsv = None
        if gt is not None:
            boxes, masks = gt
            nb, keep = warp_boxes(boxes, H, h, w)
            dsv = gt_with_keep(data_sample, nb, warp(masks, grid) if masks is not None and len(masks) else masks, keep)
            if gt_log is not None:
                gt_log.append((int(keep.sum()), len(keep)))
        out.append((gaussian_blur(view, sigma=sigma) if sigma > 0 else view,
                    warp(companion, grid) if companion is not None else None, dsv))
    return out
