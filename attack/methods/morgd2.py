"""MORGD-v2 (quy tắc khóa: progress_log 2026-10-04 "Cải tiến MORGD"): trường phân vùng theo box, mỗi object di
chuyển gần như cứng theo phép đồng dạng của riêng nó; tùy chọn xếp chồng sau PIA.

Mỗi object i (box ở khung hiện tại; xếp chồng: box SAU PIA): t_i = (λ w_i r_x, λ h_i r_y), r ~ U(−1, 1),
a_i ~ U(−ln S_MAX, ln S_MAX); gauge: trừ trung bình có trọng số diện tích ω; s_i = exp(a_i); c'_i = c_i + t_i.
u_i(q) = c_i + (q − c'_i)/s_i − q (ánh xạ ngược đồng dạng chính xác).
Vùng (khung ra) = AABB(box gốc ∪ box đã dời cỡ s_i·(w_i, h_i) tại c'_i) nới thêm MARGIN·τ_i mỗi phía (sửa
sau unit test: không nới thì cạnh box nằm đúng chỗ α = 0.5, chỉ dời nửa — lệch 6 px); α_i = σ(d_x/τ_i)·σ(d_y/τ_i),
d = khoảng cách có dấu vào trong biên vùng, τ_i = 0.05·√(w_i h_i). Ghép "over" theo diện tích giảm dần:
u ← α_i u_i + (1 − α_i) u. SMOOTH-v2: cùng (t, s, kích thước theo chỉ số), tâm đều ngẫu nhiên trên khung.
"""
import math

import torch

from attack.methods.box_transforms import get_gt
from attack.methods.grgd import _box_pair_stats
from attack.methods.morgd import pixel_coords, to_grid, warp_boxes_by_grid
from attack.methods.pia import (DELTA_PAPER, PI_PAPER, gt_with_keep, homography, sample_source_corners,
                                warp, warp_boxes, warp_grid)
from attack.methods.rrb import gaussian_blur

LAMBDA = 0.25
S_MAX = 1.25
TAU_FRAC = 0.05
MARGIN = 3.0


def sample_motion(boxes: torch.Tensor, h: int, w: int, anchors: str, rng_m, rng_c=None,
                  lam: float = LAMBDA, s_max: float = S_MAX):
    """boxes [N,4] -> (centers [N,2], sizes [(w, h)], t [N,2], s [N]) float64 CPU, hoặc None nếu N < 2 / λ = 0
    (N = 1: gauge -> t = 0, s = 1, đồng nhất). Rút (r_x, r_y, a) tuần tự mỗi object từ rng_m."""
    n = len(boxes)
    if n < 2 or lam == 0:
        return None
    b = boxes.double().cpu()
    sizes = [(max(float(x2 - x1), 1.0), max(float(y2 - y1), 1.0)) for x1, y1, x2, y2 in b.tolist()]
    raw = torch.tensor([(lam * bw * rng_m.uniform(-1, 1), lam * bh * rng_m.uniform(-1, 1),
                         rng_m.uniform(-math.log(s_max), math.log(s_max))) for bw, bh in sizes], dtype=torch.float64)
    area = torch.tensor([bw * bh for bw, bh in sizes], dtype=torch.float64)
    omega = area / area.sum()
    raw = raw - omega @ raw
    t, s = raw[:, :2], torch.exp(raw[:, 2])
    if anchors == "object":
        c = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    else:
        assert anchors == "random"
        c = torch.tensor([(rng_c.uniform(0, w - 1), rng_c.uniform(0, h - 1)) for _ in range(n)], dtype=torch.float64)
    return c, sizes, t, s


def field(centers, sizes, t, s, xs: torch.Tensor, ys: torch.Tensor):
    """-> (ux, uy) tại (xs, ys)."""
    ux = torch.zeros_like(xs); uy = torch.zeros_like(xs)
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][0] * sizes[i][1])
    for i in order:
        (cx, cy), (bw, bh), (tx, ty), si = centers[i].tolist(), sizes[i], t[i].tolist(), float(s[i])
        nx, ny = cx + tx, cy + ty
        x1, x2 = min(cx - bw / 2, nx - si * bw / 2), max(cx + bw / 2, nx + si * bw / 2)
        y1, y2 = min(cy - bh / 2, ny - si * bh / 2), max(cy + bh / 2, ny + si * bh / 2)
        tau = TAU_FRAC * math.sqrt(bw * bh)
        x1, x2, y1, y2 = x1 - MARGIN * tau, x2 + MARGIN * tau, y1 - MARGIN * tau, y2 + MARGIN * tau
        a = torch.sigmoid(torch.minimum(xs - x1, x2 - xs) / tau) * torch.sigmoid(torch.minimum(ys - y1, y2 - ys) / tau)
        ux = a * (cx + (xs - nx) / si - xs) + (1 - a) * ux
        uy = a * (cy + (ys - ny) / si - ys) + (1 - a) * uy
    return ux, uy


def _fidelity(pre, post, keep, t, anchors):
    """ratio_pos / cos của dời tâm thực tế so với t_i (chỉ bản object, object |t_i| ≥ 1 px)."""
    out = []
    if anchors != "object" or not keep.any():
        return out
    cen = lambda b: torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    real = (cen(post) - cen(pre[keep])).double().cpu()
    tgt = t[keep.cpu()]
    ok = tgt.norm(dim=1) >= 1
    for r, g in zip(real[ok], tgt[ok]):
        out.append((float(r.norm() / g.norm()), float(torch.nn.functional.cosine_similarity(r, g, dim=0))))
    return out


def morgd2_views(adv_pixels: torch.Tensor, n: int, data_sample, rng_m, rng_c=None, anchors: str = "object",
                 use_pia: bool = True, lam: float = LAMBDA, delta_max: float = DELTA_PAPER, pi=PI_PAPER,
                 sigma: float = 6.0, companion=None, with_gt: bool = False, log=None, gt_log=None, stats=None):
    """n view (MORGD-v2 [∘ PIA]); trả về [(view_nhiễu, companion_warp, data_sample_view | None)] như pia_views.
    use_pia: RNG mode/góc PIA = `random` như pia_views; nhiễu σ (torch) cộng SAU cả hai phép warp.
    Không có chuyển động (N < 2, λ = 0) -> trùng tuyệt đối pia_views (use_pia) / ảnh gốc + nhiễu."""
    h, w = adv_pixels.shape[-2:]
    dev = adv_pixels.device
    dst = torch.tensor([(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)], dtype=torch.float64)
    boxes, masks = get_gt(data_sample, dev)
    ys = xs = None
    out = []
    for _ in range(n):
        view, comp, m, nb = adv_pixels, companion, masks, boxes
        keep = torch.ones(len(boxes), dtype=torch.bool, device=dev)
        if use_pia:
            mode, src = sample_source_corners(h, w, delta_max, pi)
            if log is not None:
                log.append(mode)
            H = homography(dst, src)
            grid = warp_grid(H, h, w, dev)
            nb, keep = warp_boxes(boxes, H, h, w)
            view = warp(adv_pixels, grid)
            comp = warp(companion, grid) if companion is not None else None
            m = warp(masks, grid) if masks is not None and len(masks) else masks
        st = {"n_objects": int(keep.sum()), "moved": False}
        motion = sample_motion(nb, h, w, anchors, rng_m, rng_c, lam)
        if motion is not None:
            centers, sizes, t, s = motion
            if ys is None:
                ys, xs = pixel_coords(h, w, dev, torch.float64)
            ux, uy = field(centers, sizes, t, s, xs, ys)
            g2 = to_grid(xs, ys, ux, uy, h, w).float()
            view = warp(view, g2)
            comp = warp(comp, g2) if comp is not None else None
            st.update(moved=True, rms_u_px=float((ux ** 2 + uy ** 2).mean().sqrt()))
            if with_gt or stats is not None:
                nb2, keep2 = warp_boxes_by_grid(nb, g2, h, w)
                if stats is not None:
                    st.update(_box_pair_stats(nb, nb2, keep2))
                    st["fid"] = _fidelity(nb, nb2, keep2, t, anchors)
                if with_gt:
                    if m is not None and len(m):
                        full = torch.zeros_like(m)
                        full[keep] = warp(m[keep], g2)
                        m = full
                    kk = keep.clone()
                    kk[keep.clone()] = keep2
                    nb, keep = nb2, kk
        view = gaussian_blur(view, sigma=sigma) if sigma > 0 else view
        dsv = None
        if with_gt:
            dsv = gt_with_keep(data_sample, nb, m, keep)
            if gt_log is not None:
                gt_log.append((int(keep.sum()), len(keep)))
        if stats is not None:
            stats.append(st)
        out.append((view, comp, dsv))
    return out
