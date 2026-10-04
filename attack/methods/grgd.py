"""Pilot GRGD-v0 @ dev30 (quy tắc khóa: docs/progress_log.md entry 2026-10-04 "Pilot GRGD-v0").

T_GRGD = T_relative ∘ T_PIA. Sau PIA (H như pia_views), trên GT còn lại (N object, N < 2 -> đồng nhất):
1. Đồ thị đầy đủ i < j; mỗi cạnh ξˣ, ξʸ, ξˢ ~ U(−1, 1) (RNG riêng):
   d_x = λ_r·(w_i + w_j)/2·ξˣ, d_y = λ_r·(h_i + h_j)/2·ξʸ, d_s = ln(S_MAX)·ξˢ.
2. A (E × N, A_ei = 1, A_ej = −1); giải riêng x, y, log-scale bằng KKT với Σ ω_i z_i = 0,
   ω_i = w_i h_i / Σ w h -> t_i = (z_x, z_y), s_i = exp(z_s)  (t_i − t_j ≈ d_ij).
3. u_raw = Σ K_i u_i / (Σ K_i + ε)  — đúng công thức MORGD (morgd.flow_field), neo (c_i, w_i, h_i, t_i, s_i).
4. (BỎ trước khi chạy attack — progress_log "GRGD-v0 sửa (B)": phép chiếu xóa chính chuyển động tương đối.)
   u_rel = u_raw. Fit H_r vẫn tính để CHẨN ĐOÁN (bao nhiêu phần chuyển động một homography giải thích được):
   5 điểm/object (4 góc + tâm box w_i × h_i đặt tại c'_i = c_i + t_i), p_m = q_m + u_raw(q_m); H_r: q -> p
   (DLT, Hartley, SVD); u_H(q) = H_r(q) − q; M = Σ K_i / (Σ K_i + ε). remove_projective=True = spec v0 gốc.
5. Ảnh ra = grid_sample(ảnh sau PIA, q + u_rel), ngoài = 0; GT: AABB mặt nạ chữ nhật warp, mask warp.
RESIDUAL (đối chứng): tâm neo c_i thay bằng tâm ngẫu nhiên đều trên khung (RNG riêng thứ hai); giữ kích
thước, ξ, d, nghiệm (t, s), kernel.
"""
import math

import torch

from attack.methods.box_transforms import get_gt
from attack.methods.morgd import EPS, flow_field, pixel_coords, to_grid, warp_boxes_by_grid
from attack.methods.pia import (DELTA_PAPER, PI_PAPER, gt_with_keep, homography, sample_source_corners,
                                warp, warp_boxes, warp_grid)
from attack.methods.rrb import gaussian_blur

LAMBDA_R = 0.25
S_MAX = 1.25
KKT_EPS = 1e-6


def complete_edges(n: int):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def sample_pairwise(sizes, edges, rng, lam_r: float = LAMBDA_R, s_max: float = S_MAX):
    """sizes [(w, h)], edges [(i, j)] -> (d_x, d_y, d_s) float64 [E]. Rút ξˣ, ξʸ, ξˢ tuần tự mỗi cạnh."""
    d = torch.zeros(3, len(edges), dtype=torch.float64)
    for e, (i, j) in enumerate(edges):
        xi_x, xi_y, xi_s = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)
        d[0, e] = lam_r * (sizes[i][0] + sizes[j][0]) / 2 * xi_x
        d[1, e] = lam_r * (sizes[i][1] + sizes[j][1]) / 2 * xi_y
        d[2, e] = math.log(s_max) * xi_s
    return d[0], d[1], d[2]


def incidence(n: int, edges) -> torch.Tensor:
    A = torch.zeros(len(edges), n, dtype=torch.float64)
    for e, (i, j) in enumerate(edges):
        A[e, i], A[e, j] = 1.0, -1.0
    return A


def solve_relative(A: torch.Tensor, d: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """argmin ‖A z − d‖² s.t. ωᵀz = 0 (KKT, +1e-6·I ổn định số)."""
    n = A.shape[1]
    K = torch.zeros(n + 1, n + 1, dtype=torch.float64)
    K[:n, :n] = A.T @ A + KKT_EPS * torch.eye(n, dtype=torch.float64)
    K[:n, n] = omega
    K[n, :n] = omega
    rhs = torch.zeros(n + 1, dtype=torch.float64)
    rhs[:n] = A.T @ d
    return torch.linalg.solve(K, rhs)[:n]


def solve_graph_motion(sizes, edges, d_x, d_y, d_s):
    """-> (t [N,2], s [N], ω [N]) float64."""
    n = len(sizes)
    A = incidence(n, edges)
    area = torch.tensor([w * h for w, h in sizes], dtype=torch.float64)
    omega = area / area.sum()
    t = torch.stack([solve_relative(A, d_x, omega), solve_relative(A, d_y, omega)], 1)
    return t, torch.exp(solve_relative(A, d_s, omega)), omega


def _hartley(x: torch.Tensor) -> torch.Tensor:
    m = x.mean(0)
    sc = math.sqrt(2) / (x - m).norm(dim=1).mean().clamp_min(1e-12)
    return torch.tensor([[sc, 0, -sc * m[0]], [0, sc, -sc * m[1]], [0, 0, 1]], dtype=torch.float64)


def _apply_h(H: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den, den


def fit_homography_dlt(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """H [3,3] float64 (H[2,2] = 1) sao cho p ~ H·q, bình phương tối thiểu đại số (DLT chuẩn hóa Hartley, SVD)."""
    q, p = q.double().cpu(), p.double().cpu()
    Tq, Tp = _hartley(q), _hartley(p)
    qx, qy, _ = _apply_h(Tq, q[:, 0], q[:, 1])
    px, py, _ = _apply_h(Tp, p[:, 0], p[:, 1])
    o, z = torch.ones_like(qx), torch.zeros_like(qx)
    r1 = torch.stack([-qx, -qy, -o, z, z, z, px * qx, px * qy, px], 1)
    r2 = torch.stack([z, z, z, -qx, -qy, -o, py * qx, py * qy, py], 1)
    _, _, Vh = torch.linalg.svd(torch.cat([r1, r2], 0))
    H = torch.linalg.inv(Tp) @ Vh[-1].view(3, 3) @ Tq
    return H / H[2, 2]


def control_points(centers: torch.Tensor, sizes) -> torch.Tensor:
    """5 điểm/object: 4 góc + tâm của box w × h đặt tại centers (= c'_i). -> [5N, 2] float64."""
    pts = []
    for (cx, cy), (w, h) in zip(centers.tolist(), sizes):
        pts += [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx, cy),
                (cx - w / 2, cy + h / 2), (cx + w / 2, cy + h / 2)]
    return torch.tensor(pts, dtype=torch.float64)


class RelativeField:
    """Trường u_rel (ánh xạ ngược pixel ra -> nguồn, khung model) cho neo (centers, sizes, t, s)."""

    def __init__(self, centers: torch.Tensor, sizes, t: torch.Tensor, s: torch.Tensor, eps: float = EPS,
                 remove_projective: bool = False):
        self.anchors = [(float(c[0]), float(c[1]), w, h, float(ti[0]), float(ti[1]), float(si))
                        for c, (w, h), ti, si in zip(centers, sizes, t, s)]
        self.eps = eps
        self.remove_projective = remove_projective
        self.new_centers = centers + t
        q = control_points(self.new_centers, sizes)
        ux, uy, _ = flow_field(self.anchors, q[:, 0], q[:, 1], eps)
        p = q + torch.stack([ux, uy], 1)
        self.H = fit_homography_dlt(q, p)
        hx, hy, _ = _apply_h(self.H, q[:, 0], q[:, 1])
        self.fit_rms = float(torch.sqrt(((hx - p[:, 0]) ** 2 + (hy - p[:, 1]) ** 2).mean()))
        self.raw_rms_ctrl = float(torch.sqrt((ux ** 2 + uy ** 2).mean()))

    def at(self, xs: torch.Tensor, ys: torch.Tensor):
        """-> dict u_raw, u_proj (= M·u_H, chẩn đoán), u_rel (mỗi cái (ux, uy)), den_h (mẫu số H_r).
        u_rel = u_raw, trừ khi remove_projective (spec v0 gốc): u_rel = u_raw − M·u_H."""
        rx, ry, ksum = flow_field(self.anchors, xs, ys, self.eps)
        H = self.H.to(xs.device, xs.dtype)
        hx, hy, den = _apply_h(H, xs, ys)
        m = ksum / (ksum + self.eps)
        px, py = m * (hx - xs), m * (hy - ys)
        rel = (rx - px, ry - py) if self.remove_projective else (rx, ry)
        return {"raw": (rx, ry), "proj": (px, py), "rel": rel, "den_h": den}


def relative_motion(boxes: torch.Tensor, h: int, w: int, anchors: str, rng_xi, rng_c=None,
                    lam_r: float = LAMBDA_R, s_max: float = S_MAX):
    """boxes [N,4] (sau PIA) -> (centers [N,2], sizes, t, s, edges, d [E,3]) float64, hoặc None nếu N < 2 / λ_r = 0.
    anchors="object": tâm GT; "random": tâm đều trên khung (rng_c), CÙNG kích thước theo chỉ số."""
    n = len(boxes)
    if n < 2 or lam_r == 0:
        return None
    b = boxes.double().cpu()
    sizes = [(max(float(x2 - x1), 1.0), max(float(y2 - y1), 1.0)) for x1, y1, x2, y2 in b.tolist()]
    edges = complete_edges(n)
    d_x, d_y, d_s = sample_pairwise(sizes, edges, rng_xi, lam_r, s_max)
    t, s, _ = solve_graph_motion(sizes, edges, d_x, d_y, d_s)
    if anchors == "object":
        centers = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    else:
        assert anchors == "random"
        centers = torch.tensor([(rng_c.uniform(0, w - 1), rng_c.uniform(0, h - 1)) for _ in range(n)],
                               dtype=torch.float64)
    return centers, sizes, t, s, edges, torch.stack([d_x, d_y, d_s], 1)


def _pair_stats(field: RelativeField, centers, sizes, t, s, edges, d, dense, device):
    """Thống kê mỗi view (progress_log: R_proj, D_pair_pos / _eff, D_pair_scale, |u|; cổng độ trung thực GRGD (B):
    D_pair_pos_req theo d_ij đã rút, ρ_pos = D_pair_pos_eff / D_pair_pos_req, RMS |u_rel|)."""
    norm = lambda u: torch.sqrt(u[0] ** 2 + u[1] ** 2)
    raw, proj, rel = norm(dense["raw"]), norm(dense["proj"]), norm(dense["rel"])
    nc = field.new_centers.to(device)
    e_rel = field.at(nc[:, 0], nc[:, 1])["rel"]
    eff = -torch.stack(e_rel, 1).cpu()  # dời tâm hiệu dụng ≈ −u_rel(c'_i)
    sq = torch.tensor([math.sqrt(w * h) for w, h in sizes], dtype=torch.float64)
    ii = torch.tensor([i for i, _ in edges]); jj = torch.tensor([j for _, j in edges])
    nrm = (sq[ii] + sq[jj]) / 2
    d_eff = (eff[ii] - eff[jj]) / nrm[:, None]
    d_req = d[:, :2] / nrm[:, None]
    return {
        "d_pair_pos_req": float(d_req.norm(dim=1).mean()),
        "rho_pos": float(d_eff.norm(dim=1).mean() / d_req.norm(dim=1).mean().clamp_min(1e-12)),
        "cos_pair_eff_req": float(torch.nn.functional.cosine_similarity(d_eff, d_req, dim=1).mean()),
        "d_pair_scale_req": float(d[:, 2].abs().mean()),
        "rms_rel_px": float(rel.pow(2).mean().sqrt()),
        "disp_raw_px": float(raw.mean()), "disp_proj_px": float(proj.mean()), "disp_rel_px": float(rel.mean()),
        "r_proj": float(1 - rel.pow(2).sum().sqrt() / (raw.pow(2).sum().sqrt() + 1e-12)),
        "d_pair_pos": float(((t[ii] - t[jj]).norm(dim=1) / nrm).mean()),
        "d_pair_pos_eff": float(((eff[ii] - eff[jj]).norm(dim=1) / nrm).mean()),
        "d_pair_scale": float((torch.log(s[ii] / s[jj])).abs().mean()),
        "fit_rms_ctrl_px": field.fit_rms, "raw_rms_ctrl_px": field.raw_rms_ctrl,
        "den_h_min": float(dense["den_h"].min()),
    }


def _box_pair_stats(pre: torch.Tensor, post: torch.Tensor, keep: torch.Tensor):
    """Thay đổi hình học tương đối THỰC TẾ của GT box qua warp relative (AABB mặt nạ chữ nhật), trên các cặp
    còn giữ cả hai: D_pair_pos_box (dời tâm tương đối / (√A_i + √A_j)/2), D_pair_scale_box = |ln((w'_i/w'_j)/(w_i/w_j))|
    (trung bình chiều rộng và chiều cao)."""
    a = pre[keep].double().cpu(); b = post.double().cpu()
    if len(a) < 2:
        return {}
    ii, jj = torch.triu_indices(len(a), len(a), 1)
    wa, ha = (a[:, 2] - a[:, 0]).clamp_min(1), (a[:, 3] - a[:, 1]).clamp_min(1)
    wb, hb = (b[:, 2] - b[:, 0]).clamp_min(1), (b[:, 3] - b[:, 1]).clamp_min(1)
    ca = torch.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2], 1)
    cb = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    sq = (wa * ha).sqrt()
    dpos = ((cb[ii] - cb[jj]) - (ca[ii] - ca[jj])).norm(dim=1) / ((sq[ii] + sq[jj]) / 2)
    lw = (torch.log(wb[ii] / wb[jj]) - torch.log(wa[ii] / wa[jj])).abs()
    lh = (torch.log(hb[ii] / hb[jj]) - torch.log(ha[ii] / ha[jj])).abs()
    return {"d_pair_pos_box": float(dpos.mean()), "d_pair_scale_box": float(((lw + lh) / 2).mean())}


def grgd_views(adv_pixels: torch.Tensor, n: int, data_sample, rng_xi, rng_c=None, anchors: str = "object",
               lam_r: float = LAMBDA_R, delta_max: float = DELTA_PAPER, pi=PI_PAPER, sigma: float = 6.0,
               companion=None, with_gt: bool = False, log=None, gt_log=None, stats=None):
    """n view GRGD; trả về [(view_nhiễu, companion_warp, data_sample_view | None)] như pia.pia_views.
    RNG: mode/góc PIA = `random` (như pia_views), nhiễu σ = torch (như pia_views, cộng SAU cả hai phép warp),
    ξ = rng_xi, tâm ngẫu nhiên = rng_c. N < 2 hoặc λ_r = 0 -> view trùng tuyệt đối pia_views."""
    h, w = adv_pixels.shape[-2:]
    dev = adv_pixels.device
    dst = torch.tensor([(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)], dtype=torch.float64)
    boxes, masks = get_gt(data_sample, dev)
    ys = xs = None
    out = []
    for _ in range(n):
        mode, src = sample_source_corners(h, w, delta_max, pi)
        if log is not None:
            log.append(mode)
        H = homography(dst, src)
        grid = warp_grid(H, h, w, dev)
        nb, keep = warp_boxes(boxes, H, h, w)
        st = {"n_objects": int(keep.sum()), "n_edges": 0}
        if stats is not None:
            g = grid[0].double()
            st["disp_pia_px"] = float(torch.sqrt(((g[..., 0] + 1) / 2 * (w - 1) - torch.arange(w, device=dev)) ** 2
                                                 + ((g[..., 1] + 1) / 2 * (h - 1) - torch.arange(h, device=dev)[:, None]) ** 2).mean())
        view = warp(adv_pixels, grid)
        comp = warp(companion, grid) if companion is not None else None
        m = warp(masks, grid) if masks is not None and len(masks) else masks
        motion = relative_motion(nb, h, w, anchors, rng_xi, rng_c, lam_r)
        if motion is not None:
            centers, sizes, t, s, edges, d = motion
            field = RelativeField(centers, sizes, t, s)
            if ys is None:
                ys, xs = pixel_coords(h, w, dev, torch.float64)
            dense = field.at(xs, ys)
            g2 = to_grid(xs, ys, *dense["rel"], h, w).float()
            view = warp(view, g2)
            comp = warp(comp, g2) if comp is not None else None
            if with_gt:
                nb2, keep2 = warp_boxes_by_grid(nb, g2, h, w)
                if m is not None and len(m):
                    full = torch.zeros_like(m)
                    full[keep] = warp(m[keep], g2)
                    m = full
                if stats is not None:
                    st.update(_box_pair_stats(nb, nb2, keep2))
                kk = keep.clone()
                kk[keep.clone()] = keep2
                nb, keep = nb2, kk
            st["n_edges"] = len(edges)
            if stats is not None:
                st.update(_pair_stats(field, centers, sizes, t, s, edges, d, dense, dev))
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
