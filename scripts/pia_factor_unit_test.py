#!/usr/bin/env python3
"""Unit test Giai đoạn A (progress_log 2026-10-04): tách homography PIA thành T, R, S, K, P.

1. Ghép lại == H (sai số tương đối < 1e-9, 10 000 mẫu PIA).
2. Bỏ từng thành phần: tách lại -> thành phần đó = đồng nhất, các thành phần khác không đổi.
3. Mode 3/2/1 (v = 0 sẵn): lưới PIA-noP ≈ PIA; mode 1: lưới PIA-noT ≈ đồng nhất.
4. Không truyền drop -> PIA trùng tuyệt đối HEAD.
5. 5 ô chạy được, L∞ đúng; mode PIA + nhiễu σ trùng PIA.
Báo phân bố thành phần trong PIA và |u| trung bình mỗi ô.
"""
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.methods import pia
from attack.models import load_surrogate
from attack.preprocess import resize_to_model
from scripts.pia_unit_test import check, load_head, seed

DEV = "cuda:0"


def disp(H, h, w):
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    p = torch.stack([xs, ys, torch.ones_like(xs)], -1) @ H.T
    return float(torch.sqrt((p[..., 0] / p[..., 2] - xs) ** 2 + (p[..., 1] / p[..., 2] - ys) ** 2).mean())


def main():
    h, w = 800, 1199
    dst = torch.tensor([(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)], dtype=torch.float64)
    random.seed(0)
    errs, other_err, comps, modes = [], 0.0, [], []
    keys = ["t", "theta", "s", "k1", "k2", "v"]
    owned = {"T": ["t"], "R": ["theta"], "S": ["s"], "K": ["k1", "k2"], "P": ["v"]}
    for i in range(10000):
        mode, src = pia.sample_source_corners(h, w, pia.DELTA_PAPER)
        H = pia.homography(dst, src)
        f = pia.decompose_h(H, h, w)
        R = pia.compose_h(f, h, w)
        errs.append(float((R / R[2, 2] - H / H[2, 2]).norm() / (H / H[2, 2]).norm()))
        modes.append(mode)
        comps.append(f)
        if i < 500:
            for fac in pia.FACTORS:
                Hd, _ = pia.drop_factor(H, fac, h, w)
                g = pia.decompose_h(Hd, h, w)
                for k in keys:
                    exp = pia.IDENTITY[fac][k] if k in owned[fac] else f[k]
                    other_err = max(other_err, float((g[k] - exp).abs().max()))
    check(max(errs) < 1e-9, f"ghép lại == H (sai số tương đối tối đa {max(errs):.1e}, 10 000 mẫu)")
    check(other_err < 1e-8, f"bỏ 1 thành phần: thành phần đó = đồng nhất, các thành phần khác không đổi (lệch {other_err:.1e})")
    ar = lambda k, fn: np.array([fn(c[k]) for c in comps])
    print("     PIA (10 000 mẫu, 800×1199): " + ", ".join([
        f"|t| px trung vị {np.median(ar('t', lambda x: float(x.norm()))):.1f}",
        f"|θ|° {np.median(ar('theta', lambda x: float(x.abs()) * 180 / np.pi)):.2f}",
        f"|ln s| {np.median(ar('s', lambda x: abs(float(x.log())))):.3f}",
        f"|ln k1| {np.median(ar('k1', lambda x: abs(float(x.log())))):.3f}",
        f"|k2| {np.median(ar('k2', lambda x: abs(float(x)))):.3f}",
        f"‖v‖·w {np.median(ar('v', lambda x: float(x.norm()) * w)):.3f}"]))

    # 3
    random.seed(1)
    worst_p, worst_t = 0.0, 0.0
    for _ in range(300):
        mode, src = pia.sample_source_corners(h, w, pia.DELTA_PAPER)
        H = pia.homography(dst, src)
        if mode in (1, 2, 3):
            g0, g1 = pia.warp_grid(H, h, w, "cpu"), pia.warp_grid(pia.drop_factor(H, "P", h, w)[0], h, w, "cpu")
            worst_p = max(worst_p, float((g0 - g1).abs().max()))
        if mode == 1:
            gi = pia.warp_grid(torch.eye(3, dtype=torch.float64), h, w, "cpu")
            worst_t = max(worst_t, float((pia.warp_grid(pia.drop_factor(H, "T", h, w)[0], h, w, "cpu") - gi).abs().max()))
    check(worst_p < 1e-4 and worst_t < 1e-4, f"mode 3/2/1: PIA-noP ≈ PIA (lưới lệch {worst_p:.1e}); mode 1: PIA-noT ≈ đồng nhất ({worst_t:.1e})")

    # |u| mỗi ô
    random.seed(2)
    du = {k: [] for k in ["PIA"] + [f"no{f}" for f in pia.FACTORS]}
    for _ in range(200):
        _, src = pia.sample_source_corners(h, w, pia.DELTA_PAPER)
        H = pia.homography(dst, src)
        du["PIA"].append(disp(H, h, w))
        for fac in pia.FACTORS:
            du[f"no{fac}"].append(disp(pia.drop_factor(H, fac, h, w)[0], h, w))
    print("     |u| trung bình (px): " + ", ".join(f"{k} {np.mean(v):.1f}" for k, v in du.items()))

    # 4, 5
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))[:2]
    ds = AttackDataset(ids)
    model = load_surrogate(DEV)
    smp = ds[0]
    orig, dsm = smp["orig_inputs"].to(DEV), smp["data_sample"]
    seed(0); pia.geo_osfd_attack(model, orig, dsm, steps=1)
    old = load_head("attack/methods/pia.py", "pia_head")
    out = []
    for mod in (old, pia):
        seed(9); out.append(mod.geo_osfd_attack(model, orig, dsm, steps=3))
    check(torch.equal(out[0], out[1]), "PIA (không drop) trùng HEAD")
    blur0 = pia.gaussian_blur
    rec = {}
    for name, drop in [("PIA", None)] + [(f"PIA-no{f}", f) for f in pia.FACTORS]:
        nl, info = [], {}

        def gb(x, sigma=6.0, nl=nl):
            n_ = torch.randn_like(x)
            nl.append(n_.flatten()[:16].clone())
            return torch.clamp(x + n_ * sigma, 0.0, 255.0)
        pia.gaussian_blur = gb
        seed(12); noise = pia.geo_osfd_attack(model, orig, dsm, steps=3, info=info, pia_drop=drop)
        rec[name] = (info["modes"], nl)
        if drop is not None:
            check(bool(torch.isfinite(noise).all()) and noise.abs().max().item() <= 5.0 + 1e-4
                  and info["forward_no_grad"] == 6 and len(info["pia_factors"]) == 6,
                  f"{name}: chạy được, L∞ {noise.abs().max().item():.1f}, |u| {np.mean([v['disp_px'] for v in info['pia_factors']]):.1f} px")
    pia.gaussian_blur = blur0
    for name in rec:
        if name != "PIA":
            ok = rec[name][0] == rec["PIA"][0] and all(torch.equal(a, b) for a, b in zip(rec[name][1], rec["PIA"][1]))
            check(ok, f"{name}: mode PIA + nhiễu σ trùng PIA")
    print("TẤT CẢ ĐẠT")


if __name__ == "__main__":
    main()
