#!/usr/bin/env python3
"""Thí nghiệm cơ chế (quy tắc khóa: progress_log 2026-10-04 "đa dạng có ích = biến đổi NGOÀI nhóm đối xứng").

Đa dạng hiệu dụng d_T = 1 − cos(∇ₓL(T(x)), ∇ₓL(x)) trên pixel view phủ; L = loss feature OSFD (k = 3) với đích
co-transform F(T(x_sạch)), không nhiễu σ. Họ biến đổi cơ bản quanh tâm ảnh, hiệu chỉnh cùng |u| = 10 / 30 / 60 px.
  python scripts/mech_equivariance.py   -> results/mechanism/equivariance.json
"""
import json
import math
import os
import random
import sys

import mmcv
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.evaluation import bootstrap_indices
from attack.methods.camgeo import mean_disp, rotation_homography
from attack.models import load_surrogate
from attack.preprocess import extract_features, resize_to_model

ART = os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5/adv")
FAMILIES = ["translation", "scale", "rotation", "shear", "perspective", "lens"]
LEVELS = [10.0, 30.0, 60.0]
DRAWS = 4
K_FEAT = 3.0
BOOT_SEED, N_BOOT = 2026, 1000


def src_coords(fam, p, sgn, alt, h, w, step=1, base=False):
    """Tọa độ nguồn (xs, ys) float64 cho pixel ra; p = tham số độ lớn, sgn ∈ {±1}, alt ∈ [0, 1) chọn hướng.
    step > 1: lưới thưa (chỉ để hiệu chỉnh |u|); base: trả thêm tọa độ pixel ra."""
    ys, xs = torch.meshgrid(torch.arange(0, h, step, dtype=torch.float64), torch.arange(0, w, step, dtype=torch.float64), indexing="ij")
    if base:
        return xs, ys, src_coords(fam, p, sgn, alt, h, w, step)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    dx, dy = xs - cx, ys - cy
    if fam == "translation":
        phi = 2 * math.pi * alt
        return xs + p * math.cos(phi), ys + p * math.sin(phi)
    if fam == "scale":
        g = 1 + sgn * p
        return cx + dx * g, cy + dy * g
    if fam == "rotation":
        t = sgn * p
        return cx + math.cos(t) * dx - math.sin(t) * dy, cy + math.sin(t) * dx + math.cos(t) * dy
    if fam == "shear":
        return (xs + sgn * p * dy, ys) if alt < 0.5 else (xs, ys + sgn * p * dx)
    if fam == "perspective":
        H = rotation_homography(sgn * p, 0.0, h, w) if alt < 0.5 else rotation_homography(0.0, sgn * p, h, w)
        q = torch.stack([xs, ys, torch.ones_like(xs)], -1) @ H.T
        return q[..., 0] / q[..., 2], q[..., 1] / q[..., 2]
    if fam == "lens":
        g = 1 + sgn * p * (dx ** 2 + dy ** 2) / (cx ** 2 + cy ** 2)
        return cx + dx * g, cy + dy * g
    raise ValueError(fam)


def to_grid(sx, sy, h, w, device):
    return torch.stack([sx / (w - 1) * 2 - 1, sy / (h - 1) * 2 - 1], -1).float().unsqueeze(0).to(device)


def calibrate(fam, target, h, w):
    """Tham số p sao cho |u| trung bình = target (chia đôi; sgn = +1, alt = 0.25)."""
    lo, hi = 0.0, {"translation": 400.0, "scale": 1.0, "rotation": 1.5, "shear": 2.0, "perspective": 1.2, "lens": 2.0}[fam]
    for _ in range(40):
        mid = (lo + hi) / 2
        xs, ys, (sx, sy) = src_coords(fam, mid, 1, 0.25, h, w, step=4, base=True)
        d = float(torch.sqrt((sx - xs) ** 2 + (sy - ys) ** 2).mean())
        lo, hi = (mid, hi) if d < target else (lo, mid)
    return (lo + hi) / 2


def grad(model, x, clean, ds, grid=None):
    x = x.clone().requires_grad_(True)
    tv = (lambda t: F.grid_sample(t[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]) if grid is not None else (lambda t: t)
    with torch.no_grad():
        fc = [f.detach() for f in extract_features(model, tv(clean), ds)]
    fa = extract_features(model, tv(x), ds)
    loss = sum(F.mse_loss(K_FEAT * c, a) for c, a in zip(fc, fa))
    return torch.autograd.grad(loss, x)[0]


def main():
    dev = "cuda:0"
    model = load_surrogate(dev)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    ds_all = AttackDataset(ids)
    res = {pt: {fam: {lv: [] for lv in LEVELS} for fam in FAMILIES} for pt in ("delta0", "osfd")}
    sgn_agree = {pt: {fam: {lv: [] for lv in LEVELS} for fam in FAMILIES} for pt in ("delta0", "osfd")}
    params = {}
    rng = random.Random(2026)
    for i in range(len(ds_all)):
        smp = ds_all[i]
        ds = smp["data_sample"]
        clean = resize_to_model(smp["orig_inputs"].to(dev), tuple(ds.img_shape))
        h, w = clean.shape[-2:]
        png = torch.from_numpy(mmcv.imread(os.path.join(ART, "OSFD", f"{smp['img_id']}.png"))).permute(2, 0, 1).float().to(dev)
        adv = resize_to_model(png, tuple(ds.img_shape))
        pars = {(fam, lv): calibrate(fam, lv, h, w) for fam in FAMILIES for lv in LEVELS}
        params[smp["img_id"]] = {f"{f}@{lv}": p for (f, lv), p in pars.items()}
        draws = [(rng.choice([-1, 1]), rng.random()) for _ in range(DRAWS)]
        for pt, x in (("delta0", clean), ("osfd", adv)):
            gI = grad(model, x, clean, ds)
            for fam in FAMILIES:
                for lv in LEVELS:
                    ds_, sa = [], []
                    for sgn, alt in draws:
                        sx, sy = src_coords(fam, pars[(fam, lv)], sgn, alt, h, w)
                        valid = ((sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)).to(dev)
                        gT = grad(model, x, clean, ds, to_grid(sx, sy, h, w, dev))
                        a, b = gT[:, valid].flatten(), gI[:, valid].flatten()
                        ds_.append(float(1 - F.cosine_similarity(a, b, dim=0)))
                        sa.append(float((torch.sign(a) == torch.sign(b)).float().mean()))
                    res[pt][fam][lv].append(float(np.mean(ds_)))
                    sgn_agree[pt][fam][lv].append(float(np.mean(sa)))
        print(f"[equivariance] {i + 1}/{len(ds_all)}", flush=True)

    n = len(ds_all)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    out = {"n_images": n, "levels_px": LEVELS, "draws": DRAWS, "mean_d": {}, "mean_sign_agree": {}, "tests": {}}
    supported = True
    for pt in res:
        out["mean_d"][pt] = {fam: {str(lv): float(np.mean(res[pt][fam][lv])) for lv in LEVELS} for fam in FAMILIES}
        out["mean_sign_agree"][pt] = {fam: {str(lv): float(np.mean(sgn_agree[pt][fam][lv])) for lv in LEVELS} for fam in FAMILIES}
        out["tests"][pt] = {}
        for lv in LEVELS:
            M = {fam: (W @ np.array(res[pt][fam][lv])) / W.sum(1) for fam in FAMILIES}
            diff = np.minimum(M["rotation"], M["perspective"]) - np.maximum(M["translation"], M["scale"])
            lo, hi = np.percentile(diff[1:], [2.5, 97.5])
            out["tests"][pt][str(lv)] = {"min(rot,persp) - max(trans,scale)": float(diff[0]), "ci95": [float(lo), float(hi)]}
            supported &= bool(lo > 0)
    out["principle_supported"] = supported
    out["params_per_image"] = params
    os.makedirs(os.path.join(REPO_ROOT, "results/mechanism"), exist_ok=True)
    with open(os.path.join(REPO_ROOT, "results/mechanism/equivariance.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for pt in res:
        print(f"\n=== δ = {pt}: d_T = 1 − cos (trùng dấu) ===")
        print(f"{'họ':<13}" + "".join(f"{f'|u|={int(lv)}':>20}" for lv in LEVELS))
        for fam in FAMILIES:
            print(f"{fam:<13}" + "".join(f"{out['mean_d'][pt][fam][str(lv)]:>12.3f} ({out['mean_sign_agree'][pt][fam][str(lv)]:.2f})" for lv in LEVELS))
        for lv, t in out["tests"][pt].items():
            print(f"  |u|={lv}: min(rot,persp) − max(trans,scale) = {t['min(rot,persp) - max(trans,scale)']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]")
    print("\nNGUYÊN LÝ ĐƯỢC ỦNG HỘ:", supported)


if __name__ == "__main__":
    main()
