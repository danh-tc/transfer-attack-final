#!/usr/bin/env python3
"""Unit test Giai đoạn B (progress_log 2026-10-04 "đa dạng mô hình camera")."""
import math
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.methods import camgeo, pia
from attack.models import load_surrogate
from attack.preprocess import resize_to_model
from scripts.pia_unit_test import check, load_head, seed

DEV = "cuda:0"


def px(grid, h, w, y, x):
    g = grid[0, y, x].double()
    return float((g[0] + 1) / 2 * (w - 1)), float((g[1] + 1) / 2 * (h - 1))


def main():
    h, w = 800, 1199
    cx, cy, f = (w - 1) / 2, (h - 1) / 2, max(h, w)
    ident = camgeo.camera_grid(h, w, "cpu")
    check(camgeo.mean_disp(ident, h, w) < 1e-3, f"θ = 0, κ = 0 -> đồng nhất (|u| {camgeo.mean_disp(ident, h, w):.1e} px)")
    yaw = math.radians(10)
    g = camgeo.camera_grid(h, w, "cpu", yaw=yaw)
    # tâm ảnh (pixel gần tâm nhất) -> c_x + f·tan(yaw)
    x0, y0 = int(round(cx)), int(round(cy))
    sx, sy = px(g, h, w, y0, x0)
    check(abs(sx - x0) < 1.5 and abs(sy - y0) < 1.0, f"yaw 10° (căn tâm) -> tâm giữ nguyên ({sx:.1f}, {sy:.1f})")
    lx, _ = px(g, h, w, y0, 0); rx, _ = px(g, h, w, y0, w - 1)
    ly0, _ = px(g, h, w, 0, 0)
    check(abs((rx - (w - 1)) + lx) > 1.0, f"yaw 10° -> méo phối cảnh bất đối xứng trái/phải (mép trái dời {lx:.1f}, mép phải {rx - (w - 1):+.1f} px)")
    k = 0.2
    g = camgeo.camera_grid(h, w, "cpu", k=k)
    sx, sy = px(g, h, w, h - 1, w - 1)
    R = math.hypot(cx, cy)
    check(abs(sx - (w - 1 + k * (w - 1 - cx))) < 1e-2 and abs(sy - (h - 1 + k * (h - 1 - cy))) < 1e-2,
          f"LENS k = 0.2 -> góc phải dưới tới ({sx:.1f}, {sy:.1f}), lệch ra ngoài k·R = {k * R:.1f} px theo đường chéo")
    random.seed(0)
    for name, th, ka in (("CAM-R15", 15, 0), ("CAM-R30", 30, 0), ("LENS", 0, 0.25), ("CAM-R15+LENS", 15, 0.25)):
        d = []
        for _ in range(100):
            th_ = math.radians(th)
            yw = random.uniform(-th_, th_) if th else 0.0
            pt = random.uniform(-th_, th_) if th else 0.0
            kk = random.uniform(-ka, ka) if ka else 0.0
            d.append(camgeo.mean_disp(camgeo.camera_grid(h, w, "cpu", yw, pt, kk), h, w))
        print(f"     {name}: |u| trung bình {np.mean(d):.1f} px (PIA ~194, PIA-noT ~91)")

    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))[:2]
    ds = AttackDataset(ids)
    model = load_surrogate(DEV)
    smp = ds[0]
    orig, dsm = smp["orig_inputs"].to(DEV), smp["data_sample"]
    img = resize_to_model(orig, tuple(dsm.img_shape))
    seed(0); pia.geo_osfd_attack(model, orig, dsm, steps=1)
    seed(3)
    vv = camgeo.cam_views(img, 4, 15.0, 0.25, sigma=0, companion=img, data_sample=dsm)
    check(all(torch.equal(v, c) for v, c, _ in vv), "σ = 0, adv = sạch -> view == companion")
    old = load_head("attack/methods/pia.py", "pia_head")
    for name, kw in {"GEO-C0": dict(geo="rrb"), "PIA": dict(geo="pia"), "PIA-noT": dict(geo="pia", pia_drop="T")}.items():
        out = []
        for mod in (old, pia):
            seed(9); out.append(mod.geo_osfd_attack(model, orig, dsm, steps=3, **kw))
        check(torch.equal(out[0], out[1]), f"{name} trùng HEAD")
    for name, th, ka in (("CAM-R15", 15, 0), ("CAM-R30", 30, 0), ("LENS", 0, 0.25), ("CAM-R15+LENS", 15, 0.25)):
        info = {}
        seed(8)
        noise = pia.geo_osfd_attack(model, orig, dsm, steps=2, geo="cam", cam_theta=th, cam_kappa=ka, info=info)
        check(bool(torch.isfinite(noise).all()) and noise.abs().max().item() <= 5.0 + 1e-4 and info["forward_no_grad"] == 4,
              f"{name}: chạy được, L∞ {noise.abs().max().item():.1f}, |u| {np.mean(info['disp']):.1f} px")
    print("TẤT CẢ ĐẠT")


if __name__ == "__main__":
    main()
