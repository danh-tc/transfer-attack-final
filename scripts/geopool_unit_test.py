#!/usr/bin/env python3
"""Unit test FLIP / NOISE / POOL (progress_log 2026-10-04)."""
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.methods import geopool, pia
from attack.models import load_surrogate
from attack.preprocess import resize_to_model
from scripts.pia_unit_test import check, load_head, seed

DEV = "cuda:0"


def main():
    ds = AttackDataset(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))[:1])
    model = load_surrogate(DEV)
    smp = ds[0]
    orig, dsm = smp["orig_inputs"].to(DEV), smp["data_sample"]
    img = resize_to_model(orig, tuple(dsm.img_shape))
    h, w = img.shape[-2:]
    seed(0); pia.geo_osfd_attack(model, orig, dsm, steps=1)
    v = geopool.flip_views(img, 3, p=1.0, sigma=0, companion=img)
    check(all(torch.equal(a, torch.flip(img, dims=[-1])) and torch.equal(c, a) for a, c, _ in v), "FLIP p = 1, σ = 0 -> view == lật ngang, companion cùng phép")
    v = geopool.flip_views(img, 3, p=0.0, sigma=0, companion=img)
    check(all(torch.equal(a, img) for a, _, _ in v), "NOISE σ = 0 -> view == ảnh")
    seed(4)
    v = geopool.pool_views(img, 6, sigma=0, companion=img)
    check(all(torch.equal(a, c) for a, c, _ in v), "POOL σ = 0, adv = sạch -> view == companion")
    # nhánh (a) trùng lưới PIA-noT khi cùng tham số
    random.seed(11)
    while True:
        st = random.getstate()
        g, br = geopool.pool_grid(h, w, DEV)
        if br == "a":
            break
    random.setstate(st)
    random.random()  # bỏ lần rút chọn nhánh
    dst = torch.tensor([(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)], dtype=torch.float64)
    _, src = pia.sample_source_corners(h, w, pia.DELTA_PAPER)
    g2 = pia.warp_grid(pia.drop_factor(pia.homography(dst, src), "T", h, w)[0], h, w, DEV)
    check(torch.equal(g, g2), "POOL nhánh (a) == lưới PIA-noT cùng tham số")
    old = load_head("attack/methods/pia.py", "pia_head")
    for name, kw in {"PIA": dict(geo="pia"), "PIA-noT": dict(geo="pia", pia_drop="T"), "CAM-R30": dict(geo="cam", cam_theta=30.0)}.items():
        out = []
        for mod in (old, pia):
            seed(9); out.append(mod.geo_osfd_attack(model, orig, dsm, steps=3, **kw))
        check(torch.equal(out[0], out[1]), f"{name} trùng HEAD")
    for geo in ("noise", "flip", "pool"):
        info = {}
        seed(8); noise = pia.geo_osfd_attack(model, orig, dsm, steps=2, geo=geo, info=info)
        check(bool(torch.isfinite(noise).all()) and noise.abs().max().item() <= 5.0 + 1e-4 and info["forward_no_grad"] == 4,
              f"{geo}: chạy được, L∞ {noise.abs().max().item():.1f}" + (f", |u| {np.mean(info['disp']):.1f} px" if info["disp"] else ""))
    print("TẤT CẢ ĐẠT")


if __name__ == "__main__":
    main()
