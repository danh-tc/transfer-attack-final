#!/usr/bin/env python3
"""Unit test pilot MORGD (progress_log 2026-10-04) — chạy TRƯỚC mọi attack.

1. Ô cũ trùng HEAD: GEO-C0, PIA tuyệt đối; C2-PIA (vốn không tất định — ROIAlign backward) cùng mức
   lệch HEAD/HEAD.
2. 1 neo, ε = 0 -> lưới trùng phép đồng dạng tương ứng (homography của pia).
3. Xa neo -> ảnh gần như giữ nguyên.
4. Box 1 neo, ε = 0 khớp c' ± s·(w, h)/2.
5. SMOOTH dùng đúng multiset kích thước GT.
6. C2-MORGD / C2-SMOOTH chạy được (2 step × 2 ảnh): hữu hạn, L∞ đúng, forward no_grad đúng.
   Báo |u| trung bình (px) mỗi view: MORGD / SMOOTH / PIA.
"""
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.methods import morgd, pia
from attack.methods.box_transforms import get_gt
from attack.models import load_surrogate
from attack.preprocess import resize_to_model
from scripts.pia_unit_test import check, load_head, seed

DEV = "cuda:0"


def main():
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))[:3]
    ds = AttackDataset(ids)
    model = load_surrogate(DEV)
    samples = [ds[i] for i in range(len(ds))]
    smp = samples[0]
    orig, dsm = smp["orig_inputs"].to(DEV), smp["data_sample"]
    seed(0); pia.geo_osfd_attack(model, orig, dsm, steps=1)  # khởi động GPU

    # 1. ô cũ
    old = load_head("attack/methods/pia.py", "pia_head")
    for name, kw in {"GEO-C0": dict(geo="rrb"), "PIA": dict(geo="pia")}.items():
        out = []
        for mod in (old, pia):
            seed(9); out.append(mod.geo_osfd_attack(model, orig, dsm, steps=3, **kw))
        check(torch.equal(out[0], out[1]), f"{name} trùng HEAD")
    out = []
    for mod in (old, old, pia):
        seed(9); out.append(mod.geo_osfd_attack(model, orig, dsm, steps=3, cls_weight=4.0))
    hh, hn = (out[0] != out[1]).float().mean().item(), (out[0] != out[2]).float().mean().item()
    check(hn <= max(2 * hh, 0.10), f"C2-PIA: lệch HEAD/HEAD {hh:.3f}, HEAD/mới {hn:.3f}")

    img = resize_to_model(orig, tuple(dsm.img_shape))
    h, w = img.shape[-2:]

    # 2. 1 neo, ε = 0 == đồng dạng
    cx, cy, bw, bh, dx, dy, s = 400.0, 300.0, 120.0, 80.0, 17.0, -9.0, 1.2
    g_m, _ = morgd.flow_grid([(cx, cy, bw, bh, dx, dy, s)], h, w, DEV, eps=0.0)
    H = torch.tensor([[1 / s, 0, cx - (cx + dx) / s], [0, 1 / s, cy - (cy + dy) / s], [0, 0, 1]], dtype=torch.float64)
    g_h = pia.warp_grid(H, h, w, DEV)
    check((g_m - g_h).abs().max().item() < 1e-4, f"1 neo ε = 0 == đồng dạng (lưới lệch {(g_m - g_h).abs().max().item():.1e})")

    # 3. xa neo -> giữ nguyên
    g, _ = morgd.flow_grid([(100.0, 100.0, 50.0, 50.0, 12.0, 12.0, 1.25)], h, w, DEV)
    far = (pia.warp(img, g) - img)[:, :, 600:].abs().max().item()
    check(far < 0.5, f"xa neo (x ≥ 600, neo ở x = 100, cỡ 50) lệch tối đa {far:.3f}")

    # 4. box 1 neo
    box = torch.tensor([[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]], device=DEV)
    nb, keep = morgd.warp_boxes_by_grid(box, g_m, h, w)
    exp = torch.tensor([cx + dx - s * bw / 2, cy + dy - s * bh / 2, cx + dx + s * bw / 2, cy + dy + s * bh / 2], device=DEV)
    check(keep.all().item() and (nb[0] - exp).abs().max().item() <= 2.0, f"box 1 neo: {nb[0].tolist()} vs {exp.tolist()}")

    # 5. SMOOTH: multiset kích thước
    boxes, _ = get_gt(dsm, DEV)
    seed(1)
    a = morgd.sample_anchors(boxes, h, w, "random")
    gt_sizes = sorted((max(float(x2 - x1), 1.0), max(float(y2 - y1), 1.0)) for x1, y1, x2, y2 in boxes.tolist())
    check(sorted((x[2], x[3]) for x in a) == gt_sizes and len(a) == len(boxes), "SMOOTH: đúng multiset kích thước GT")

    # 6. chạy + biên độ
    disp = {"MORGD": [], "SMOOTH": [], "PIA": []}
    for smp_ in samples:
        d = smp_["data_sample"]
        im = resize_to_model(smp_["orig_inputs"].to(DEV), tuple(d.img_shape))
        hh_, ww_ = im.shape[-2:]
        seed(2)
        for name, anc in (("MORGD", "object"), ("SMOOTH", "random")):
            lg = []
            morgd.morgd_views(im, 10, d, anchors=anc, log=lg)
            disp[name] += lg
        dst = torch.tensor([(0.0, 0.0), (ww_ - 1.0, 0.0), (ww_ - 1.0, hh_ - 1.0), (0.0, hh_ - 1.0)], dtype=torch.float64)
        ys, xs = torch.meshgrid(torch.arange(hh_, dtype=torch.float64), torch.arange(ww_, dtype=torch.float64), indexing="ij")
        for _ in range(10):
            _, src = pia.sample_source_corners(hh_, ww_, pia.DELTA_PAPER)
            Hp = pia.homography(dst, src)
            p = torch.stack([xs, ys, torch.ones_like(xs)], -1) @ Hp.T
            disp["PIA"].append(float(torch.sqrt((p[..., 0] / p[..., 2] - xs) ** 2 + (p[..., 1] / p[..., 2] - ys) ** 2).mean()))
    print("     |u| trung bình (px): " + ", ".join(f"{k} {np.mean(v):.1f}" for k, v in disp.items()))
    for name in ("C2-MORGD", "C2-SMOOTH"):
        geo = "morgd" if name == "C2-MORGD" else "smooth"
        for smp_ in samples[:2]:
            info = {}
            seed(8)
            noise = pia.geo_osfd_attack(model, smp_["orig_inputs"].to(DEV), smp_["data_sample"], steps=2,
                                        geo=geo, cls_weight=4.0, info=info)
            k_, t_ = map(sum, zip(*info["gt_kept"]))
            check(bool(torch.isfinite(noise).all()) and noise.abs().max().item() <= 5.0 + 1e-4
                  and info["forward_no_grad"] == 4,
                  f"{name} ảnh {smp_['img_id']}: L∞ {noise.abs().max().item():.1f}, giữ {k_}/{t_}, "
                  f"|u| {np.mean(info['disp']):.1f} px")
    print("TẤT CẢ ĐẠT")


if __name__ == "__main__":
    main()
