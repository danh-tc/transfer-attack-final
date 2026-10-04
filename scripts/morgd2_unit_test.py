#!/usr/bin/env python3
"""Unit test MORGD-v2 (progress_log 2026-10-04 "Cải tiến MORGD") — chạy TRƯỚC cổng độ trung thực.

1. λ = 0 và không có GT -> view (ảnh, companion, GT) trùng tuyệt đối pia_views.
2. 2 object tách xa (không chồng lấn) -> mỗi box dời đúng c' ± s·(w, h)/2 (≤ 2 px).
3. Xa mọi vùng -> |u| ≈ 0.
4. σ = 0, adv = sạch -> view == companion (cả 3 ô).
5. Ô cũ trùng HEAD: GEO-C0, PIA, MORGD tuyệt đối; C2-PIA cùng mức lệch HEAD/HEAD.
6. RNG: 3 ô mới có mode PIA + nhiễu σ trùng C2-PIA mỗi view (ô không PIA: nhiễu trùng nhau giữa 2 lần chạy).
7. C2-PIA-MORGD2 / C2-PIA-SMOOTH2 / C2-MORGD2 chạy được (2 step × 2 ảnh): hữu hạn, L∞ đúng, forward no_grad đúng.
"""
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.methods import grgd, morgd, morgd2, pia
from attack.models import load_surrogate
from attack.preprocess import resize_to_model
from scripts.grgd_unit_test import same_views
from scripts.pia_unit_test import check, load_head, seed

DEV = "cuda:0"


def main():
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))[:4]
    ds = AttackDataset(ids)
    model = load_surrogate(DEV)
    samples = [ds[i] for i in range(len(ds))]
    smp = max(samples, key=lambda s_: len(s_["data_sample"].gt_instances))
    orig, dsm = smp["orig_inputs"].to(DEV), smp["data_sample"]
    seed(0); pia.geo_osfd_attack(model, orig, dsm, steps=1)  # khởi động GPU
    img = resize_to_model(orig, tuple(dsm.img_shape))
    h, w = img.shape[-2:]
    adv = (img + torch.randn_like(img) * 3).clamp(0, 255)

    # 1
    seed(5); a = pia.pia_views(adv, 6, companion=img, data_sample=dsm)
    seed(5); b = morgd2.morgd2_views(adv, 6, dsm, random.Random(1), lam=0, companion=img, with_gt=True)
    check(same_views(a, b), "λ = 0 -> trùng tuyệt đối pia_views")
    d0 = dsm.clone(); d0.gt_instances = dsm.gt_instances[:0]
    seed(6); a = pia.pia_views(adv, 4, companion=img, data_sample=d0)
    seed(6); b = morgd2.morgd2_views(adv, 4, d0, random.Random(1), companion=img, with_gt=True)
    check(same_views(a, b), "không có GT -> trùng tuyệt đối pia_views")

    # 2
    ys, xs = morgd.pixel_coords(h, w, DEV, torch.float64)
    two = torch.tensor([[100.0, 100.0, 220.0, 180.0], [700.0, 450.0, 800.0, 600.0]], device=DEV)
    c, sz, t, s = morgd2.sample_motion(two, h, w, "object", random.Random(7))
    ux, uy = morgd2.field(c, sz, t, s, xs, ys)
    g = morgd.to_grid(xs, ys, ux, uy, h, w).float()
    nb, keep = morgd.warp_boxes_by_grid(two, g, h, w)
    exp = torch.stack([c[:, 0] + t[:, 0] - s * torch.tensor([x[0] for x in sz], dtype=torch.float64) / 2,
                       c[:, 1] + t[:, 1] - s * torch.tensor([x[1] for x in sz], dtype=torch.float64) / 2,
                       c[:, 0] + t[:, 0] + s * torch.tensor([x[0] for x in sz], dtype=torch.float64) / 2,
                       c[:, 1] + t[:, 1] + s * torch.tensor([x[1] for x in sz], dtype=torch.float64) / 2], 1).float().to(DEV)
    err = (nb - exp).abs().max().item()
    check(keep.all().item() and err <= 2.0, f"2 object tách xa: box dời đúng (lệch tối đa {err:.2f} px; t = "
          f"{np.round(t.numpy(), 1).tolist()}, s = {np.round(s.numpy(), 3).tolist()})")

    # 3
    far = torch.sqrt(ux ** 2 + uy ** 2)[:, 350:600].max().item()
    check(far < 1e-3, f"xa mọi vùng (x ∈ [350, 600)): max |u| = {far:.1e} px")

    # 4
    for anc, use_pia in (("object", True), ("random", True), ("object", False)):
        seed(8)
        vv = morgd2.morgd2_views(img, 4, dsm, random.Random(2), random.Random(3), anchors=anc, use_pia=use_pia,
                                 sigma=0, companion=img, with_gt=True)
        check(all(torch.equal(v, c_) for v, c_, _ in vv), f"σ = 0, adv = sạch -> view == companion ({anc}, PIA {use_pia})")

    # 5
    old = load_head("attack/methods/pia.py", "pia_head")
    for name, kw in {"GEO-C0": dict(geo="rrb"), "PIA": dict(geo="pia"), "MORGD": dict(geo="morgd")}.items():
        out = []
        for mod in (old, pia):
            seed(9); out.append(mod.geo_osfd_attack(model, orig, dsm, steps=3, **kw))
        check(torch.equal(out[0], out[1]), f"{name} trùng HEAD")
    out = []
    for mod in (old, old, pia):
        seed(9); out.append(mod.geo_osfd_attack(model, orig, dsm, steps=3, cls_weight=4.0))
    hh, hn = (out[0] != out[1]).float().mean().item(), (out[0] != out[2]).float().mean().item()
    check(hn <= max(2 * hh, 0.10), f"C2-PIA: lệch HEAD/HEAD {hh:.3f}, HEAD/mới {hn:.3f}")

    # 6
    rec = {}
    blur0 = {m: m.gaussian_blur for m in (morgd2, pia)}

    def recorder(lst):
        def gb(x, sigma=6.0):
            n_ = torch.randn_like(x)
            lst.append(n_.flatten()[:16].clone())
            return torch.clamp(x + n_ * sigma, 0.0, 255.0)
        return gb

    for name, geo in (("C2-PIA", "pia"), ("C2-PIA-MORGD2", "pia_morgd2"), ("C2-PIA-SMOOTH2", "pia_smooth2"),
                      ("C2-MORGD2", "morgd2"), ("C2-MORGD2#2", "morgd2")):
        nl, info = [], {}
        morgd2.gaussian_blur = pia.gaussian_blur = recorder(nl)
        seed(12); pia.geo_osfd_attack(model, orig, dsm, steps=4, cls_weight=4.0, info=info, geo=geo)
        rec[name] = (info["modes"], nl)
    morgd2.gaussian_blur, pia.gaussian_blur = blur0[morgd2], blur0[pia]
    for name in ("C2-PIA-MORGD2", "C2-PIA-SMOOTH2"):
        sm = rec[name][0] == rec["C2-PIA"][0]
        sn = all(torch.equal(x, y) for x, y in zip(rec[name][1], rec["C2-PIA"][1]))
        check(sm and sn, f"{name} vs C2-PIA: mode PIA trùng {sm}, nhiễu σ trùng {sn}")
    check(all(torch.equal(x, y) for x, y in zip(rec["C2-MORGD2"][1], rec["C2-MORGD2#2"][1])), "C2-MORGD2: nhiễu tất định giữa 2 lần chạy")

    # 7
    for name, geo in (("C2-PIA-MORGD2", "pia_morgd2"), ("C2-PIA-SMOOTH2", "pia_smooth2"), ("C2-MORGD2", "morgd2")):
        for smp_ in samples[:2]:
            info = {}
            seed(8)
            noise = pia.geo_osfd_attack(model, smp_["orig_inputs"].to(DEV), smp_["data_sample"], steps=2,
                                        geo=geo, cls_weight=4.0, info=info)
            k_, t_ = map(sum, zip(*info["gt_kept"]))
            mv = [x for x in info["morgd2"] if x["moved"]]
            fid = [r for x in mv for r, _ in x.get("fid", [])]
            check(bool(torch.isfinite(noise).all()) and noise.abs().max().item() <= 5.0 + 1e-4
                  and info["forward_no_grad"] == 4,
                  f"{name} ảnh {smp_['img_id']}: L∞ {noise.abs().max().item():.1f}, giữ {k_}/{t_}, "
                  f"RMS |u| {np.mean([x['rms_u_px'] for x in mv]) if mv else 0:.1f} px, "
                  f"ratio_pos median {np.median(fid) if fid else float('nan'):.2f}")
    print("TẤT CẢ ĐẠT")


if __name__ == "__main__":
    main()
