#!/usr/bin/env python3
"""Unit test pilot GRGD-v0 (progress_log 2026-10-04) — chạy TRƯỚC mọi attack.

1. λ_r = 0 -> view (ảnh, companion, GT) trùng tuyệt đối pia_views.
2. N < 2 (0 và 1 object) -> trùng tuyệt đối pia_views.
3. ξ = 0 -> |u_rel| ≈ 0; mặc định (sửa B) u_rel == u_raw, không trừ phối cảnh.
4. 2 box: t₁ − t₂ = d₁₂ đúng dấu và độ lớn (cả log-scale); dời tương đối hiệu dụng sau warp (sửa B) cùng dấu,
   ≥ 0.8 mục tiêu; báo kèm spec v0 (trừ phối cảnh).
5. Σ ω t = 0, Σ ω log s = 0.
6. Điểm điều khiển sinh từ MỘT homography -> fit lại đúng H, RMS dư ≈ 0.
7. 2 object chuyển động khác nhau -> báo RMS dư (> 0).
8. Xa object -> |u_rel| ≈ 0.
9. σ = 0, adv = sạch -> view == companion.
10. IoU(AABB mask warp, box warp) view GRGD (so PIA 0.823, xoay RRB 0.834).
11. Ô cũ trùng HEAD: GEO-C0, PIA, MORGD (attack) tuyệt đối; morgd_views HEAD == mới; C2-PIA cùng mức lệch HEAD/HEAD.
12. C2-GRGD / C2-PIA-RESIDUAL chạy được (2 step × 2 ảnh): hữu hạn, L∞ đúng, forward no_grad đúng.
Chẩn đoán RNG: 2 ô cùng seed -> mode PIA, nhiễu σ trùng nhau mỗi view (và trùng C2-PIA).
"""
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.methods import grgd, morgd, pia
from attack.methods.box_transforms import get_gt
from attack.models import load_surrogate
from attack.preprocess import resize_to_model
from scripts.pia_unit_test import check, load_head, seed

DEV = "cuda:0"


class Fixed:
    """RNG giả: uniform(a, b) trả lần lượt các ξ cho trước (lặp vòng)."""

    def __init__(self, vals):
        self.vals, self.i = list(vals), 0

    def uniform(self, a, b):
        v = self.vals[self.i % len(self.vals)]
        self.i += 1
        return v


def same_views(a, b):
    for (v1, c1, d1), (v2, c2, d2) in zip(a, b):
        if not (torch.equal(v1, v2) and torch.equal(c1, c2)):
            return False
        g1, g2 = d1.gt_instances, d2.gt_instances
        if not (torch.equal(g1.bboxes, g2.bboxes) and torch.equal(g1.labels, g2.labels)
                and np.array_equal(g1.masks.masks, g2.masks.masks)):
            return False
    return len(a) == len(b)


def main():
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))[:4]
    ds = AttackDataset(ids)
    model = load_surrogate(DEV)
    samples = [ds[i] for i in range(len(ds))]
    for s_ in samples:
        print(f"     ảnh {s_['img_id']}: {len(s_['data_sample'].gt_instances)} GT")
    smp = max(samples, key=lambda s_: len(s_["data_sample"].gt_instances))
    orig, dsm = smp["orig_inputs"].to(DEV), smp["data_sample"]
    seed(0); pia.geo_osfd_attack(model, orig, dsm, steps=1)  # khởi động GPU
    img = resize_to_model(orig, tuple(dsm.img_shape))
    h, w = img.shape[-2:]
    adv = (img + torch.randn_like(img) * 3).clamp(0, 255)

    # 1. λ_r = 0
    seed(5); a = pia.pia_views(adv, 6, companion=img, data_sample=dsm)
    seed(5); b = grgd.grgd_views(adv, 6, dsm, random.Random(1), lam_r=0, companion=img, with_gt=True)
    check(same_views(a, b), "λ_r = 0 -> trùng tuyệt đối pia_views (ảnh, companion, GT)")

    # 2. N < 2
    for k in (0, 1):
        d1 = dsm.clone()
        d1.gt_instances = dsm.gt_instances[:k]
        seed(6); a = pia.pia_views(adv, 4, companion=img, data_sample=d1)
        seed(6); b = grgd.grgd_views(adv, 4, d1, random.Random(1), companion=img, with_gt=True)
        check(same_views(a, b), f"N = {k} -> trùng tuyệt đối pia_views")

    boxes, _ = get_gt(dsm, DEV)
    ys, xs = morgd.pixel_coords(h, w, DEV, torch.float64)

    # 3. ξ = 0
    c, sz, t, s, e, _ = grgd.relative_motion(boxes, h, w, "object", Fixed([0.0]))
    u = grgd.RelativeField(c, sz, t, s).at(xs, ys)["rel"]
    mx = torch.sqrt(u[0] ** 2 + u[1] ** 2).max().item()
    check(mx < 1e-6, f"ξ = 0 -> max |u_rel| = {mx:.1e} px")
    c, sz, t, s, e, _ = grgd.relative_motion(boxes, h, w, "object", random.Random(5))
    dd = grgd.RelativeField(c, sz, t, s).at(xs, ys)
    check(all(torch.equal(x, y) for x, y in zip(dd["rel"], dd["raw"])), "mặc định: u_rel == u_raw (không trừ phối cảnh)")

    # 4. 2 box
    two = torch.tensor([[200.0, 200.0, 320.0, 280.0], [600.0, 350.0, 700.0, 500.0]], device=DEV)
    xi = (0.8, -0.5, 0.6)
    c, sz, t, s, e, _ = grgd.relative_motion(two, h, w, "object", Fixed(xi))
    dx = grgd.LAMBDA_R * (120 + 100) / 2 * xi[0]
    dy = grgd.LAMBDA_R * (80 + 150) / 2 * xi[1]
    dsc = np.log(grgd.S_MAX) * xi[2]
    got = (t[0] - t[1]).tolist() + [float(torch.log(s[0] / s[1]))]
    check(np.allclose(got, [dx, dy, dsc], atol=1e-4), f"2 box: t₁ − t₂, log s₁/s₂ = {np.round(got, 3).tolist()} "
          f"vs d = {[round(dx, 3), round(dy, 3), round(dsc, 3)]}")
    f2 = grgd.RelativeField(c, sz, t, s)
    eff = {}
    for name, fld in (("sửa B (u_rel = u_raw)", f2),
                      ("spec v0 (trừ phối cảnh)", grgd.RelativeField(c, sz, t, s, remove_projective=True))):
        g2 = morgd.to_grid(xs, ys, *fld.at(xs, ys)["rel"], h, w).float()
        nb, keep = morgd.warp_boxes_by_grid(two, g2, h, w)
        cen = lambda bb: torch.stack([(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2], 1)
        rel_change = ((cen(nb)[0] - cen(nb)[1]) - (cen(two)[0] - cen(two)[1])).tolist()
        eff[name] = rel_change
        print(f"     2 box, dời tương đối hiệu dụng sau warp [{name}]: {np.round(rel_change, 1).tolist()} "
              f"(mục tiêu {[round(dx, 1), round(dy, 1)]})")
    rb = eff["sửa B (u_rel = u_raw)"]
    check(rb[0] / dx >= 0.8 and rb[1] / dy >= 0.8, f"2 box sửa B: dời hiệu dụng / mục tiêu = {rb[0] / dx:.2f}, {rb[1] / dy:.2f}")
    print(f"     2 box: RMS dư fit H_r tại điểm điều khiển {f2.fit_rms:.2f} px / RMS u_raw {f2.raw_rms_ctrl:.2f} px")

    # 5. Σ ω t = 0
    seed(3)
    sizes = [(random.uniform(10, 200), random.uniform(10, 200)) for _ in range(6)]
    edges = grgd.complete_edges(6)
    t6, s6, om = grgd.solve_graph_motion(sizes, edges, *grgd.sample_pairwise(sizes, edges, random.Random(4)))
    res = [float(om @ t6[:, 0]), float(om @ t6[:, 1]), float(om @ torch.log(s6))]
    check(max(map(abs, res)) < 1e-8, f"Σ ω t = 0, Σ ω log s = 0: {[f'{r:.1e}' for r in res]}")

    # 6. homography -> fit lại đúng
    H0 = torch.tensor([[1.03, 0.02, 5.0], [-0.01, 0.98, -7.0], [2e-5, -1e-5, 1.0]], dtype=torch.float64)
    q = grgd.control_points(torch.tensor([[150.0, 120.0], [500.0, 400.0], [900.0, 200.0]], dtype=torch.float64),
                            [(80, 60), (120, 90), (50, 70)])
    px, py, _ = grgd._apply_h(H0, q[:, 0], q[:, 1])
    Hf = grgd.fit_homography_dlt(q, torch.stack([px, py], 1))
    fx, fy, _ = grgd._apply_h(Hf, q[:, 0], q[:, 1])
    rms = torch.sqrt(((fx - px) ** 2 + (fy - py) ** 2).mean()).item()
    check(rms < 1e-6 and (Hf - H0).abs().max().item() < 1e-6, f"1 homography -> fit lại đúng (RMS dư {rms:.1e} px)")

    # 7. 2 object khác chuyển động -> RMS dư > 0 (cùng ca test 4) + nhiều object thật
    check(f2.fit_rms > 0.01, f"2 object khác chuyển động: RMS dư {f2.fit_rms:.2f} px > 0")
    seed(7)
    c, sz, t, s, e, _ = grgd.relative_motion(boxes, h, w, "object", random.Random(11))
    fN = grgd.RelativeField(c, sz, t, s)
    print(f"     ảnh {smp['img_id']} ({len(boxes)} object): RMS dư {fN.fit_rms:.2f} px / RMS u_raw {fN.raw_rms_ctrl:.2f} px")

    # 8. xa object
    small = torch.tensor([[60.0, 60.0, 110.0, 110.0], [140.0, 80.0, 190.0, 120.0]], device=DEV)
    c, sz, t, s, e, _ = grgd.relative_motion(small, h, w, "object", Fixed((1.0, -1.0, 1.0)))
    u = grgd.RelativeField(c, sz, t, s).at(xs, ys)["rel"]
    far = torch.sqrt(u[0] ** 2 + u[1] ** 2)[:, 600:].max().item()
    check(far < 0.05, f"xa object (x ≥ 600, object ở x ≤ 190, cỡ ~50): max |u_rel| = {far:.1e} px")

    # 9. σ = 0, adv = sạch
    seed(8)
    vv = grgd.grgd_views(img, 4, dsm, random.Random(2), sigma=0, companion=img, with_gt=True)
    check(all(torch.equal(v, c_) for v, c_, _ in vv), "σ = 0, adv = sạch -> view == companion")

    # 10. IoU(mask, box) + thống kê view
    ious, st, kept = [], [], []
    for smp_ in samples:
        d = smp_["data_sample"]
        im = resize_to_model(smp_["orig_inputs"].to(DEV), tuple(d.img_shape))
        seed(9)
        for v, _, dsv in grgd.grgd_views(im, 10, d, random.Random(3), with_gt=True, stats=st, gt_log=kept):
            g = dsv.gt_instances
            for bb, mm in zip(g.bboxes, g.masks.masks):
                yy, xx = np.nonzero(mm)
                if len(xx) == 0:
                    continue
                mb = torch.tensor([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1], dtype=torch.float32, device=DEV)
                lt, rb = torch.max(bb[:2], mb[:2]), torch.min(bb[2:], mb[2:])
                inter = (rb - lt).clamp_min(0).prod()
                ious.append(float(inter / ((bb[2:] - bb[:2]).prod() + (mb[2:] - mb[:2]).prod() - inter)))
    check(np.mean(ious) > 0.7, f"IoU(AABB mask warp, box warp) mean {np.mean(ious):.3f}, trung vị {np.median(ious):.3f}, "
          f"p5 {np.percentile(ious, 5):.3f} (PIA 0.823, xoay RRB 0.834)")
    rel = [x for x in st if x["n_edges"] > 0]
    print(f"     {len(rel)}/{len(st)} view có T_relative; giữ GT {sum(a for a, _ in kept)}/{sum(b for _, b in kept)}")
    for k in ("disp_pia_px", "disp_raw_px", "disp_proj_px", "disp_rel_px", "rms_rel_px", "r_proj", "d_pair_pos_req",
              "d_pair_pos", "d_pair_pos_eff", "rho_pos", "cos_pair_eff_req", "d_pair_pos_box", "d_pair_scale_req",
              "d_pair_scale", "d_pair_scale_box", "fit_rms_ctrl_px", "den_h_min"):
        vals = [x[k] for x in (st if k == "disp_pia_px" else rel) if k in x]
        print(f"     {k}: mean {np.mean(vals):.3f}, min {np.min(vals):.3f}, max {np.max(vals):.3f}")
    check(all(x["den_h_min"] > 0 for x in rel), "mẫu số H_r > 0 trên toàn khung ở mọi view")

    # 11. ô cũ trùng HEAD
    old = load_head("attack/methods/pia.py", "pia_head")
    old_m = load_head("attack/methods/morgd.py", "morgd_head")
    for name, kw in {"GEO-C0": dict(geo="rrb"), "PIA": dict(geo="pia"), "MORGD": dict(geo="morgd")}.items():
        out = []
        for mod in (old, pia):
            seed(9); out.append(mod.geo_osfd_attack(model, orig, dsm, steps=3, **kw))
        check(torch.equal(out[0], out[1]), f"{name} trùng HEAD")
    a, b = [], []
    for mod, lst in ((old_m, a), (morgd, b)):
        seed(10); lst += mod.morgd_views(adv, 4, dsm, anchors="object", companion=img, with_gt=True)
    check(same_views(a, b), "morgd_views HEAD == mới (refactor flow_field)")
    out = []
    for mod in (old, old, pia):
        seed(9); out.append(mod.geo_osfd_attack(model, orig, dsm, steps=3, cls_weight=4.0))
    hh, hn = (out[0] != out[1]).float().mean().item(), (out[0] != out[2]).float().mean().item()
    check(hn <= max(2 * hh, 0.10), f"C2-PIA: lệch HEAD/HEAD {hh:.3f}, HEAD/mới {hn:.3f}")

    # Chẩn đoán RNG giữa các ô: mode PIA + nhiễu σ mỗi view
    rec = {}
    orig_blur = {m: m.gaussian_blur for m in (grgd, pia)}

    def recorder(lst):
        def gb(x, sigma=6.0):
            n = torch.randn_like(x)
            lst.append(n.flatten()[:16].clone())
            return torch.clamp(x + n * sigma, 0.0, 255.0)
        return gb

    for name, kw in (("C2-PIA", dict(geo="pia")), ("C2-GRGD", dict(geo="grgd")),
                     ("C2-PIA-RESIDUAL", dict(geo="grgd_random"))):
        noise_log, info = [], {}
        grgd.gaussian_blur = pia.gaussian_blur = recorder(noise_log)
        seed(12); pia.geo_osfd_attack(model, orig, dsm, steps=4, cls_weight=4.0, info=info, **kw)
        rec[name] = (info["modes"], noise_log)
    grgd.gaussian_blur, pia.gaussian_blur = orig_blur[grgd], orig_blur[pia]
    for name in ("C2-GRGD", "C2-PIA-RESIDUAL"):
        same_mode = rec[name][0] == rec["C2-PIA"][0]
        same_noise = all(torch.equal(x, y) for x, y in zip(rec[name][1], rec["C2-PIA"][1]))
        check(same_mode and same_noise, f"{name} vs C2-PIA (4 step × 2 view): mode PIA trùng {same_mode}, nhiễu σ trùng {same_noise}")

    # 12. chạy
    for name, geo in (("C2-GRGD", "grgd"), ("C2-PIA-RESIDUAL", "grgd_random")):
        for smp_ in samples[:2]:
            info = {}
            seed(8)
            noise = pia.geo_osfd_attack(model, smp_["orig_inputs"].to(DEV), smp_["data_sample"], steps=2,
                                        geo=geo, cls_weight=4.0, info=info)
            k_, t_ = map(sum, zip(*info["gt_kept"]))
            r = [x for x in info["grgd"] if x["n_edges"] > 0]
            check(bool(torch.isfinite(noise).all()) and noise.abs().max().item() <= 5.0 + 1e-4
                  and info["forward_no_grad"] == 4,
                  f"{name} ảnh {smp_['img_id']}: L∞ {noise.abs().max().item():.1f}, giữ {k_}/{t_}, "
                  f"|u_rel| {np.mean([x['disp_rel_px'] for x in r]) if r else 0:.1f} px, "
                  f"R_proj {np.mean([x['r_proj'] for x in r]) if r else float('nan'):.2f}")
    print("TẤT CẢ ĐẠT")


if __name__ == "__main__":
    main()
