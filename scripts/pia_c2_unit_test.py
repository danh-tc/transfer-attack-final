#!/usr/bin/env python3
"""Unit test C2-PIA + PIA-noCT (progress_log 2026-10-04, pilot PIA bổ sung) — chạy TRƯỚC mọi attack.

1. Ô GEO-C0 / PIA / PIA-d0.3 / PIA-aff trùng tuyệt đối code HEAD (cờ mặc định, 3 step).
2. warp_boxes: tịnh tiến nguyên (dx, dy) -> box dời (−dx, −dy); đẩy ra ngoài khung -> bỏ hết.
3. IoU(AABB mask đã warp, box đã warp) trên view PIA — so mức không biến đổi và mức xoay RRB (0.834).
4. PIA-noCT: 0 forward no_grad; loss view == make_osfd_loss_fn (feature sạch chưa biến đổi).
5. make_view_loss(c2, weight 0) == _view_loss trên view PIA (hook feature đúng); GT rỗng -> loss hữu
   hạn; C2-PIA chạy 3 step trên 2 ảnh, noise hữu hạn, L∞ đúng; báo tỉ lệ instance bị bỏ.
"""
import functools
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.losses.osfd import make_osfd_loss_fn
from attack.methods import pia
from attack.methods.box_transforms import get_gt
from attack.methods.repair import make_view_loss
from attack.models import load_surrogate
from attack.preprocess import extract_features, resize_to_model
from scripts.pia_unit_test import check, load_head, seed

DEV = "cuda:0"


def box_iou(a, b):
    lt, rb = torch.max(a[:, :2], b[:, :2]), torch.min(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(1)
    area = lambda x: (x[:, 2:] - x[:, :2]).clamp(min=0).prod(1)
    return inter / (area(a) + area(b) - inter)


def mask_aabb(m):
    out = []
    for mi in m:
        ys, xs = torch.nonzero(mi > 0.5, as_tuple=True)
        out.append(torch.stack([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]).float() if len(xs)
                   else torch.full((4,), float("nan"), device=m.device))
    return torch.stack(out)


def main():
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))[:3]
    ds = AttackDataset(ids)
    model = load_surrogate(DEV)
    samples = [ds[i] for i in range(len(ds))]
    smp = samples[0]
    orig, dsm = smp["orig_inputs"].to(DEV), smp["data_sample"]
    seed(0); pia.geo_osfd_attack(model, orig, dsm, steps=1)  # khởi động GPU (không tất định lần đầu)

    # 1. ô cũ trùng HEAD
    old = load_head("attack/methods/pia.py", "pia_head")
    cells = {"GEO-C0": dict(geo="rrb"), "PIA": dict(geo="pia"), "PIA-d0.3": dict(geo="pia", delta_max=0.3),
             "PIA-aff": dict(geo="pia", pi=(0.0, 1.0, 0.0, 0.0))}
    for name, kw in cells.items():
        out = []
        for mod in (old, pia):
            seed(9)
            out.append(mod.geo_osfd_attack(model, orig, dsm, steps=3, **kw))
        check(torch.equal(out[0], out[1]), f"{name} trùng HEAD")

    # 2. warp_boxes tịnh tiến + ra ngoài khung
    h, w = 800, 1199
    dst = torch.tensor([(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)], dtype=torch.float64)
    boxes = torch.tensor([[100.0, 200.0, 300.0, 400.0], [500.0, 50.0, 700.0, 150.0]])
    dx, dy = 13.0, -7.0
    H = pia.homography(dst, dst + torch.tensor([dx, dy], dtype=torch.float64))
    nb, keep = pia.warp_boxes(boxes, H, h, w)
    check(keep.all().item() and torch.allclose(nb, boxes - torch.tensor([dx, dy, dx, dy]), atol=1e-4),
          "warp_boxes tịnh tiến đúng (box dời −dx, −dy)")
    H = pia.homography(dst, dst + torch.tensor([-(w + 10.0), 0.0], dtype=torch.float64))
    nb, keep = pia.warp_boxes(boxes, H, h, w)
    check(len(nb) == 0 and not keep.any().item(), "box bị đẩy ra ngoài khung -> bỏ")

    # 3. IoU mask/box trên view PIA
    ious0, ious, kept, total = [], [], 0, 0
    for s in samples:
        d = s["data_sample"]
        img = resize_to_model(s["orig_inputs"].to(DEV), tuple(d.img_shape))
        b0, m0 = get_gt(d, DEV)
        ious0 += box_iou(mask_aabb(m0), b0).tolist()
        seed(4)
        for _, _, dsv in pia.pia_views(img, 20, data_sample=d):
            bv, mv = get_gt(dsv, DEV)
            kept += len(bv); total += len(b0)
            if len(bv):
                ious += box_iou(mask_aabb(mv), bv).tolist()
    ious0 = [x for x in ious0 if x == x]
    ious = [x for x in ious if x == x]
    check(np.mean(ious) > 0.7, f"IoU(mask, box) không biến đổi {np.mean(ious0):.3f}; view PIA mean "
          f"{np.mean(ious):.3f}, trung vị {np.median(ious):.3f}, p5 {np.percentile(ious, 5):.3f} (xoay RRB 0.834); "
          f"giữ {kept}/{total} instance")

    # 4. PIA-noCT
    clean = resize_to_model(orig, tuple(dsm.img_shape))
    info = {}
    seed(5)
    noise = pia.geo_osfd_attack(model, orig, dsm, steps=2, co_transform=False, info=info)
    check(info["forward_no_grad"] == 0 and noise.abs().max().item() <= 5.0 + 1e-4, "PIA-noCT: 0 forward no_grad, L∞ đúng")
    with torch.no_grad():
        fc0 = tuple(f.detach() for f in extract_features(model, clean, dsm))
        seed(6)
        v = pia.pia_views(clean + 2.0, 1)[0][0]
        a = pia._view_loss(fc0, 3.0)(model, v, dsm).item()
        b = make_osfd_loss_fn(model, clean, dsm)(model, v, dsm).item()
    check(abs(a - b) <= 1e-6 * abs(b), f"PIA-noCT loss == loss OSFD ({a:.6g} vs {b:.6g})")

    # 5. C2-PIA
    seed(7)
    v, c, dsv = pia.pia_views(clean + 2.0, 1, companion=clean, data_sample=dsm)[0]
    with torch.no_grad():
        fc = tuple(f.detach() for f in extract_features(model, c, dsm))
    torch.manual_seed(0); a = make_view_loss(fc, 3.0, "c2", 0.0)(model, v, dsv).item()
    b = pia._view_loss(fc, 3.0)(model, v, dsv).item()
    check(abs(a - b) <= 1e-5 * abs(b), f"make_view_loss(c2, 0) == _view_loss ({a:.6g} vs {b:.6g})")
    empty = pia.gt_with_keep(dsm, torch.zeros(0, 4, device=DEV), get_gt(dsm, DEV)[1] * 0,
                             torch.zeros(len(dsm.gt_instances), dtype=torch.bool, device=DEV))
    le = make_view_loss(fc, 3.0, "c2", 4.0)(model, v, empty)
    check(bool(torch.isfinite(le)), f"GT rỗng -> loss hữu hạn ({le.item():.4g})")
    for s in samples[:2]:
        info = {}
        seed(8)
        noise = pia.geo_osfd_attack(model, s["orig_inputs"].to(DEV), s["data_sample"], steps=3,
                                    cls_weight=4.0, info=info)
        k_, t_ = map(sum, zip(*info["gt_kept"]))
        check(bool(torch.isfinite(noise).all()) and noise.abs().max().item() <= 5.0 + 1e-4
              and info["forward_no_grad"] == 6,
              f"C2-PIA ảnh {s['img_id']}: hữu hạn, L∞ {noise.abs().max().item():.1f}, giữ {k_}/{t_} instance")
    print("TẤT CẢ ĐẠT")


if __name__ == "__main__":
    main()
