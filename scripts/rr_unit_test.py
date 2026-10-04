#!/usr/bin/env python3
"""Unit test Repair-Robust (progress_log 2026-10-04) — chạy TRƯỚC mọi attack.

1. rrb_views(companion=None) trùng tuyệt đối bản rrb.py ở HEAD (3 ảnh × 3 seed, cả with_gt).
2. companion: không đổi RNG (view giống hệt khi có/không companion); tắt nhiễu thì view == companion
   tại δ = 0.
3. make_view_loss(cls="c2", không sửa) == preprocess.make_feature_plus_gt_loss (cùng seed sampler).
4. vanish_loss: hữu hạn, ≤ 0, có anchor/RoI dương với ảnh có GT; ảnh không GT -> 0.
5. R-a tại δ = 0, σ = 0: loss có sửa == loss không sửa (C2/C3 sạch == C2/C3 adv).
6. repair_robust_attack chạy được (p = 0 / 0.5 / 1, 3 step): L∞ ≤ ε, đếm forward no_grad đúng.
"""
import importlib.util
import os
import random
import subprocess
import sys
import tempfile

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.methods import rrb
from attack.methods.repair import make_view_loss, repair_robust_attack, vanish_loss
from attack.models import load_surrogate
from attack.preprocess import extract_features, make_feature_plus_gt_loss, resize_to_model, to_batch

DEV = "cuda:0"


def load_head_rrb():
    src = subprocess.check_output(["git", "show", "HEAD:attack/methods/rrb.py"], cwd=REPO_ROOT)
    path = os.path.join(tempfile.mkdtemp(), "rrb_head.py")
    with open(path, "wb") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("rrb_head", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def seed(s):
    random.seed(s); torch.manual_seed(s)


def main():
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:3]
    ds = AttackDataset(ids)
    model = load_surrogate(DEV)
    old = load_head_rrb()
    samples = [ds[i] for i in range(len(ds))]

    # 1 + 2
    for smp in samples:
        img = resize_to_model(smp["orig_inputs"].to(DEV), tuple(smp["data_sample"].img_shape))
        dsm = smp["data_sample"]
        for s in range(3):
            seed(s); a = old.rrb_views(img, dsm)
            seed(s); b = rrb.rrb_views(img, dsm)
            assert all(torch.equal(x, y) for x, y in zip(a, b)), "rrb_views lệch HEAD"
            seed(s); a = old.rrb_views(img, dsm, with_gt=True)
            seed(s); b = rrb.rrb_views(img, dsm, with_gt=True)
            seed(s); c = rrb.rrb_views(img, dsm, with_gt=True, companion=img.clone())
            for x, y, z in zip(a, b, c):
                assert torch.equal(x[0], y[0]) and torch.equal(x[0], z[0]), "with_gt/companion đổi RNG"
            seed(s); d = rrb.rrb_views(img, dsm, with_gt=True, companion=img.clone(), use_noise=False)
            assert all(torch.equal(v[0], v[2]) for v in d), "companion != view khi δ=0, không nhiễu"
    print("[1-2] rrb_views: trùng HEAD; companion không đổi RNG, khớp hình học ✓")

    # 3 + 4 + 5
    for smp in samples:
        dsm = smp["data_sample"]
        clean = resize_to_model(smp["orig_inputs"].to(DEV), tuple(dsm.img_shape))
        with torch.no_grad():
            fc = tuple(f.detach() for f in extract_features(model, clean, dsm))
            seed(7); ref = make_feature_plus_gt_loss(fc, 3.0, ("loss_rpn_cls", "loss_cls"), 4.0)(model, clean, dsm)
            seed(7); new = make_view_loss(fc, 3.0, "c2", 4.0)(model, clean, dsm)
            assert torch.allclose(ref, new, rtol=1e-6, atol=0), f"C2 lệch: {ref} vs {new}"
            bi, bds = to_batch(model, [clean], [dsm])
            lv, (n_rpn, n_roi) = vanish_loss(model, model.extract_feat(bi), bds, return_counts=True)
            assert torch.isfinite(lv) and lv <= 0 and n_rpn > 0 and n_roi > 0, (lv, n_rpn, n_roi)
            # 5: σ = 0, δ = 0 -> view == companion -> sửa không đổi gì
            seed(3); views = rrb.rrb_views(clean, dsm, with_gt=True, companion=clean, sigma=0.0)
            for v, dsv, cv in views:
                rep = tuple(f.detach() for f in extract_features(model, cv, dsv)[:2])
                for loss_cls in ("c2", "vanish"):
                    seed(9); a = make_view_loss(fc, 3.0, loss_cls, 4.0)(model, v, dsv)
                    seed(9); b = make_view_loss(fc, 3.0, loss_cls, 4.0, rep)(model, v, dsv)
                    assert torch.allclose(a, b, rtol=1e-5), f"R-a δ=0 lệch ({loss_cls}): {a} vs {b}"
        print(f"[3-5] img {smp['img_id']}: C2 trùng ({float(ref):.4f}); L_V {float(lv):.3f} "
              f"(RPN dương {n_rpn}, RoI dương {n_roi}); R-a δ=0 trùng ✓")
    # ảnh không GT -> L_V = 0
    dsm = samples[0]["data_sample"].clone()
    dsm.gt_instances = dsm.gt_instances[:0]
    with torch.no_grad():
        bi, bds = to_batch(model, [samples[0]["inputs"].to(DEV)], [dsm])
        assert float(vanish_loss(model, model.extract_feat(bi), bds)) == 0.0
    print("[4] ảnh không GT -> L_V = 0 ✓")

    # 6
    smp = samples[0]
    for p in (0.0, 0.5, 1.0):
        info = {}
        seed(0)
        noise = repair_robust_attack(model, smp["orig_inputs"].to(DEV), smp["data_sample"], steps=3,
                                     cls="vanish", cls_weight=1.0, repair_p=p, repair_seed=0, info=info)
        n_on = sum(info["repair_on"])
        assert float(noise.abs().max()) <= 5.0 + 1e-6 and info["forward_no_grad"] == 2 * n_on
        print(f"[6] p={p}: L∞ {float(noise.abs().max()):.1f}, step sửa {n_on}/3, forward no_grad "
              f"{info['forward_no_grad']} ✓")
    print("TẤT CẢ ĐẠT")


if __name__ == "__main__":
    main()
