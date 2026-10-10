#!/usr/bin/env python3
"""Pilot loss "object → nền" (progress_log 2026-10-10): unit test + hiệu chỉnh λ.
(1) L_bg tại δ = 0 = −1, gradient theo ảnh hữu hạn, ≠ 0; ảnh không có nền / object -> 0.
(2) BG-OCT 10 bước trên 1 ảnh: s_R50 (tỉ lệ khoảng cách object→nền ở C4) tăng rõ so ảnh sạch (> NOISE cùng ảnh).
(3) λ (luật C2, làm tròn 1 chữ số có nghĩa như λ_V): median_{5 ảnh đầu dev100} mean|∇L_OSFD| / mean|∇L_bg| tại δ = 0.
  python scripts/bg_unit_test.py   -> results/bg/lambda_calibration.json
"""
import functools
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.methods.pia import _bg_loss, _obj_mask, geo_osfd_attack
from attack.models import load_surrogate
from attack.preprocess import extract_features, resize_to_model

SEED = 0


def gt_boxes(dsm):
    b = dsm.gt_instances.bboxes
    return (b.tensor if hasattr(b, "tensor") else b).float().cuda()


def lbg(model, img, dsm, fc):
    fa = extract_features(model, img, dsm)
    obj, (yy, xx) = _obj_mask(fa, gt_boxes(dsm))
    valid = (yy < img.shape[-2])[:, None] & (xx < img.shape[-1])[None]
    return _bg_loss(fa[2][0].float(), fc[2][0].float(), obj, valid), obj, valid


def s_r50(model, adv_img, dsm, fc):
    """s = mean_obj (Δ_t·u)/‖u‖², u = μ_nền − c_t (như E14c, μ_nền chung cho mọi token object)."""
    with torch.no_grad():
        fa = extract_features(model, adv_img, dsm)[2][0].float()
    c = fc[2][0].float()
    obj, (yy, xx) = _obj_mask((c[None],) * 3, gt_boxes(dsm))
    valid = (yy < adv_img.shape[-2])[:, None] & (xx < adv_img.shape[-1])[None]
    mu = c[:, valid & ~obj].mean(1, keepdim=True)
    u = mu - c[:, valid & obj]
    d = fa[:, valid & obj] - c[:, valid & obj]
    return float(((d * u).sum(0) / (u * u).sum(0)).mean())


def main():
    dev300 = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:5]
    ds = AttackDataset(ids)
    model = load_surrogate("cuda:0")
    ratios, checks = [], []
    for i in range(len(ds)):
        smp = ds[i]
        dsm, orig = smp["data_sample"], smp["orig_inputs"].cuda()
        with torch.no_grad():
            fc = extract_features(model, resize_to_model(orig, tuple(dsm.img_shape)), dsm)
        x = orig.clone().requires_grad_(True)
        img = resize_to_model(x, tuple(dsm.img_shape))
        feat = sum(F.mse_loss(3.0 * c, a) for c, a in zip(fc, extract_features(model, img, dsm)))
        g_feat = torch.autograd.grad(feat, x)[0]
        x = orig.clone().requires_grad_(True)
        img = resize_to_model(x, tuple(dsm.img_shape))
        L, obj, valid = lbg(model, img, dsm, fc)
        g_bg = torch.autograd.grad(L, x)[0]
        checks.append({"img_id": smp["img_id"], "L_bg_at_0": float(L), "n_obj": int((obj & valid).sum()),
                       "n_bg": int((~obj & valid).sum()), "g_bg_finite": bool(torch.isfinite(g_bg).all()),
                       "g_bg_mean_abs": float(g_bg.abs().mean())})
        ratios.append(float(g_feat.abs().mean() / g_bg.abs().mean()))
        assert abs(float(L) + 1.0) < 1e-4 and checks[-1]["g_bg_finite"] and checks[-1]["g_bg_mean_abs"] > 0, checks[-1]
    # ảnh không nền / không object -> 0
    a = torch.randn(8, 4, 4, device="cuda")
    assert float(_bg_loss(a, a, torch.ones(4, 4, dtype=torch.bool, device="cuda"), torch.ones(4, 4, dtype=torch.bool, device="cuda"))) == 0.0
    assert float(_bg_loss(a, a, torch.zeros(4, 4, dtype=torch.bool, device="cuda"), torch.ones(4, 4, dtype=torch.bool, device="cuda"))) == 0.0
    med = float(np.median(ratios))
    lam = float(f"{med:.1g}")
    print("δ=0 checks:", checks)
    print("ratios", [round(r, 3) for r in ratios], "median", round(med, 3), "-> λ", lam)

    # (2) thao tác: BG-OCT 10 bước trên ảnh đầu
    smp = ds[0]
    dsm, orig = smp["data_sample"], smp["orig_inputs"].cuda()
    seed = SEED + dev300.index(smp["img_id"])
    with torch.no_grad():
        fc = extract_features(model, resize_to_model(orig, tuple(dsm.img_shape)), dsm)
    res = {}
    for name, kw in (("BG-OCT", dict(bg_weight=1.0, bg_only=True)), ("OCT", {})):
        random.seed(seed); torch.manual_seed(seed)
        noise = geo_osfd_attack(model, orig, dsm, steps=10, geo="elem", elem={"scale": math.log(2) / 2, "shift": "ph8"}, **kw)
        adv = resize_to_model(torch.clamp(orig + noise, 0, 255), tuple(dsm.img_shape))
        res[name] = s_r50(model, adv, dsm, fc)
    torch.manual_seed(seed)
    nz = torch.clamp(orig + torch.empty_like(orig).uniform_(-5, 5).round(), 0, 255)
    res["NOISE_unif5"] = s_r50(model, resize_to_model(nz, tuple(dsm.img_shape)), dsm, fc)
    print("s_R50 sau 10 bước:", {k: round(v, 3) for k, v in res.items()})
    assert res["BG-OCT"] > res["NOISE_unif5"] + 0.1, res
    path = os.path.join(REPO_ROOT, "results/bg/lambda_calibration.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"img_ids": ids, "ratios": ratios, "median": med, "lambda": lam, "checks": checks, "s_r50_10steps": res}, f, indent=2)
    print("ĐẠT; saved:", path)


if __name__ == "__main__":
    main()
