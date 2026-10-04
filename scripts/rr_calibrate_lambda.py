#!/usr/bin/env python3
"""Hiệu chỉnh λ theo luật đã dùng cho C2 (progress_log 2026-10-03, khóa lại cho probe V 2026-10-04):
λ = median_{5 ảnh đầu dev100} mean|∇ₓL_OSFD| / mean|∇ₓL_cls| tại δ = 0 (không RRB), ∇ theo ảnh gốc
(qua resize khả vi như core.py), làm tròn số nguyên gần nhất (≥ 1). Seed sampler = SEED + chỉ số dev300.
Kiểm tái tạo: luật áp cho C2 phải ra ~3.95 (3.7 / 4.57 / 3.95 / 7.7 / 2.15 — không tất định nhẹ).
"""
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.methods.repair import cls_term
from attack.models import load_surrogate
from attack.preprocess import extract_features, resize_to_model, to_batch

SEED = 0


def main():
    dev300 = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:5]
    ds = AttackDataset(ids)
    model = load_surrogate("cuda:0")
    out = {}
    for cls in ("c2", "vanish"):
        ratios = []
        for i in range(len(ds)):
            smp = ds[i]
            dsm = smp["data_sample"]
            orig = smp["orig_inputs"].cuda()
            seed = SEED + dev300.index(smp["img_id"])
            with torch.no_grad():
                fc = extract_features(model, resize_to_model(orig, tuple(dsm.img_shape)), dsm)
            x = orig.clone().requires_grad_(True)
            img = resize_to_model(x, tuple(dsm.img_shape))
            feat = sum(F.mse_loss(3.0 * c, a) for c, a in zip(fc, extract_features(model, img, dsm)))
            g_feat = torch.autograd.grad(feat, x)[0]
            x = orig.clone().requires_grad_(True)
            img = resize_to_model(x, tuple(dsm.img_shape))
            random.seed(seed); torch.manual_seed(seed)
            bi, bds = to_batch(model, [img], [dsm])
            g_cls = torch.autograd.grad(cls_term(model, bi, bds, cls), x)[0]
            ratios.append(float(g_feat.abs().mean() / g_cls.abs().mean()))
        med = float(np.median(ratios))
        out[cls] = {"ratios": ratios, "median": med, "lambda": max(1, int(round(med)))}
        print(cls, [round(r, 2) for r in ratios], "median", round(med, 2), "-> λ", out[cls]["lambda"])
    path = os.path.join(REPO_ROOT, "results/repair_robust/lambda_calibration.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"img_ids": ids, **out}, f, indent=2)
    print("saved:", path)


if __name__ == "__main__":
    main()
