#!/usr/bin/env python3
"""Hiệu chỉnh trọng số L_OSFD-C2 (progress_log 2026-10-24, C2-OCT-1V-WP2-C2W*): R = trung vị mean|∇ₓ L_rest| / mean|∇ₓ L_OSFD-C2|
tại δ = 0, ảnh không biến đổi, θ0 (không WP), 5 ảnh đầu dev100 (cùng cách chọn λ = 4 / thang MRG). L_OSFD-C2 = MSE(3·F_sạch, F_adv)
ở stage C2 (stride 4); L_rest = Σ_{C3..C5} MSE + 4·(loss_rpn_cls + loss_cls) (phần còn lại của loss C2). Cùng seed RoI sampling.
w_C2 = a·R với a khóa trước trong progress_log -> gradient số hạng C2 bằng a lần phần còn lại."""
import os
import random
import statistics
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.methods.repair import make_view_loss
from attack.models import load_model
from attack.preprocess import extract_features, resize_to_model

dev = "cuda:0"
model = load_model("r50", dev)
model.roi_head.mask_head = None  # như skip_mask
ds_all = build_attack_dataset("dev300")
keep = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:5]
pos = [ds_all.image_ids.index(i) for i in keep]


def grad(orig, ds, fc, stage_w, weight, seed):
    random.seed(seed); torch.manual_seed(seed)
    n = torch.zeros_like(orig, requires_grad=True)
    l = make_view_loss(fc, 3.0, "c2", weight, stage_w=stage_w)(model, resize_to_model(orig + n, tuple(ds.img_shape)), ds)
    return torch.autograd.grad(l, n)[0]


ratios = []
for j in pos:
    s = ds_all[j]
    orig, ds = s["orig_inputs"].to(dev).float(), s["data_sample"]
    if "masks" in ds.gt_instances:
        del ds.gt_instances.masks
    with torch.no_grad():
        fc = tuple(f.detach() for f in extract_features(model, resize_to_model(orig, tuple(ds.img_shape)), ds))
    g_c2 = grad(orig, ds, fc, (1.0, 0.0, 0.0, 0.0), 0.0, j)
    g_rest = grad(orig, ds, fc, (0.0, 1.0, 1.0, 1.0), 4.0, j)
    r = float(g_rest.abs().mean() / (g_c2.abs().mean() + 1e-20))
    cos = float(torch.nn.functional.cosine_similarity(g_c2.flatten(), g_rest.flatten(), 0))
    ratios.append(r)
    print(f"img {ds_all.image_ids[j]}: |∇rest| {g_rest.abs().mean():.3e}, |∇C2| {g_c2.abs().mean():.3e}, R {r:.1f}, cos {cos:+.3f}")
print(f"R = trung vị {statistics.median(ratios):.1f} (từ {[round(x, 1) for x in ratios]})")
