#!/usr/bin/env python3
"""Hiệu chỉnh thang margin (progress_log 2026-10-20, C2-OCT-1V-MRG / -MRGF): scale = trung vị mean|∇ₓ CE_RoI| / mean|∇ₓ L_mrg|
tại δ = 0, ảnh không biến đổi, 5 ảnh đầu dev100 (cùng cách chọn λ = 4, progress_log 2026-10-03) -> số hạng margin có gradient
cỡ CE gốc, giữ nguyên 4·(loss_rpn_cls + ·). Cùng seed RoI sampling cho CE và margin. Kèm: giá trị margin tại δ = 0 (âm = đúng lớp)."""
import contextlib
import os
import random
import statistics
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.methods.repair import roi_margin
from attack.models import load_model
from attack.preprocess import resize_to_model, to_batch

dev = "cuda:0"
model = load_model("r50", dev)
model.roi_head.mask_head = None  # như skip_mask
ds_all = build_attack_dataset("dev300")
keep = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:5]
pos = [ds_all.image_ids.index(i) for i in keep]


def grad_cls(orig, ds, ctx, seed):
    random.seed(seed); torch.manual_seed(seed)
    n = torch.zeros_like(orig, requires_grad=True)
    bi, bds = to_batch(model, [resize_to_model(orig + n, tuple(ds.img_shape))], [ds])
    with ctx():
        l = model.loss(bi, bds)["loss_cls"]
    return torch.autograd.grad(l, n)[0], float(l)


ratios = {"all": [], "fg": []}
for j in pos:
    s = ds_all[j]
    orig, ds = s["orig_inputs"].to(dev), s["data_sample"]
    if "masks" in ds.gt_instances:
        del ds.gt_instances.masks
    g_ce, l_ce = grad_cls(orig, ds, contextlib.nullcontext, j)
    for mode in ratios:
        g_m, l_m = grad_cls(orig, ds, lambda: roi_margin(model, mode, 1.0), j)
        r = float(g_ce.abs().mean() / (g_m.abs().mean() + 1e-12))
        ratios[mode].append(r)
        print(f"img {ds_all.image_ids[j]} {mode}: CE {l_ce:.3f}, margin {l_m:.3f}, |∇CE|/|∇mrg| {r:.4f}")
for mode, r in ratios.items():
    print(f"{mode}: scale = trung vị {statistics.median(r):.4f} (từ {[round(x, 4) for x in r]})")
