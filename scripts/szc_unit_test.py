#!/usr/bin/env python3
"""Kiểm roi_size_weight (progress_log 2026-10-22, C2-OCT-1V-WP2-SZC) trên 5 ảnh đầu dev30, δ = 0, ảnh không biến đổi,
cùng seed RoI sampling: (1) w_large = 1 trùng loss_cls gốc; (2) w_large = 0 -> loss_cls giảm đúng bằng phần RoI dương gán
cho GT lớn (tính độc lập qua sampling_results); (3) ảnh không có GT lớn -> không đổi; (4) khôi phục phương thức gốc."""
import contextlib
import os
import random
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.methods.repair import roi_size_weight
from attack.models import load_model
from attack.preprocess import resize_to_model, to_batch

dev = "cuda:0"
model = load_model("r50", dev)
model.roi_head.mask_head = None  # như skip_mask
head = model.roi_head.bbox_head
ds_all = build_attack_dataset("dev300")
keep = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))[:5]


def loss_cls(orig, ds, ctx, seed):
    random.seed(seed); torch.manual_seed(seed)
    bi, bds = to_batch(model, [resize_to_model(orig, tuple(ds.img_shape))], [ds])
    with torch.no_grad(), ctx():
        return float(model.loss(bi, bds)["loss_cls"])


for i in keep:
    j = ds_all.image_ids.index(i)
    s = ds_all[j]
    orig, ds = s["orig_inputs"].to(dev), s["data_sample"]
    if "masks" in ds.gt_instances:
        del ds.gt_instances.masks
    sx, sy = ds.metainfo["scale_factor"]
    thr = 96 ** 2 * sx * sy
    gb = ds.gt_instances.bboxes
    gb = (gb.tensor if hasattr(gb, "tensor") else gb).float()
    n_large = int((((gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])) >= thr).sum())
    base = loss_cls(orig, ds, contextlib.nullcontext, j)
    w1 = loss_cls(orig, ds, lambda: roi_size_weight(model, thr, 1.0), j)
    w0 = loss_cls(orig, ds, lambda: roi_size_weight(model, thr, 0.0), j)
    print(f"img {i}: GT lớn {n_large}/{len(gb)} | loss_cls gốc {base:.5f}  w=1 {w1:.5f}  w=0 {w0:.5f}")
    assert abs(base - w1) < 1e-5, "w = 1 phải trùng gốc"
    assert (w0 < base - 1e-6) if n_large else abs(w0 - base) < 1e-5
assert "loss_and_target" not in vars(head) and "forward" not in vars(head.loss_cls), "chưa khôi phục phương thức"
print("OK")
