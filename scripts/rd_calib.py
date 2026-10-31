#!/usr/bin/env python3
"""E39 (progress_log 2026-10-28): λ cho WP2 rút gọn theo ĐÚNG quy tắc chọn λ = 4 cũ — trung vị 5 ảnh đầu dev300 của
mean|∇ₓL_feat| / mean|∇ₓL_cls| tại δ = 0 (pia.calib_cls_weight); in cả cls_mode "c2" (kiểm tái lập ~3.8) và "roi".
E45: thêm cls_mode qua argv (vd `python scripts/rd_calib.py vanish`)."""
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset
from attack.methods.pia import calib_cls_weight
from attack.models import load_model
from attack.preprocess import resize_to_model

model = load_model(os.environ.get("SUR", "r50"), "cuda:0")  # SUR: surrogate khác R50 (E63)
data = build_attack_dataset("dev300")
for mode in (sys.argv[1:] or ["c2", "roi"]):
    raws = []
    for i in range(5):
        s = data[i]
        o, ds = s["orig_inputs"].cuda().float(), s["data_sample"]
        torch.manual_seed(0)
        raws.append(calib_cls_weight(model, o, resize_to_model(o, tuple(ds.img_shape)), ds, 3.0, mode)[1])
    print(mode, " ".join(f"{r:.2f}" for r in raws), "| trung vị", f"{np.median(raws):.2f}")
