#!/usr/bin/env python3
"""Unit test OTSV (E21, progress_log 2026-10-24): otsv_mix khớp công thức viết tay; β = 0 / không box -> đồng nhất;
1 box -> mọi token cộng cùng β·v. Chạy: python scripts/otsv_unit_test.py"""
import os
import sys

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.methods.pia import otsv_mix

torch.manual_seed(0)
f = torch.randn(1, 16, 9, 12).relu()
boxes = torch.tensor([[10.0, 20.0, 80.0, 120.0], [100.0, 5.0, 180.0, 60.0], [0.0, 0.0, 190.0, 140.0]])
assert torch.equal(otsv_mix(f, boxes, 0.0), f) and torch.equal(otsv_mix(f, boxes[:0], 0.5), f)
z = F.layer_norm(f.permute(0, 2, 3, 1), (16,)).permute(0, 3, 1, 2)
t = roi_align(z, [boxes], 7, 1 / 16, 2, True).mean((2, 3))
v = roi_align(f, [boxes], 7, 1 / 16, 2, True).mean((2, 3))
ref = f.clone()
for i in range(9):
    for j in range(12):
        a = (z[0, :, i, j] @ t.t() / 4.0).softmax(-1)
        ref[0, :, i, j] += 0.5 * (a @ v)
err = float((otsv_mix(f, boxes, 0.5) - ref).abs().max())
assert err < 1e-5, err
one = otsv_mix(f, boxes[:1], 1.0) - f
assert float((one - one[:, :, :1, :1]).abs().max()) < 1e-5
print(f"OTSV unit test ĐẠT (max|Δ| so công thức = {err:.1e})")
