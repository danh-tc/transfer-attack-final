#!/usr/bin/env python3
"""Kiểm repair._xbg_mse (C2-OCT-1V-WP2-XBG / -XBGS, progress_log 2026-10-23) trên tensor giả — không cần GPU / model.
1. Không box -> trùng F.mse_loss.  2. Giá trị: add = MSE gốc − phần trong box; swap = phần ngoài box − phần trong box.
3. Gradient số hạng kéo chỉ ở ô trong box, cùng chiều (F_adv − F_bg) (descent của T -> F_adv tiến về F_bg)."""
import os
import sys

import torch
import torch.nn.functional as F
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attack.methods.repair import _box_mask, _xbg_mse


def ds(boxes):
    d = DetDataSample()
    d.gt_instances = InstanceData(bboxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4))
    return d


torch.manual_seed(0)
stage = 2  # stride 16
tgt, fbg = torch.randn(1, 8, 10, 12), torch.randn(1, 8, 10, 12)
fa = torch.randn(1, 8, 10, 12, requires_grad=True)

# 1
for mode in ("add", "swap"):
    v = _xbg_mse(tgt, fbg, fa, stage, ds([]), mode)
    assert torch.allclose(v, F.mse_loss(tgt, fa)), mode
print("1 OK: không box = F.mse_loss")

# 2
d = ds([[16, 32, 80, 96]])  # ô x 1..4, y 2..5 -> 4 × 4 = 16 ô
m = _box_mask(fa, stage, d)
assert int(m.sum()) == 16, int(m.sum())
sq = (tgt - fa).pow(2).mean(1)[0].detach()
frac = m.float().mean()
v_add = _xbg_mse(tgt, fbg, fa, stage, d, "add")
v_swap = _xbg_mse(tgt, fbg, fa, stage, d, "swap")
assert torch.allclose(v_add, sq.mean() - frac * sq[m].mean(), atol=1e-6)
assert torch.allclose(v_swap, sq[~m].sum() / sq.numel() - frac * sq[m].mean(), atol=1e-6)
print(f"2 OK: add {v_add.item():.4f}, swap {v_swap.item():.4f}")

# 3: gradient phần kéo = grad(add) − grad(MSE gốc)
g_add = torch.autograd.grad(_xbg_mse(tgt, fbg, fa, stage, d, "add"), fa)[0]
g_full = torch.autograd.grad(F.mse_loss(tgt, fa), fa)[0]
g_pull = -(g_add - g_full)  # gradient của +c·T
assert g_pull[..., ~m].abs().max() < 1e-7
cos = F.cosine_similarity(g_pull[..., m].flatten(), (fa - fbg)[..., m].detach().flatten(), dim=0)
assert cos > 0.999, cos
print(f"3 OK: kéo chỉ trong box, cos(∇T, F_adv − F_bg) = {cos.item():.4f} -> ascent của −c·T đưa F_adv về F_bg")
