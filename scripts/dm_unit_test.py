#!/usr/bin/env python3
"""Kiểm C2-OCT-1V-DM / -DMA + snapshot theo B (progress_log 2026-10-20), 3 ảnh đầu dev300.

1. make_view_loss(split=True): tổng 2 phần trùng tuyệt đối loss gốc (cùng view, cùng RNG).
2. DM / DMA chạy được, L∞ ≤ ε; DMA: bước chẵn chỉ L_feat, bước lẻ chỉ L_cls (đếm forward sạch).
3. Snapshot: noise sau k bước của lần chạy B lớn vs lần chạy steps=k (cùng seed) — lệch pixel so với sàn không tất định
   (2 lần chạy B = k y hệt).
4. Thời gian / ảnh của OSFD B50, C2-OCT-1V, DM, DMA (để chọn B cùng thời gian, khóa trước khi chạy).
"""
import functools
import os
import random
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from attack.data import build_attack_dataset
from attack.methods.baselines import osfd_attack
from attack.methods.elem import elem_views
from attack.methods.repair import make_view_loss
from attack.models import load_model
from attack.preprocess import extract_features, resize_to_model
from run_baselines import METHODS, SEED, _OCT

dev = "cuda:0"
model = load_model("r50", dev)
ds_all = build_attack_dataset("dev300")


def seed(i):
    random.seed(SEED + i); torch.manual_seed(SEED + i)


def frac(a, b):
    return float((torch.round(a) != torch.round(b)).float().mean())


# 1. split
s = ds_all[0]
orig, ds = s["orig_inputs"].to(dev), s["data_sample"]
clean = resize_to_model(orig, tuple(ds.img_shape))
seed(0)
(v, c, dsv), = elem_views(clean + 0.0, 1, {"scale": _OCT, "shift": "ph8"}, 6.0, companion=clean, data_sample=ds)
with torch.no_grad():
    fc = tuple(f.detach() for f in extract_features(model, c, ds))
torch.manual_seed(1); tot = make_view_loss(fc, 3.0, "c2", 4.0)(model, v, dsv)
torch.manual_seed(1); a, b = make_view_loss(fc, 3.0, "c2", 4.0, split=True)(model, v, dsv)
print(f"[1] tổng {float(tot):.6f} vs feat + cls {float(a + b):.6f} (feat {float(a):.4f}, 4·cls {float(b):.4f})")
assert abs(float(tot) - float(a + b)) < 1e-4

# 2 + 4. chạy + thời gian
fns = {"OSFD": (osfd_attack, 50), "C2-OCT-1V": (METHODS["C2-OCT-1V"][0], 41),
       "C2-OCT-1V-DM": (METHODS["C2-OCT-1V-DM"][0], 41), "C2-OCT-1V-DMA": (METHODS["C2-OCT-1V-DMA"][0], 41)}
times = {k: [] for k in fns}
for i in range(3):
    s = ds_all[i]
    orig, ds = s["orig_inputs"].to(dev), s["data_sample"]
    for name, (fn, B) in fns.items():
        seed(i)
        info, kw = {}, ({} if name == "OSFD" else {"info": {}})
        if name != "OSFD":
            kw["info"] = info
        torch.cuda.synchronize(); t0 = time.time()
        nz = fn(model, orig, ds, steps=B, epsilon=5.0, **kw)
        torch.cuda.synchronize(); dt = time.time() - t0
        if i > 0:  # ảnh 0 = khởi động (cudnn / JIT)
            times[name].append(dt / B)
        linf = float(torch.round(nz).abs().max())
        extra = f" forward_no_grad={info.get('forward_no_grad')}" if info else ""
        print(f"[2] img {i} {name} B{B}: L∞ {linf:.1f} {dt:.2f}s{extra}")
        assert linf <= 5.0
        if name == "C2-OCT-1V-DMA":
            assert info["forward_no_grad"] == (B + 1) // 2, info["forward_no_grad"]
print("[4] s / bước (TB ảnh 1–2):", {k: round(sum(v) / len(v), 4) for k, v in times.items()})
t_osfd = 50 * sum(times["OSFD"]) / len(times["OSFD"])
print("[4] B cùng thời gian OSFD B50:", {k: round(t_osfd / (sum(v) / len(v)), 1) for k, v in times.items()})

# 3. snapshot
s = ds_all[1]
orig, ds = s["orig_inputs"].to(dev), s["data_sample"]
fn = METHODS["C2-OCT-1V"][0]
snap = {}
seed(1); fn(model, orig, ds, steps=20, epsilon=5.0, info={}, on_step=lambda k, n: snap.__setitem__(k, n.clone()) if k == 10 else None)
seed(1); r1 = fn(model, orig, ds, steps=10, epsilon=5.0, info={})
seed(1); r2 = fn(model, orig, ds, steps=10, epsilon=5.0, info={})
print(f"[3] snapshot k=10 vs chạy B10: lệch {frac(snap[10], r1):.3f} pixel; sàn (B10 vs B10) {frac(r1, r2):.3f}")
print("OK")
