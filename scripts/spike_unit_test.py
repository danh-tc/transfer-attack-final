#!/usr/bin/env python3
"""Unit test SPIKE (E27): LIF update trên δ. θ_fire=0 mọi pixel phát mỗi bước; θ cao → thưa; V reset; L∞ ≤ ε.
Chạy: python scripts/spike_unit_test.py"""
import os, sys, math
import torch
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.models import load_model
from attack.data import AttackDataset, load_image_ids
from attack.methods.pia import geo_osfd_attack

m = load_model("r50", "cuda:0")
ds = AttackDataset(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")))
_C2OCT = dict(geo="elem", elem={"scale": math.log(2)/2, "shift": "ph8"}, n_views=1, cls_weight=4.0, skip_mask=True, amp=True)
s = ds[0]; o = s["orig_inputs"].cuda(); d = s["data_sample"]
base = [p.detach().clone() for p in m.backbone.parameters()]
frac = {}
def hook_frac(th):
    # đếm % pixel phát ở bước cuối: so δ sau B và B-1 qua snapshot
    seen = {}
    def on_step(k, nz): seen[k] = nz.clone()
    torch.manual_seed(0)
    n = geo_osfd_attack(m, o, d, steps=6, epsilon=5, **_C2OCT, weight_rho=0.002, spike=th, on_step=on_step)
    r = all(torch.equal(p, b) for p, b in zip(m.backbone.parameters(), base))
    fired = (seen[6] != seen[5]).float().mean().item() if 6 in seen and 5 in seen else -1
    return n, fired, r
for th in (0.0, 1.0, 2.0):
    n, fired, r = hook_frac(th)
    print(f"θ_fire={th}: L∞={float(n.abs().max()):.1f} %pixel_phát_bước_cuối={100*fired:.0f}% θ_restored={r} δ_thưa(|δ|>0)={100*(n.abs()>0.01).float().mean():.0f}%")
print("SPIKE unit test xong")
