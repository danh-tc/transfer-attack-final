#!/usr/bin/env python3
"""Unit test SPIKE-WP (E28): phát xung trên trọng số. θ khôi phục; θ cao → ít element phát; ‖θ'−θ0‖≈ρ‖θ0‖.
Chạy: python scripts/wspike_unit_test.py"""
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
torch.manual_seed(0); n_wp = geo_osfd_attack(m, o, d, steps=8, epsilon=5, **_C2OCT, weight_rho=0.002)
r0 = all(torch.equal(p, b) for p, b in zip(m.backbone.parameters(), base))
for th in (1.0, 2.0):
    torch.manual_seed(0); n = geo_osfd_attack(m, o, d, steps=8, epsilon=5, **_C2OCT, weight_rho=0.002, weight_spike=th)
    r = all(torch.equal(p, b) for p, b in zip(m.backbone.parameters(), base))
    print(f"θ_fire={th}: L∞={float(n.abs().max()):.1f} θ_restored={r} ‖Δ vs WP2‖={float((n-n_wp).norm()):.0f}")
print(f"WP2 θ_restored={r0}; ‖δ_WP2‖={float(n_wp.norm()):.0f}")
print("SPIKE-WP unit test xong")
