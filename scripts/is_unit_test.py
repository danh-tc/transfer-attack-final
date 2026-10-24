#!/usr/bin/env python3
"""Unit test IS (E30): _is_reweight triệt kênh gradient tương quan cao; α=0/N=0 ≡ không đổi; w∈[0,1]; θ khôi phục.
Chạy: python scripts/is_unit_test.py [--gpu]"""
import os, sys, math, argparse
import torch
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.methods.repair import _is_reweight

p = argparse.ArgumentParser(); p.add_argument("--gpu", action="store_true"); args = p.parse_args()
gen = torch.Generator().manual_seed(0)
# 8 kênh: 0-3 độc lập, 4-7 = bản sao kênh 0 (tương quan =1, dư thừa)
base = torch.randn(1, 8, 6, 6)
base[:, 4:] = base[:, :1] * torch.tensor([1.,1.,1.,1.])[None,:,None,None].repeat(1,1,6,6)[:, :4] + base[:,:1]*0  # 4..7 ≡ kênh 0
g = base.clone()
out = _is_reweight(g, 200, 1.0, 0.3, gen)
w = (out[0].reshape(8,-1) / (g[0].reshape(8,-1)+1e-9)).mean(1)
print("w mỗi kênh (0-3 độc lập, 4-7 ≡ kênh0 dư thừa):", [round(float(x),2) for x in w])
assert float(out.abs().max()) <= float(g.abs().max())+1e-6
# α=0 ≡ không đổi; N=0 ≡ không đổi
assert torch.allclose(_is_reweight(g, 200, 1.0, 0.0, torch.Generator().manual_seed(0)), g, atol=1e-5)
assert torch.equal(_is_reweight(g, 0, 1.0, 0.3, gen), g)
assert float(w.min()) >= 0 and float(w.max()) <= 1.001
print("α=0 / N=0 ≡ không đổi ✓; w∈[0,1] ✓")
if args.gpu:
    from attack.models import load_model
    from attack.data import AttackDataset, load_image_ids
    from attack.methods.pia import geo_osfd_attack
    m = load_model("r50", "cuda:0")
    ds = AttackDataset(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")))
    _C2OCT = dict(geo="elem", elem={"scale": math.log(2)/2, "shift": "ph8"}, n_views=1, cls_weight=4.0, skip_mask=True, amp=True)
    s = ds[0]; o = s["orig_inputs"].cuda(); d = s["data_sample"]
    b = [pp.detach().clone() for pp in m.backbone.parameters()]
    torch.manual_seed(0); n_wp = geo_osfd_attack(m, o, d, steps=6, epsilon=5, **_C2OCT, weight_rho=0.002)
    torch.manual_seed(0); n = geo_osfd_attack(m, o, d, steps=6, epsilon=5, **_C2OCT, weight_rho=0.002, is_supp=(frozenset({2}),1.0,0.3,10))
    r = all(torch.equal(pp, bb) for pp, bb in zip(m.backbone.parameters(), b))
    nh = len(m.backbone.layer3._forward_hooks)
    print(f"GPU: L∞={float(n.abs().max()):.1f} θ_restored={r} hook_gỡ={nh==0} ‖Δ vs WP2‖={float((n-n_wp).norm()):.0f}")
print("IS unit test xong")
