#!/usr/bin/env python3
"""Unit test E33 (WP-L4 / WP2-L4x / WP2-RS). Phần 1 (CPU, core): rho float trùng tuyệt đối rho list hằng và callable hằng;
rho list -> ‖θ' − θ0‖ = ρ_i‖θ0‖ từng tensor; callable (0.004 -> 0) -> bước cuối dùng ρ của bước T−2.
Phần 2 (GPU, R50): scope layer4 chỉ chạm tensor layer4; θ khôi phục torch.equal; L∞ = 5 cho 3 ô.
Chạy: python scripts/e33_unit_test.py"""
import math
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from mmdet.structures import DetDataSample
from attack.methods.core import run_iterative_attack

torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU(), torch.nn.Conv2d(8, 4, 3))
params = [m.weight for m in net if isinstance(m, torch.nn.Conv2d)]
theta0 = [p.detach().clone() for p in params]
x = torch.rand(3, 24, 24) * 255
ds = DetDataSample(metainfo=dict(img_shape=(24, 24)))
loss_fn = lambda m, img, d: m(img[None] / 255).pow(2).mean()


def run(steps, rho):
    with torch.no_grad():
        for p, p0 in zip(params, theta0):
            p.copy_(p0)
    n = run_iterative_attack(net, x, ds, loss_fn, steps=steps, epsilon=5, momentum=1.0,
                             weight_perturb=(params, rho, theta0, None, "def", None))
    return n, [float((p - p0).norm() / p0.norm()) for p, p0 in zip(params, theta0)]


ok = True
n_f, r_f = run(6, 0.002)
n_l, r_l = run(6, [0.002, 0.002])
n_c, r_c = run(6, lambda t, T: 0.002)
c = torch.equal(n_f, n_l) and torch.equal(n_f, n_c)
print(f"[1] float = list hằng = callable hằng (δ trùng tuyệt đối): {c}"); ok &= c
_, r = run(3, [0.002, 0.005])
c = abs(r[0] - 0.002) < 1e-5 and abs(r[1] - 0.005) < 1e-5
print(f"[2] rho list: ‖Δθ‖/‖θ0‖ = {[round(v, 6) for v in r]} (kỳ vọng [0.002, 0.005]): {c}"); ok &= c
T = 5
_, r = run(T, lambda t, TT: 0.004 + (0.0 - 0.004) * t / (TT - 1))
c = all(abs(v - 0.0) < 1e-6 for v in r)  # θ cuối = θ' của bước T−1 (ρ = 0) -> θ0
print(f"[3] lịch 0.004 -> 0: θ sau bước cuối ‖Δθ‖/‖θ0‖ = {[round(v, 6) for v in r]} (kỳ vọng 0): {c}"); ok &= c

if torch.cuda.is_available() and os.path.isdir(os.path.join(REPO_ROOT, "checkpoints")):
    from attack.data import AttackDataset, load_image_ids
    from attack.methods.pia import geo_osfd_attack
    from attack.models import load_model
    m = load_model("r50", "cuda:0")
    dset = AttackDataset(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")))
    _C2OCT = dict(geo="elem", elem={"scale": math.log(2) / 2, "shift": "ph8"}, n_views=1, cls_weight=4.0,
                  skip_mask=True, amp=True)
    s = dset[0]; o = s["orig_inputs"].cuda(); d = s["data_sample"]
    base = {n: p.detach().clone() for n, p in m.backbone.named_parameters()}
    # scope layer4: chụp θ' giữa chừng qua on_step (sau bước 1, θ' đã đặt)
    seen = {}

    def snap(k, noise):
        if k == 1:
            seen.update({n: float((p - base[n]).norm()) for n, p in m.backbone.named_parameters()})
    torch.manual_seed(0)
    geo_osfd_attack(m, o, d, steps=3, epsilon=5, **_C2OCT, weight_rho=0.002, weight_scope="layer4", on_step=snap)
    moved = {n for n, v in seen.items() if v > 0}
    c = bool(moved) and all(n.startswith("layer4.") for n in moved)
    print(f"[4] scope layer4: {len(moved)} tensor bị nhiễu, đều trong layer4: {c}"); ok &= c
    for kw in (dict(weight_scope="layer4"), dict(weight_rho_l4=0.005), dict(weight_rho_sched=(0.004, 0.0))):
        torch.manual_seed(0)
        n = geo_osfd_attack(m, o, d, steps=8, epsilon=5, **_C2OCT, weight_rho=0.002, **kw)
        r = all(torch.equal(p, base[k]) for k, p in m.backbone.named_parameters())
        c = r and abs(float(n.abs().max()) - 5.0) < 1e-4
        print(f"[5] {kw}: θ khôi phục={r}, L∞={float(n.abs().max()):.1f}: {c}"); ok &= c
print("E33 unit test", "QUA" if ok else "TRƯỢT")
sys.exit(0 if ok else 1)
