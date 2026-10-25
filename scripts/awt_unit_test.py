#!/usr/bin/env python3
"""Unit test AWT-OD (E31). Phần 1 (CPU, model đồ chơi, core): β = lr = 0 -> θ không đổi, δ trùng không-WP; 1 bước:
‖θ1 − θ0‖ = lr‖θ0‖ từng tensor; T bước: ‖θT − θ0‖ ≤ T·lr‖θ0‖ (tích lũy, > 1 bước); β ≠ 0 đổi hướng descend.
Phần 2 (GPU, R50 nếu có checkpoint): θ khôi phục y nguyên (torch.equal) sau geo_osfd_attack(weight_awt), L∞ = ε.
Chạy: python scripts/awt_unit_test.py"""
import os, sys, math
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


def run(steps, beta, lr):
    with torch.no_grad():
        for p, p0 in zip(params, theta0):
            p.copy_(p0)
    wp = (params, None, theta0, None, ("awt", beta, lr), None)
    n = run_iterative_attack(net, x, ds, loss_fn, steps=steps, epsilon=5, momentum=1.0, weight_perturb=wp)
    return n, [p.detach().clone() for p in params]


def rel(a, b):
    return [float((ai - bi).norm() / bi.norm()) for ai, bi in zip(a, b)]


ok = True
n_plain = run_iterative_attack(net, x, ds, loss_fn, steps=5, epsilon=5, momentum=1.0)
n0, th = run(5, 0.0, 0.0)
c = torch.equal(n0, n_plain) and all(torch.equal(a, b) for a, b in zip(th, theta0))
print(f"[1] β=lr=0: δ trùng không-WP và θ không đổi: {c}"); ok &= c
_, th1 = run(1, 0.005, 0.002)
r1 = rel(th1, theta0)
c = all(abs(r - 0.002) < 1e-5 for r in r1)
print(f"[2] 1 bước: ‖θ1−θ0‖/‖θ0‖ = {[round(r, 6) for r in r1]} (kỳ vọng 0.002): {c}"); ok &= c
_, th8 = run(8, 0.005, 0.002)
r8 = rel(th8, theta0)
c = all(0.002 + 1e-5 < r <= 8 * 0.002 + 1e-5 for r in r8)
print(f"[3] 8 bước tích lũy: {[round(r, 5) for r in r8]} ∈ (0.002, 0.016]: {c}"); ok &= c
_, th1b = run(1, 0.0, 0.002)
d = max(float((a - b).norm()) for a, b in zip(th1, th1b))
c = d > 0
print(f"[4] β=0.005 vs β=0: hướng descend khác (‖Δθ1‖ = {d:.2e}): {c}"); ok &= c

if os.path.exists(os.path.join(REPO_ROOT, "checkpoints")) and torch.cuda.is_available():
    try:
        from attack.models import load_model
        from attack.data import AttackDataset, load_image_ids
        from attack.methods.pia import geo_osfd_attack
        m = load_model("r50", "cuda:0")
    except Exception as e:  # checkpoint / dataset chưa tải xong
        print(f"[5] bỏ qua phần R50: {type(e).__name__}: {e}")
    else:
        dset = AttackDataset(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")))
        _C2OCT = dict(geo="elem", elem={"scale": math.log(2) / 2, "shift": "ph8"}, n_views=1, cls_weight=4.0,
                      skip_mask=True, amp=True)
        s = dset[0]; o = s["orig_inputs"].cuda(); dsm = s["data_sample"]
        base = [p.detach().clone() for p in m.parameters()]
        torch.manual_seed(0); n = geo_osfd_attack(m, o, dsm, steps=8, epsilon=5, **_C2OCT, weight_awt=(0.005, 0.002))
        r = all(torch.equal(p, b) for p, b in zip(m.parameters(), base))
        c = r and abs(float(n.abs().max()) - 5.0) < 1e-4
        print(f"[5] R50: θ khôi phục={r}, L∞={float(n.abs().max()):.1f}, chạm ±5 = {float((n.abs() >= 5).float().mean()):.2f}: {c}")
        ok &= c
print("AWT unit test", "QUA" if ok else "TRƯỢT")
sys.exit(0 if ok else 1)
