#!/usr/bin/env python3
"""Kiểm core.wp_update (C2-OCT-1V-WPA, progress_log 2026-10-23) trên tensor giả — không cần GPU / model.
1. eta = None trùng tuyệt đối công thức WP2 cũ.  2. WPA: ‖θ' − θ0‖ ≤ rho·‖θ0‖ sau mọi bước (không tăng tuyến tính theo B).
3. Gradient hướng cố định: WPA tới biên sau rho / eta bước và trùng điểm WP2.  4. Gradient đổi hướng: WPA giữ trí nhớ
(cos với gradient bước trước > 0), WP2 không."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attack.methods.core import wp_update

torch.manual_seed(0)
rho, eta = 0.002, 0.0005
p0 = torch.randn(64, 32, 3, 3)
r = rho * p0.norm()

g = torch.randn_like(p0)
assert torch.equal(wp_update(p0.clone(), p0, g, rho), p0 - rho * p0.norm() * g / (g.norm() + 1e-12))
print("1 OK: eta = None = WP2")

p = p0.clone()
for _ in range(200):
    p = wp_update(p, p0, torch.randn_like(p0), rho, eta)
    assert (p - p0).norm() <= r * (1 + 1e-5)
print(f"2 OK: 200 bước gradient ngẫu nhiên, ‖Δ‖ / bán kính = {float((p - p0).norm() / r):.3f} ≤ 1")

p, norms = p0.clone(), []
for _ in range(6):
    p = wp_update(p, p0, g, rho, eta)
    norms.append(float((p - p0).norm() / r))
assert abs(norms[3] - 1) < 1e-4 and norms[2] < 0.76
assert torch.allclose(p, wp_update(p0.clone(), p0, g, rho), atol=1e-6)
print("3 OK: hướng cố định ‖Δ‖ / bán kính =", [round(x, 3) for x in norms], "-> trùng điểm WP2")

g1, g2 = torch.randn_like(p0), torch.randn_like(p0)
p = p0.clone()
for _ in range(8):
    p = wp_update(p, p0, g1, rho, eta)
p = wp_update(p, p0, g2, rho, eta)
cos = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()))
c_wpa, c_wp2 = cos(p0 - p, g1), cos(p0 - wp_update(p0.clone(), p0, g2, rho), g1)
assert c_wpa > 0.5 and abs(c_wp2) < 0.05
print(f"4 OK: sau khi gradient đổi hướng, cos(−Δ, g_cũ): WPA {c_wpa:.3f} | WP2 {c_wp2:.3f}")
