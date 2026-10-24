#!/usr/bin/env python3
"""Unit test FPA (E29): FP layer trong forward adv — hoán vị CHỈ F_adv, đích k·F_cln giữ nguyên.
p=0 ≡ MSE thường; p=1 tạo mismatch (MSE tăng khi adv≈clean); γ lớn → mismatch nhiều hơn. Chạy: python scripts/fpa_unit_test.py"""
import os, sys
import torch
import torch.nn.functional as F
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.methods.repair import _fpa_mse

torch.manual_seed(0)
f = torch.randn(1, 64, 20, 25).relu()  # giả lập feature
# p=0 ≡ MSE thường
assert abs(float(_fpa_mse(3*f, 3*f, 0.6, 0.0))) < 1e-9
# adv ≈ clean (fa = k·fc): không hoán vị MSE=0; có hoán vị (p=1) → MSE > 0 (mismatch chéo vị trí)
torch.manual_seed(1); m_perm = float(_fpa_mse(3*f, 3*f, 0.6, 1.0))
assert m_perm > 1e-3, f"hoán vị adv khi adv=clean phải tạo mismatch, được {m_perm}"
# γ lớn → nhiều kênh bị hoán vị → mismatch lớn hơn
torch.manual_seed(2); m_lo = float(_fpa_mse(3*f, 3*f, 0.2, 1.0))
torch.manual_seed(2); m_hi = float(_fpa_mse(3*f, 3*f, 0.9, 1.0))
print(f"p=0 ≡ MSE ✓ | adv=clean+perm → mismatch MSE={m_perm:.3f} ✓ | γ=0.2 {m_lo:.3f} < γ=0.9 {m_hi:.3f}: {m_lo<m_hi}")
# gradient chảy qua (differentiable)
fa = (3*f).clone().requires_grad_(True)
g = torch.autograd.grad(_fpa_mse(3*f, fa, 0.6, 1.0), fa)[0]
print(f"gradient chảy qua hoán vị: ‖g‖={float(g.norm()):.2f} (>0: {float(g.norm())>0})")
print("FPA unit test xong")
