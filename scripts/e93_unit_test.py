"""E93 unit test — HardBank (hardview) khớp loss RD-CNX sau khi truyền stage_w (progress_log 2026-11-14 E93).
T1 regression: stage_w=None (ô E37 C2-OCT-1V-HB) — code mới trùng code HEAD~ (E37) trên ConvNeXt, 3 bước, seed cố định;
   so cả mức nhiễu nền (code mới chạy 2 lần).
T2 fidelity: cùng view ép cố định (ln s = 0, pha (0,0)), cùng δ = 0, cùng RNG, cùng autocast: (view, loss_fn) do nhánh
   HB trả về vs nhánh RD-CNX thường — loss VÀ ∇δ phải trùng.
T3 có hiệu lực: HB code CŨ (bỏ stage_w) cho ∇δ KHÁC nhánh RD-CNX (chứng minh bản sửa đổi đúng chỗ).
Ảnh: devC300[0] (đã dùng ở E90b; không tiêu ảnh mới).  python scripts/e93_unit_test.py
"""
import functools
import importlib.util
import os
import random
import subprocess
import sys
import types

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import attack.methods.elem as elem_mod
import attack.methods.pia as pia_new
from attack.data import load_image_ids
from attack.models import load_model
from attack.preprocess import resize_to_model
from attack.targets import target_dataset
from scripts.run_baselines import METHODS, _C2OCT, _E63

SEED = 4242


def old_pia():
    src = subprocess.check_output(["git", "-C", REPO_ROOT, "show", "HEAD:attack/methods/pia.py"], text=True)
    assert "stage_w=stage_w)(model, v, dsv)" not in src, "HEAD đã chứa bản sửa — so sai mốc"
    m = types.ModuleType("attack.methods.pia_old")
    m.__file__ = "pia_old.py"
    exec(compile(src, "pia_old.py", "exec"), m.__dict__)
    return m


def seed_all(s):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)


def run(mod, kw, model, orig, ds, steps=3):
    seed_all(SEED)
    return mod.geo_osfd_attack(model, orig, ds, steps=steps, epsilon=5.0, **kw)


def capture(mod, kw, model, orig, ds):
    """Thay run_iterative_attack bằng hàm bắt view bước 0 tại δ = 0 + tính loss/∇δ NGAY trong ngữ cảnh của pia
    (autocast, skip_mask...). RNG đặt lại cố định trước forward loss."""
    out = {}

    def fake(model_, orig_pixels, data_sample, views_fn=None, steps=1, **k):
        noise = torch.zeros_like(orig_pixels, requires_grad=True)
        adv = resize_to_model(torch.clamp(orig_pixels + noise, 0, 255), tuple(data_sample.img_shape))
        views = views_fn(adv, data_sample, 0, steps)
        assert len(views) == 1
        v, dsv, lf = views[0][:3]
        torch.manual_seed(SEED + 1); torch.cuda.manual_seed_all(SEED + 1); random.seed(SEED + 1)
        L = lf(model_, v, dsv)
        L = L[0] + L[1] if isinstance(L, tuple) else L
        g, = torch.autograd.grad(L, noise)
        out.update(v=v.detach().clone(), L=float(L), g=g.detach().clone())
        return torch.zeros_like(orig_pixels)

    saved = mod.run_iterative_attack
    mod.run_iterative_attack = fake
    try:
        seed_all(SEED)
        mod.geo_osfd_attack(model, orig, ds, steps=1, epsilon=5.0, **kw)
    finally:
        mod.run_iterative_attack = saved
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    i = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/devC300.csv"))[0]
    model = load_model("convnext_t", "cuda:0")
    s = target_dataset("convnext_t", [i])[0]
    orig, ds = s["orig_inputs"].cuda().float(), s["data_sample"]
    old = old_pia()
    ok = True

    # ---- T1 regression (ô E37) ----
    hb37 = METHODS["C2-OCT-1V-HB"][0].keywords
    d_new1 = run(pia_new, hb37, model, orig, ds)
    d_new2 = run(pia_new, hb37, model, orig, ds)
    d_old = run(old, hb37, model, orig, ds)
    noise_floor = float((d_new1 != d_new2).float().mean())
    diff = float((d_new1 != d_old).float().mean())
    t1 = diff <= noise_floor
    print(f"T1 regression stage_w=None: frac pixel khác (mới vs E37) {diff:.6f} | nền (mới vs mới) {noise_floor:.6f} "
          f"| max|Δ| {float((d_new1 - d_old).abs().max()):.4f} -> {'PASS' if t1 else 'FAIL'}")
    ok &= t1

    # ---- T2 / T3 fidelity: ép 1 view cố định cho mọi lời gọi elem_views ----
    orig_ev = elem_mod.elem_views

    def forced(adv_pixels, n, spec, sigma=6.0, **k):
        k["fixed"] = [(0.0, 0, 0)] * n
        return orig_ev(adv_pixels, n, spec, sigma, **k)
    elem_mod.elem_views = forced
    try:
        rd = dict(_C2OCT, cls_weight=0.2, **_E63, weight_rho=0.002, weight_linear=True)   # = RD-CNX
        hb = dict(rd, hardview=(1, 1))                                                   # bank 1 view = view ép
        c_rd = capture(pia_new, rd, model, orig, ds)
        c_hb = capture(pia_new, hb, model, orig, ds)
        c_hb_old = capture(old, hb, model, orig, ds)
    finally:
        elem_mod.elem_views = orig_ev
    same_v = torch.equal(c_rd["v"], c_hb["v"])
    dL = abs(c_rd["L"] - c_hb["L"])
    dg = float((c_rd["g"] - c_hb["g"]).abs().max())
    gnorm = float(c_rd["g"].abs().max())
    t2 = same_v and dL <= 1e-6 * max(1.0, abs(c_rd["L"])) and dg <= 1e-6 * max(gnorm, 1e-12)
    print(f"T2 fidelity (HB mới vs RD-CNX, cùng view/δ/RNG/autocast): view trùng {same_v} | |ΔL| {dL:.3e} (L {c_rd['L']:.4f}) "
          f"| max|Δg| {dg:.3e} (max|g| {gnorm:.3e}) -> {'PASS' if t2 else 'FAIL'}")
    ok &= t2
    dL3 = abs(c_rd["L"] - c_hb_old["L"])
    cos3 = float((c_rd["g"] * c_hb_old["g"]).sum() / (c_rd["g"].norm() * c_hb_old["g"].norm() + 1e-12))
    t3 = dL3 > 1e-3 * max(1.0, abs(c_rd["L"]))
    print(f"T3 bản sửa có hiệu lực (HB CŨ vs RD-CNX): |ΔL| {dL3:.4f} | cos(g) {cos3:.4f} -> {'PASS' if t3 else 'FAIL'}")
    ok &= t3
    print(f"\nTẤT CẢ: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
