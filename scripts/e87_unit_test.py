"""E87 unit test — trọng số theo stage cho L_OSFD thuần (`feat_stage_w`, attack/methods/pia.py `_view_loss`).
Chạy trên 2 ảnh đầu x200 (dev300 \\ dev100) — KHÔNG đụng devB100. Surrogate CNX + R50, OCT 1 view, amp như E85/E87.
Kiểm (phải qua hết):
  ⚠️ Chính code HEAD KHÔNG tất định bit-chính-xác trên GPU (atomicAdd ở backward warp/resize; đo 2026-11-13: lặp cùng seed
  lệch ≤ 0.02% pixel δ, kể cả cudnn.deterministic). Nên "trùng" = (a) loss forward tại cùng input TRÙNG TUYỆT ĐỐI và
  (b) tỉ lệ pixel δ khác ≤ TOL = 1e-3 (nhiễu nền HEAD-vs-HEAD được đo và in ngay trong test).
  T1 đường mặc định (feat_stage_w=None) trùng bản code trước sửa (git HEAD, nạp vào module tạm).
  T2 α=1 (1,1,1,1) trùng mặc định;  T3 α=0 (0,1,1,1) trùng feat_stages=(1,2,3) (C3–C5).
  T4 α=0.5 thực sự đổi: δ khác α=1 và α=0; loss = L345 + 0.5·L2; ∇ = g345 + 0.5·g2 (allclose, fp32).
  T5 số backward (torch.autograd.grad trong core) không đổi giữa α ∈ {None, 1, 0.5, 0} (= STEPS).
  python scripts/e87_unit_test.py"""
import importlib.util
import os
import random
import subprocess
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
import attack.methods.core as core
from attack.methods import pia
from attack.methods.pia import _view_loss, geo_osfd_attack
from attack.models import load_model
from attack.preprocess import extract_features, resize_to_model
from attack.targets import target_dataset
from scripts.run_baselines import _C2OCT

STEPS = 4
TOL = 1e-3
OCT_KW = dict(geo="elem", elem=_C2OCT["elem"], n_views=1, amp=True)


def seed_all(s):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)


def old_pia():
    src = subprocess.check_output(["git", "-C", REPO_ROOT, "show", "HEAD:attack/methods/pia.py"])
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "pia_head_e87.py")
    open(path, "wb").write(src)
    spec = importlib.util.spec_from_file_location("pia_head_e87", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    d100 = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i not in d100][:2]
    old = old_pia()
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)

    n_grad = {"n": 0}
    _orig_grad = torch.autograd.grad

    def counting_grad(*a, **kw):
        n_grad["n"] += 1
        return _orig_grad(*a, **kw)
    core.torch.autograd.grad = counting_grad   # core dùng torch.autograd.grad -> đếm số backward

    for sur in ("convnext_t", "r50"):
        print(f"[e87-test] surrogate {sur}", flush=True)
        model = load_model(sur, "cuda:0")
        ds = {s["img_id"]: s for s in target_dataset(sur, ids)}
        for n, i in enumerate(ids):
            orig, dsm = ds[i]["orig_inputs"].cuda().float(), ds[i]["data_sample"]

            def run(fn=geo_osfd_attack, **kw):
                seed_all(1000 * (n + 1)); n_grad["n"] = 0
                d = fn(model, orig, dsm, steps=STEPS, epsilon=5.0, **OCT_KW, **kw)
                return d.detach().clone(), n_grad["n"]
            d_old, b_old = run(old.geo_osfd_attack)
            d_old2, _ = run(old.geo_osfd_attack)
            fr = lambda a, b: float((a != b).float().mean())
            print(f"  nhiễu nền HEAD-vs-HEAD img {i}: {fr(d_old, d_old2):.2e}", flush=True)
            d_def, b_def = run()
            d_old35, _ = run(old.geo_osfd_attack, feat_stages=(1, 2, 3))
            d_1, b_1 = run(feat_stage_w=(1.0, 1.0, 1.0, 1.0))
            d_0, b_0 = run(feat_stage_w=(0.0, 1.0, 1.0, 1.0))
            d_h, b_h = run(feat_stage_w=(0.5, 1.0, 1.0, 1.0))
            check(f"T1 img {i}: mặc định ≈ HEAD (δ khác {fr(d_old, d_def):.2e})", fr(d_old, d_def) <= TOL)
            check(f"T2 img {i}: α=1 ≈ mặc định (δ khác {fr(d_1, d_def):.2e})", fr(d_1, d_def) <= TOL)
            check(f"T3 img {i}: α=0 ≈ C3–C5 HEAD (δ khác {fr(d_0, d_old35):.2e})", fr(d_0, d_old35) <= TOL)
            check(f"T4a img {i}: α=0.5 ≠ α=1 và ≠ α=0 (frac khác {float((d_h != d_1).float().mean()):.3f} / "
                  f"{float((d_h != d_0).float().mean()):.3f})", not torch.equal(d_h, d_1) and not torch.equal(d_h, d_0))
            check(f"T5 img {i}: backward {b_old}/{b_def}/{b_1}/{b_h}/{b_0} == {STEPS}",
                  b_old == b_def == b_1 == b_h == b_0 == STEPS)
            # T4b: loss / gradient tuyến tính theo α (fp32, không amp)
            clean = resize_to_model(orig, tuple(dsm.img_shape))
            with torch.no_grad():
                fc = tuple(f.detach() for f in extract_features(model, clean, dsm))
            x = (orig + torch.empty_like(orig).uniform_(-5, 5)).requires_grad_(True)
            xin = lambda: resize_to_model(x, tuple(dsm.img_shape))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):   # (a) loss forward trùng tuyệt đối
                L = lambda f: float(f(model, xin(), dsm))
                l_def, l_1 = L(old._view_loss(fc, 3.0)), L(_view_loss(fc, 3.0, weights=(1, 1, 1, 1)))
                l_35, l_0 = L(old._view_loss(fc, 3.0, (1, 2, 3))), L(_view_loss(fc, 3.0, weights=(0, 1, 1, 1)))
                l_new = L(_view_loss(fc, 3.0))
            check(f"T1-3a img {i}: loss HEAD == None == α=1 ({l_def!r}) và C3–C5 HEAD == α=0 ({l_35!r})",
                  l_def == l_1 == l_new and l_35 == l_0)
            out = {}
            for nm, lf in (("h", _view_loss(fc, 3.0, weights=(0.5, 1, 1, 1))), ("c35", _view_loss(fc, 3.0, (1, 2, 3))),
                           ("c2", _view_loss(fc, 3.0, (0,)))):
                L = lf(model, xin(), dsm)
                g, = _orig_grad(L, x)
                out[nm] = (float(L), g.double())
            lin = out["c35"][0] + 0.5 * out["c2"][0]
            check(f"T4b img {i}: L(α=.5) = L345 + .5·L2 ({out['h'][0]:.6g} vs {lin:.6g})",
                  abs(out["h"][0] - lin) <= 1e-4 * abs(lin))
            ge = out["c35"][1] + 0.5 * out["c2"][1]
            rel = float((out["h"][1] - ge).norm() / ge.norm())
            check(f"T4c img {i}: ∇(α=.5) = g345 + .5·g2 (rel err {rel:.2e})", rel < 1e-3)
        del model; torch.cuda.empty_cache()
    core.torch.autograd.grad = _orig_grad
    print("[e87-test] " + ("TẤT CẢ PASS" if ok else "CÓ FAIL"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
