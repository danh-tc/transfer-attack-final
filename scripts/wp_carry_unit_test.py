"""Regression test cho wp_in / wp_out (bước 2 strict-1 + WP, docs/plan_paper_lock.md). Cần GPU.

4 test khóa trong plan (đều phải QUA trước khi chạy E82):
  T1 Đường mặc định trùng tuyệt đối bản cũ: so geo_osfd_attack MỚI vs bản ở git HEAD trước khi sửa (nạp từ
     `git show <REF>:attack/methods/pia.py`), cùng seed, cùng input. Mốc nhiễu = 2 lần chạy bản cũ (CUDA không
     tất định: ROIAlign backward). Thêm: wp_out=[] (chỉ đọc ra) phải trùng đường mặc định.
  T2 Trọng số gốc của surrogate được khôi phục sau khi sinh (kể cả khi nạp wp_in).
  T3 θ' thực sự làm đổi gradient ở lượt kế tiếp của CHÍNH surrogate: wp_out ≠ θ0 với ‖θ'−θ0‖ ≈ ρ‖θ0‖ từng tensor;
     cùng (δ, momentum, seed), gọi tiếp với wp_in=θ' cho δ khác rõ so với wp_in=None (> mốc nhiễu).
  T4 Không áp nhầm θ' giữa 2 surrogate (R50 -> ConvNeXt phải báo lỗi); số lần autograd.grad / lời gọi = steps
     (không thêm backward).

  python scripts/wp_carry_unit_test.py [--ref f5c003e]
"""
import argparse
import importlib.util
import os
import random
import subprocess
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import attack.methods.core as core
from attack.methods.pia import geo_osfd_attack
from attack.models import load_model
from attack.targets import target_dataset
from scripts.e81_ens_alloc_rho import kw_for, COMMON, R, C
from attack.data import load_image_ids

RHO = 0.002


def load_old_pia(ref):
    src = subprocess.check_output(["git", "-C", REPO_ROOT, "show", f"{ref}:attack/methods/pia.py"])
    path = os.path.join(tempfile.mkdtemp(), "pia_old.py")
    open(path, "wb").write(src)
    spec = importlib.util.spec_from_file_location("pia_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.geo_osfd_attack


def seed(s):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)


def wp_params(model, linear):
    t = (torch.nn.Conv2d, torch.nn.Linear) if linear else torch.nn.Conv2d
    return [m.weight for _, m in model.backbone.named_modules() if isinstance(m, t) and m.weight.requires_grad]


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def same_state(model, snap):
    return all(torch.equal(v, snap[k]) for k, v in model.state_dict().items())


def run(fn, model, orig, ds, kw, s, **extra):
    seed(s)
    return fn(model, orig, ds, steps=extra.pop("steps", 1), return_state=True, **kw, **COMMON, **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="f5c003e", help="commit trước khi thêm wp_in/wp_out")
    args = ap.parse_args()
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev5.csv"))[:2]
    models = {R: load_model(R, "cuda:0"), C: load_model(C, "cuda:0")}
    ds = {k: {s["img_id"]: s for s in target_dataset(k, ids)} for k in models}
    kw = {k: kw_for(k, RHO) for k in models}
    old_fn = load_old_pia(args.ref)
    ok = True

    for i in ids:
        orig = ds[R][i]["orig_inputs"].cuda().float()
        for key in (R, C):
            m, d = models[key], ds[key][i]["data_sample"]
            # ---- T1: mặc định trùng bản cũ (steps=2: WP2 có tác dụng ở bước 2 trong cùng lời gọi) ----
            a1, _ = run(old_fn, m, orig, d, kw[key], 7, steps=2)
            a2, _ = run(old_fn, m, orig, d, kw[key], 7, steps=2)
            b, _ = run(geo_osfd_attack, m, orig, d, kw[key], 7, steps=2)
            bo, _ = run(geo_osfd_attack, m, orig, d, kw[key], 7, steps=2, wp_out=[])
            floor = float((a1 - a2).abs().mean())
            dn, do = float((b - a1).abs().mean()), float((bo - a1).abs().mean())
            t1 = dn <= max(floor, 1e-6) * 3 + 1e-6 and do <= max(floor, 1e-6) * 3 + 1e-6
            print(f"[T1 {key} img {i}] floor(cũ vs cũ) {floor:.4f} | mới {dn:.4f} | mới+wp_out {do:.4f} -> "
                  f"{'QUA' if t1 else 'TRƯỢT'}")
            ok &= t1

            # ---- T2 + T3: strict-1 carry ----
            snap = snapshot(m)
            th0 = [p.detach().clone() for p in wp_params(m, key == C)]
            out1 = []
            d1, mom1 = run(geo_osfd_attack, m, orig, d, kw[key], 11, wp_out=out1)
            t2a = same_state(m, snap)
            rel = [float((q - p).norm() / (RHO * p.norm())) for q, p in zip(out1, th0)]
            # ≥ 95% tensor có ‖θ'−θ0‖ ≈ ρ‖θ0‖ (tensor nào gradient fp16 underflow về 0 thì θ' = θ0, chấp nhận lẻ tẻ)
            frac = float(np.mean([0.9 < r < 1.1 for r in rel])) if rel else 0.0
            t3a = len(out1) == len(th0) and frac >= 0.95 and max(rel) < 1.1
            out2 = []
            e_wp, _ = run(geo_osfd_attack, m, orig, d, kw[key], 12, init_noise=d1, mom_in=mom1, wp_in=out1,
                          wp_out=out2)
            t2b = same_state(m, snap)
            e_n1, _ = run(geo_osfd_attack, m, orig, d, kw[key], 12, init_noise=d1, mom_in=mom1)
            e_n2, _ = run(geo_osfd_attack, m, orig, d, kw[key], 12, init_noise=d1, mom_in=mom1)
            fl = float((e_n1 - e_n2).abs().mean())
            eff = float((e_wp - e_n1).abs().mean())
            t3b = eff > 3 * fl + 1e-3
            print(f"[T2 {key}] trọng số khôi phục: sau wp_out {t2a}, sau wp_in {t2b} -> "
                  f"{'QUA' if t2a and t2b else 'TRƯỢT'}")
            print(f"[T3 {key}] ‖θ'−θ0‖/(ρ‖θ0‖) ∈ [{min(rel):.3f}, {max(rel):.3f}], {frac:.0%} trong [0.9,1.1] "
                  f"({len(out1)} tensor) | "
                  f"δ wp_in vs none {eff:.4f} (mốc nhiễu {fl:.4f}) -> {'QUA' if t3a and t3b else 'TRƯỢT'}")
            ok &= t2a and t2b and t3a and t3b

            # ---- T4b: đếm autograd.grad ----
            n_calls = [0]
            real = torch.autograd.grad

            def counting(*a, **k):
                n_calls[0] += 1
                return real(*a, **k)
            core.torch.autograd.grad = counting
            try:
                n_calls[0] = 0
                run(geo_osfd_attack, m, orig, d, kw[key], 13, init_noise=d1, mom_in=mom1)
                c_none = n_calls[0]
                n_calls[0] = 0
                run(geo_osfd_attack, m, orig, d, kw[key], 13, init_noise=d1, mom_in=mom1, wp_in=out1, wp_out=[])
                c_wp = n_calls[0]
            finally:
                core.torch.autograd.grad = real
            t4b = c_none == c_wp == 1
            print(f"[T4b {key}] autograd.grad / lời gọi steps=1: none {c_none}, wp_in+wp_out {c_wp} -> "
                  f"{'QUA' if t4b else 'TRƯỢT'}")
            ok &= t4b

        # ---- T4a: θ' của R50 không áp được lên ConvNeXt ----
        outR = []
        run(geo_osfd_attack, models[R], orig, ds[R][i]["data_sample"], kw[R], 21, wp_out=outR)
        try:
            run(geo_osfd_attack, models[C], orig, ds[C][i]["data_sample"], kw[C], 21, wp_in=outR)
            t4a = False
        except AssertionError:
            t4a = True
        print(f"[T4a img {i}] θ' R50 -> ConvNeXt bị chặn: {t4a} -> {'QUA' if t4a else 'TRƯỢT'}")
        ok &= t4a

    print("\n=== TẤT CẢ QUA ===" if ok else "\n=== CÓ TEST TRƯỢT — KHÔNG chạy E82 ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
