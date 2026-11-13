"""E88 unit test — grad_tap + vòng ensemble trung bình (scripts/e88_ens_baselines.py). 2 ảnh đầu x200, R50 + CNX.
  ⚠️ RD trên R50 KHÔNG tất định (RoIAlign backward, E82; đo 2026-11-13: lặp cùng seed lệch tới ~15% pixel δ sau 4 bước,
  khuếch đại qua sign) -> T1/T3 so với nhiễu nền đo tại chỗ (≤ TOL + 2·nền); các đường tất định (OSFD, RD/CNX) phải ~0.
  T1 grad_tap không đổi kết quả: osfd_attack / geo_osfd_attack (RD kw) có tap ≈ không tap (≤ TOL tỉ lệ pixel δ, nhiễu nền
     GPU như E87), tap nhận đúng `steps` gradient.
  T2 alpha=0 không bước: δ trả về == init_noise (tuyệt đối).
  T3 tương đương vòng ngoài: 1 surrogate, gọi steps=1 alpha=0 lấy g thô + MI ngoài (mang θ' WP qua bước) ≈ gọi thẳng
     steps=K (≤ TOL_LOOP) — cho OSFD (R50) và RD (R50 + CNX).
  T4 ensemble 18 bước gọi đúng 36 backward (đếm torch.autograd.grad).
  python scripts/e88_unit_test.py"""
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import attack.methods.core as core
from attack.data import load_image_ids
from attack.methods.baselines import osfd_attack
from attack.methods.pia import geo_osfd_attack
from attack.models import load_model
from attack.targets import target_dataset
from scripts.e81_ens_alloc_rho import COMMON, C, R, kw_for
from scripts import e88_ens_baselines as E

K, TOL, TOL_LOOP = 4, 1e-3, 5e-3


def seed_all(s):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)


def main():
    d100 = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i not in d100][:2]
    ok = True
    fr = lambda a, b: float((a != b).float().mean())

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)

    models = {k: load_model(k, "cuda:0") for k in (R, C)}
    ds = {k: {s["img_id"]: s for s in target_dataset(k, ids)} for k in (R, C)}
    for n, i in enumerate(ids):
        for key in (R, C):
            m, dsm = models[key], ds[key][i]["data_sample"]
            orig = ds[key][i]["orig_inputs"].cuda().float()
            fns = {"RD": lambda **kw: geo_osfd_attack(m, orig, dsm, epsilon=5.0, **kw_for(key, 0.002), **COMMON, **kw)}
            if key == R:
                fns["OSFD"] = lambda **kw: osfd_attack(m, orig, dsm, epsilon=5.0, **E.OSFD_PUB, **kw)
            for kind, fn in fns.items():
                tag = f"{kind}/{key} img {i}"
                seed_all(7); a = fn(steps=K)
                seed_all(7); tap = []; b = fn(steps=K, grad_tap=tap)
                seed_all(7); a2 = fn(steps=K)
                lim1 = TOL + 2 * fr(a, a2)          # RD/R50: RoIAlign backward không tất định -> so với nhiễu nền đo tại chỗ
                check(f"T1 {tag}: tap ≈ không tap (δ khác {fr(a, b):.1e}, nền {fr(a, a2):.1e}), {len(tap)} gradient",
                      fr(a, b) <= lim1 and len(tap) == K)
                d0 = torch.empty_like(orig).uniform_(-5, 5).round()
                extra = dict(wp_out=[]) if kind == "RD" else {}
                seed_all(7); d1 = fn(steps=1, alpha=0.0, init_noise=d0, grad_tap=[], **extra)
                check(f"T2 {tag}: alpha=0 giữ nguyên δ", torch.equal(d1, d0))
                # T3: vòng ngoài 1 surrogate vs gọi thẳng (cùng dòng RNG: seed 1 lần)
                seed_all(11); direct = fn(steps=K)
                seed_all(11); direct2 = fn(steps=K)
                seed_all(11)
                delta, mom, wp = torch.zeros_like(orig), torch.zeros_like(orig), None
                for t in range(K):
                    tp = []
                    if kind == "RD":
                        out = []
                        fn(steps=1, alpha=0.0, init_noise=delta, grad_tap=tp, wp_in=wp, wp_out=out); wp = out
                    else:
                        fn(steps=1, alpha=0.0, init_noise=delta, grad_tap=tp)
                    g = tp[0]
                    mom = mom + g / (g.abs().mean() + 1e-12)
                    delta = torch.clamp(delta + torch.sign(mom), -5.0, 5.0)
                nz = fr(direct, direct2)
                check(f"T3 {tag}: vòng ngoài ≈ gọi thẳng (δ khác {fr(direct, delta):.1e}, nền {nz:.1e})",
                      fr(direct, delta) <= TOL_LOOP + 2 * nz)
        # T4: đếm backward ensemble
        cnt = {"n": 0}
        _g = torch.autograd.grad

        def counting(*a, **kw):
            cnt["n"] += 1
            return _g(*a, **kw)
        core.torch.autograd.grad = counting
        try:
            for kind in ("RD", "OSFD"):
                cnt["n"] = 0; nb = [0]
                E.ens_avg_attack(kind, "GN", models, ds, i, n, steps=E.ENS_STEPS, n_backward=nb)
                check(f"T4 ENS-{kind} img {i}: {cnt['n']} autograd.grad / {nb[0]} lời gọi == 36", cnt["n"] == nb[0] == 36)
        finally:
            core.torch.autograd.grad = _g
    print("[e88-test] " + ("TẤT CẢ PASS" if ok else "CÓ FAIL"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
