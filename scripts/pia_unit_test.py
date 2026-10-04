#!/usr/bin/env python3
"""Unit test pilot PIA @ dev30 (progress_log 2026-10-04) — chạy TRƯỚC mọi attack.

1. core.run_iterative_attack (backward_per_step = 1 mặc định) trùng tuyệt đối bản ở HEAD (OSFD, 3 step).
2. Homography: H·dst = src; ràng buộc mode — 1: tịnh tiến thuần; 2: h12 = h21 = h31 = h32 = 0;
   3: h31 = h32 = 0 (affine); 4: có số hạng phối cảnh.
3. Warp: src = dst -> ảnh giữ nguyên; tịnh tiến nguyên (dx, dy) -> out[y, x] = in[y + dy, x + dx].
4. Lấy mẫu: tần suất mode ≈ π; độ dời góc ≤ δ_max·(w/2, h/2).
5. Cụm view: pia_views(n=4) == 2 lần pia_views(n=2) cùng seed -> gradient cộng 2 backward == 1 backward.
6. Co-transform: σ = 0, adv = sạch -> view == companion (PIA và GEO-C0).
7. Mọi ô chạy được (2 step): L∞ ≤ ε, forward no_grad = step × view.
"""
import functools
import importlib.util
import os
import random
import subprocess
import sys
import tempfile

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.losses.osfd import make_osfd_loss_fn
from attack.methods import core, pia
from attack.methods.rrb import rrb_views
from attack.models import load_surrogate
from attack.preprocess import extract_features, resize_to_model

DEV = "cuda:0"


def load_head(rel, name):
    src = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT)
    path = os.path.join(tempfile.mkdtemp(), f"{name}.py")
    with open(path, "wb") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def seed(s):
    random.seed(s); torch.manual_seed(s)


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg, flush=True)
    assert cond, msg


def main():
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))[:2]
    ds = AttackDataset(ids)
    model = load_surrogate(DEV)
    samples = [ds[i] for i in range(len(ds))]

    # 1. core mặc định trùng HEAD
    old_core = load_head("attack/methods/core.py", "core_head")
    for smp in samples:
        orig, dsm = smp["orig_inputs"].to(DEV), smp["data_sample"]
        clean = resize_to_model(orig, tuple(dsm.img_shape))
        lf = make_osfd_loss_fn(model, clean, dsm)
        vf = lambda img, d, k, kk: [(v, d) for v in rrb_views(img, d)]
        out = []
        for fn in (old_core.run_iterative_attack, core.run_iterative_attack):
            seed(7)
            out.append(fn(model, orig, dsm, loss_fn=lf, steps=3, epsilon=5.0, momentum=1.0, views_fn=vf))
        check(torch.equal(out[0], out[1]), f"core mặc định == HEAD (ảnh {smp['img_id']})")

    # 2. homography + ràng buộc mode
    h, w = 800, 1199
    dst = torch.tensor([(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)], dtype=torch.float64)
    one_hot = {4: (1, 0, 0, 0), 3: (0, 1, 0, 0), 2: (0, 0, 1, 0), 1: (0, 0, 0, 1)}
    for mode, p in one_hot.items():
        seed(mode)
        for _ in range(50):
            m, src = pia.sample_source_corners(h, w, 0.6, p)
            assert m == mode
            H = pia.homography(dst, src)
            proj = torch.cat([dst, torch.ones(4, 1, dtype=torch.float64)], 1) @ H.T
            assert torch.allclose(proj[:, :2] / proj[:, 2:], src, atol=1e-6)
            persp = H[2, :2].abs().max().item()
            if mode in (1, 2, 3):
                assert persp < 1e-12, (mode, H)
            if mode in (1, 2):
                assert abs(H[0, 1]) < 1e-12 and abs(H[1, 0]) < 1e-12
            if mode == 1:
                assert abs(H[0, 0] - 1) < 1e-12 and abs(H[1, 1] - 1) < 1e-12
        check(True, f"mode {mode}: H·dst = src, ràng buộc DOF đúng")
    seed(0)
    check(max(pia.homography(dst, pia.sample_source_corners(h, w, 0.6, (1, 0, 0, 0))[1])[2, :2].abs().max().item()
              for _ in range(20)) > 1e-7, "mode 4 có số hạng phối cảnh")

    # 3. warp: đồng nhất và tịnh tiến
    img = resize_to_model(samples[0]["orig_inputs"].to(DEV), tuple(samples[0]["data_sample"].img_shape))
    hh, ww = img.shape[-2:]
    d2 = torch.tensor([(0.0, 0.0), (ww - 1.0, 0.0), (ww - 1.0, hh - 1.0), (0.0, hh - 1.0)], dtype=torch.float64)
    ident = pia.warp(img, pia.warp_grid(pia.homography(d2, d2), hh, ww, DEV))
    check((ident - img).abs().max().item() < 1e-2, f"warp đồng nhất (max lệch {(ident - img).abs().max().item():.2e})")
    dx, dy = 13, -7
    sh = pia.warp(img, pia.warp_grid(pia.homography(d2, d2 + torch.tensor([dx, dy], dtype=torch.float64)), hh, ww, DEV))
    # out[y, x] = in[y + dy, x + dx]: hợp lệ khi y ≥ −dy và x ≤ ww − 1 − dx
    got, ref = sh[:, -dy:hh, 0:ww - dx], img[:, 0:hh + dy, dx:ww]
    check((got - ref).abs().max().item() < 1e-2, "warp tịnh tiến đúng chiều")
    check(sh[:, :, ww - dx:].abs().max().item() == 0 and sh[:, :-dy].abs().max().item() == 0, "ngoài ảnh nguồn = 0")

    # 4. tần suất mode + biên độ
    seed(1)
    modes, maxdx, maxdy = [], 0.0, 0.0
    for _ in range(20000):
        m, src = pia.sample_source_corners(h, w, 0.6)
        modes.append(m)
        if m != 3:  # mode 3: góc hoàn thiện có thể dời tới 3 lần biên (hình bình hành), như paper
            maxdx = max(maxdx, (src - dst)[:, 0].abs().max().item())
            maxdy = max(maxdy, (src - dst)[:, 1].abs().max().item())
    frac = [modes.count(m) / len(modes) for m in (4, 3, 2, 1)]
    check(all(abs(f - p) < 0.015 for f, p in zip(frac, pia.PI_PAPER)), f"tần suất mode {[round(f, 3) for f in frac]}")
    check(maxdx <= 0.6 * w / 2 and maxdy <= 0.6 * h / 2, f"độ dời ≤ δ·(w/2, h/2): {maxdx:.0f}, {maxdy:.0f}")

    # 5. cụm view: 1 backward × 4 view == 2 backward × 2 view
    smp = samples[0]
    dsm = smp["data_sample"]
    clean = resize_to_model(smp["orig_inputs"].to(DEV), tuple(dsm.img_shape))
    with torch.no_grad():
        fc = tuple(f.detach() for f in extract_features(model, clean, dsm))
    x = (clean + 3.0).clamp(0, 255).requires_grad_(True)

    def g_of(chunks):
        seed(11)
        g = None
        for n in chunks:
            vs = pia.pia_views(x, n)
            loss = sum(pia._view_loss(fc, 3.0)(model, v, dsm) for v, _, _ in vs)
            gi = torch.autograd.grad(loss, x)[0]
            g = gi if g is None else g + gi
        return g
    g4, g22 = g_of([4]), g_of([2, 2])
    rel = ((g4 - g22).norm() / g4.norm()).item()
    check(rel < 1e-4, f"gradient 4 view 1 backward == 2×2 (lệch tương đối {rel:.1e})")

    # 6. co-transform
    seed(3)
    vs = pia.pia_views(clean, 3, sigma=0.0, companion=clean)
    check(all(torch.equal(v, c) for v, c, _ in vs), "PIA σ = 0: view == companion")
    seed(3)
    vs = rrb_views(clean, dsm, sigma=0.0, with_gt=True, companion=clean)
    check(all((v - c).abs().max().item() == 0 for v, _, c in vs), "GEO-C0 σ = 0: view == companion")

    # 7. mọi ô chạy được
    cells = {
        "GEO-C0": (dict(geo="rrb"), 2, 1),
        "PIA": (dict(geo="pia"), 2, 1),
        "PIA-d0.3": (dict(geo="pia", delta_max=0.3), 2, 1),
        "PIA-aff": (dict(geo="pia", pi=(0.0, 1.0, 0.0, 0.0)), 2, 1),
        "PIA-N20": (dict(geo="pia", n_views=5, backward_per_step=4), 5, 4),
    }
    orig = smp["orig_inputs"].to(DEV)
    for name, (kw, nv, bps) in cells.items():
        seed(5)
        info = {}
        noise = pia.geo_osfd_attack(model, orig, dsm, steps=2, info=info, **kw)
        ok = noise.abs().max().item() <= 5.0 + 1e-4 and info["forward_no_grad"] == 2 * nv * bps
        check(ok, f"{name}: L∞ {noise.abs().max().item():.2f}, forward no_grad {info['forward_no_grad']}"
                  + (f", mode {info['modes']}" if info["modes"] else ""))
    print("TẤT CẢ ĐẠT")


if __name__ == "__main__":
    main()
