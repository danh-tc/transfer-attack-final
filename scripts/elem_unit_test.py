#!/usr/bin/env python3
"""Unit test phép biến đổi cơ bản (progress_log 2026-10-05) — chạy TRƯỚC khi attack.
(1) liều 0 -> lưới đồng nhất; (2) tâm ảnh cố định, xoay đúng góc/chiều so với pia.warp_grid của homography tương ứng;
(3) det A: aniso/shear/rot = 1; (4) |u| trung bình mỗi ô (khung 800 × 1199, 2000 view);
(5) [cần GPU + checkpoint] σ = 0, adv = sạch -> view == companion; mọi ô ELEM chạy 3 step, L∞ đúng, hữu hạn.
  python scripts/elem_unit_test.py [--gpu]
"""
import math
import os
import random
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.methods.camgeo import mean_disp
from attack.methods.elem import elem_matrix, elem_views, linear_homography, sample_elem_h
from attack.methods.pia import warp_grid

H_, W_ = 800, 1199


def main():
    # (1) liều 0
    random.seed(0)
    for kind in ("rot", "shear", "aniso", "scale"):
        g = warp_grid(sample_elem_h({kind: 0.0}, H_, W_), H_, W_, "cpu")
        assert mean_disp(g, H_, W_) < 1e-3, kind
    print("(1) liều 0 -> đồng nhất: OK")

    # (2) xoay 10°: tâm cố định; điểm (cx + 100, cy) -> nguồn (cx + 100 cos, cy + 100 sin)
    t = math.radians(10)
    A = torch.tensor([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], dtype=torch.float64)
    H = linear_homography(A, H_, W_)
    cx, cy = (W_ - 1) / 2, (H_ - 1) / 2
    c = H @ torch.tensor([cx, cy, 1.0], dtype=torch.float64)
    p = H @ torch.tensor([cx + 100, cy, 1.0], dtype=torch.float64)
    assert abs(c[0] - cx) < 1e-9 and abs(c[1] - cy) < 1e-9
    assert abs(p[0] - (cx + 100 * math.cos(t))) < 1e-9 and abs(p[1] - (cy + 100 * math.sin(t))) < 1e-9
    print("(2) tâm cố định, xoay đúng: OK")

    # (3) det
    random.seed(1)
    for kind in ("rot", "shear", "aniso"):
        for _ in range(100):
            assert abs(float(torch.det(elem_matrix(kind, 0.3))) - 1) < 1e-9, kind
    print("(3) det = 1 (rot, shear, aniso): OK")

    # (4) |u|
    cells = elem_cells()
    random.seed(2)
    for name, spec in cells.items():
        d = [mean_disp(warp_grid(sample_elem_h(spec, H_, W_), H_, W_, "cpu"), H_, W_) for _ in range(300)]
        print(f"(4) |u| {name:<6} {sum(d) / len(d):6.1f} px")

    if "--gpu" in sys.argv:
        gpu_tests(cells)


def gpu_tests(cells):
    from attack.data import AttackDataset, load_image_ids
    from attack.methods.pia import geo_osfd_attack
    from attack.models import load_surrogate
    from attack.preprocess import resize_to_model
    dev = "cuda:0"
    model = load_surrogate(dev)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))[:2]
    ds_all = AttackDataset(ids)
    for i in range(len(ds_all)):
        smp = ds_all[i]
        ds = smp["data_sample"]
        orig = smp["orig_inputs"].to(dev)
        clean = resize_to_model(orig, tuple(ds.img_shape))
        random.seed(3)
        for spec in cells.values():
            for v, c, _ in elem_views(clean, 2, spec, sigma=0.0, companion=clean):
                assert torch.equal(v, c)
        for name, spec in cells.items():
            random.seed(i); torch.manual_seed(i)
            info = {}
            noise = geo_osfd_attack(model, orig, ds, steps=3, geo="elem", elem=spec, info=info)
            assert torch.isfinite(noise).all() and float(noise.abs().max()) <= 5.0 + 1e-6, name
            assert info["forward_no_grad"] == 6, name
    print("(5) σ = 0 view == companion; 8 ô chạy 3 step, L∞ ≤ 5, hữu hạn, forward no_grad = 2 × step: OK")


def elem_cells():
    """ELEM_CELLS của run_baselines.py (parse literal — không import run_baselines, vốn cần checkpoint/dataset)."""
    import ast
    with open(os.path.join(REPO_ROOT, "scripts/run_baselines.py")) as f:
        tree = ast.parse(f.read())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == "ELEM_CELLS")
    return ast.literal_eval(node.value)


if __name__ == "__main__":
    main()
