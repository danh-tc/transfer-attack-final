#!/usr/bin/env python3
"""Unit test PFA (progress_log 2026-10-24): pfa_mix khớp bản vòng lặp ngây thơ (cửa sổ lệch + pad), α = 0 đồng nhất,
τ nhỏ ≈ đồng nhất; hook trên layer3 với α = 0 trùng R50 thường. Chạy: python scripts/pfa_unit_test.py [--gpu]"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.methods.pia import pfa_hook, pfa_mix


def naive(f, alpha, tau, w, off):
    b, c, h, wd = f.shape
    oy, ox = off
    out = torch.empty_like(f)
    for i in range(h):
        for j in range(wd):
            wy, wx = (i + oy) // w, (j + ox) // w  # chỉ số cửa sổ trong lưới lệch
            ys = [y for y in range(h) if (y + oy) // w == wy]
            xs = [x for x in range(wd) if (x + ox) // w == wx]
            keys = f[:, :, ys][:, :, :, xs].reshape(b, c, -1)  # [b, c, n]
            q = F.normalize(f[:, :, i, j], dim=1)  # [b, c]
            s = torch.einsum("bc,bcn->bn", q, F.normalize(keys, dim=1)) / tau
            out[:, :, i, j] = torch.einsum("bn,bcn->bc", s.softmax(-1), keys)
    return (1 - alpha) * f + alpha * out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true", help="thêm kiểm hook trên R50 thật")
    args = p.parse_args()
    torch.manual_seed(0)
    for shape, off in [((1, 8, 10, 13), (0, 0)), ((1, 8, 10, 13), (3, 5)), ((2, 5, 7, 7), (6, 6)), ((1, 4, 3, 16), (2, 0))]:
        f = torch.randn(shape).relu()
        a, b = pfa_mix(f, 0.5, 0.1, 7, off), naive(f, 0.5, 0.1, 7, off)
        err = float((a - b).abs().max())
        assert err < 1e-5, (shape, off, err)
        print(f"khớp naive {shape} lệch {off}: max|Δ| = {err:.2e}")
    f = torch.randn(1, 8, 10, 13)
    assert torch.equal(pfa_mix(f, 0.0, 0.1, 7, (2, 2)), f)
    err = float((pfa_mix(f, 1.0, 1e-4, 7, (2, 2)) - f).abs().max())
    assert err < 1e-4, err
    print(f"α = 0 đồng nhất; τ = 1e-4 ≈ đồng nhất (max|Δ| = {err:.1e})")
    if args.gpu:
        from mmdet.apis import inference_detector, init_detector
        from attack.models import CONTROLLED_PANEL
        cfg = CONTROLLED_PANEL["r50"]
        model = init_detector(cfg.config_path, cfg.checkpoint_path, device="cuda:0")
        img = os.path.join(REPO_ROOT, "data/coco/val2017/000000000139.jpg")
        r0 = inference_detector(model, img).pred_instances
        with pfa_hook(model, 0.0, 0.1, 7, {"off": (0, 0)}):
            r1 = inference_detector(model, img).pred_instances
        assert torch.equal(r0.bboxes, r1.bboxes) and torch.equal(r0.scores, r1.scores)
        with pfa_hook(model, 0.5, 0.1, 7, {"off": (0, 0)}):
            r2 = inference_detector(model, img).pred_instances
        print(f"hook α = 0 trùng R50 thường; α = 0.5: {len(r2.scores)} box (thường {len(r0.scores)}); hook đã gỡ:",
              len(model.backbone.layer3._forward_hooks) == 0)
    print("PFA unit test ĐẠT")


if __name__ == "__main__":
    main()
