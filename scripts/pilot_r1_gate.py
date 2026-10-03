#!/usr/bin/env python3
"""Sanity kỹ thuật (bước 0) của pilot R1 — OSFD-adapted SGM (quy tắc LOCKED: docs/progress_log.md
2026-10-03). Không đo transfer, không có ngưỡng hiệu năng. 5 ảnh đầu dev100, δ = 0, KHÔNG RRB/nhiễu.

- forward (feature 4 stage + loss OSFD) giữ nguyên tuyệt đối khi bật SGM;
- γ = 1 tái tạo gradient chuẩn (sai số tương đối < 1e-6);
- hướng SGM: loss chỉ ở đầu ra stage 4 -> cos(∇ adapted, ∇ port nguyên văn) >= 0.999, γ ∈ {0.2, 0.5};
- sign agreement(∇SGM, ∇chuẩn), loss OSFD đầy đủ, trung bình 5 ảnh < 0.95;
- báo cáo: tỉ trọng ‖∇ số hạng stage s‖ / Σ_s ‖∇ số hạng s‖ (chuẩn vs SGM).

Chạy: python scripts/pilot_r1_gate.py   -> results/r1_pilot/dev300/sanity_gate.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.methods.sgm import GAMMAS, LiteralSGM, StageBalancedSGM
from attack.models import load_surrogate
from attack.preprocess import extract_features, resize_to_model

N_IMG, K = 5, 3.0


def main():
    dev = "cuda:0"
    model = load_surrogate(dev)
    data = build_attack_dataset("dev300")
    pos = {img_id: i for i, img_id in enumerate(data.image_ids)}
    order = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:N_IMG]

    res = {"forward_identical": True, "gamma1_max_rel": 0.0,
           "dir_cos_stage4_only": {g: [] for g in GAMMAS}, "sign_agreement": {g: [] for g in GAMMAS},
           "stage_share": {"standard": [], **{g: [] for g in GAMMAS}}}
    for img_id in order:
        s = data[pos[img_id]]
        orig, ds = s["orig_inputs"].to(dev), s["data_sample"]
        shape = tuple(ds.img_shape)
        with torch.no_grad():
            fc = [f.detach() for f in extract_features(model, resize_to_model(orig, shape), ds)]

        def run(ctx, stages=(0, 1, 2, 3)):
            """-> (gradient theo δ tại δ = 0, feature adv, loss)."""
            noise = torch.zeros_like(orig, requires_grad=True)
            adv = resize_to_model(torch.clamp(orig + noise, 0, 255), shape)
            if ctx is None:
                fa = extract_features(model, adv, ds)
            else:
                with ctx:
                    fa = extract_features(model, adv, ds)
            loss = sum(F.mse_loss(K * fc[i], fa[i]) for i in stages)
            g = torch.autograd.grad(loss, noise)[0]
            return g, [f.detach() for f in fa], loss.detach()

        g0, fa0, l0 = run(None)
        g1, fa1, l1 = run(StageBalancedSGM(model.backbone, 1.0))
        res["gamma1_max_rel"] = max(res["gamma1_max_rel"], float((g1 - g0).abs().max() / g0.abs().max()))
        shares0 = [float(run(None, (i,))[0].norm()) for i in range(4)]
        res["stage_share"]["standard"].append([x / sum(shares0) for x in shares0])
        for g in GAMMAS:
            gs, fas, ls = run(StageBalancedSGM(model.backbone, g))
            res["forward_identical"] &= bool(torch.equal(ls, l0) and all(torch.equal(a, b) for a, b in zip(fas, fa0)))
            res["sign_agreement"][g].append(float((torch.sign(gs) == torch.sign(g0)).float().mean()))
            ga = run(StageBalancedSGM(model.backbone, g), (3,))[0].flatten()
            gl = run(LiteralSGM(model.backbone, g), (3,))[0].flatten()
            res["dir_cos_stage4_only"][g].append(float(F.cosine_similarity(ga, gl, dim=0)))
            sh = [float(run(StageBalancedSGM(model.backbone, g), (i,))[0].norm()) for i in range(4)]
            res["stage_share"][g].append([x / sum(sh) for x in sh])
        print(f"[r1 gate] img {img_id}: sign " + " ".join(f"γ{g} {res['sign_agreement'][g][-1]:.3f}" for g in GAMMAS)
              + "  cos(adapted, literal | stage4) " + " ".join(f"γ{g} {res['dir_cos_stage4_only'][g][-1]:.5f}" for g in GAMMAS),
              flush=True)

    checks = {
        "forward_identical": res["forward_identical"],
        "gamma1_reproduces_standard": res["gamma1_max_rel"] < 1e-6,
        **{f"dir_cos_stage4_gamma{g}>=0.999": min(res["dir_cos_stage4_only"][g]) >= 0.999 for g in GAMMAS},
        **{f"sign_agreement_gamma{g}<0.95": float(np.mean(res["sign_agreement"][g])) < 0.95 for g in GAMMAS},
    }
    checks["all_pass"] = all(checks.values())
    out = {"rule": "progress_log 2026-10-03 (pilot R1, LOCKED)", "n_images": N_IMG, "images": order,
           "checks": checks, "gamma1_max_rel_diff": res["gamma1_max_rel"],
           "dir_cos_stage4_only_min": {str(g): min(v) for g, v in res["dir_cos_stage4_only"].items()},
           "sign_agreement_mean": {str(g): float(np.mean(v)) for g, v in res["sign_agreement"].items()},
           "stage_share_mean": {str(k): np.mean(v, axis=0).round(4).tolist() for k, v in res["stage_share"].items()},
           "per_image": {"sign_agreement": {str(g): v for g, v in res["sign_agreement"].items()},
                         "dir_cos": {str(g): v for g, v in res["dir_cos_stage4_only"].items()}}}
    path = os.path.join(REPO_ROOT, "results/r1_pilot/dev300/sanity_gate.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "per_image"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
