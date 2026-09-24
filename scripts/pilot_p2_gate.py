#!/usr/bin/env python3
"""Mechanism gate của pilot P2 (khóa trong docs/progress_log.md 2026-09-24). Không đo transfer.

20 ảnh đầu của dev100.csv, với P2a (τ = median) và P2b (τ = 2·median):
- Sanity: mean_c d_{l,c} (φ(d) = d) trùng F.mse_loss của OSFD gốc.
- G1: sign agreement(∇ₓL_P2, ∇ₓL_OSFD) tại δ = 0, không RRB, trung bình 20 ảnh < 0.95.
- G2: sau 10 step full recipe (MI + RRB, seed ảnh SEED + i), top-10% share của distortion thật
  e_{4,c} = mean_hw (F_adv − F_clean)² ở stage 4 R50 (ảnh adv uint8 -> Resize pipeline), trung
  bình 20 ảnh: C10₄(P2) ≤ C10₄(OSFD) − 0.05.
Chỉ cấu hình qua CẢ G1 và G2 mới vào sàng lọc dev300.

Chạy: python scripts/pilot_p2_gate.py
"""
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset
from attack.losses.osfd import channel_distortion, make_osfd_loss_fn
from attack.methods.baselines import osfd_attack
from attack.models import load_surrogate
from attack.preprocess import extract_features, pipeline_resize, resize_to_model, to_adv_image

SEED = 0  # = run_baselines.py
N_IMG, STEPS = 20, 10
CONFIGS = {"OSFD-P2a": "median", "OSFD-P2b": "2median"}


def top10_share(v):
    return float(v.sort(descending=True).values[: max(1, len(v) // 10)].sum() / v.sum())


def main():
    dev = "cuda:0"
    model = load_surrogate(dev)
    data = build_attack_dataset("dev100")
    sign = {c: [] for c in CONFIGS}
    c10 = {c: [] for c in ["OSFD"] + list(CONFIGS)}
    max_rel = 0.0
    for i in range(N_IMG):
        s = data[i]
        orig, ds, x = s["orig_inputs"].to(dev), s["data_sample"], s["inputs"].to(dev)
        shape = tuple(ds.img_shape)
        clean = resize_to_model(orig, shape)

        # Sanity φ(d) = d  <=>  F.mse_loss (trên feature thật, adv = clean + nhiễu ±1).
        with torch.no_grad():
            noisy = resize_to_model(torch.clamp(orig + torch.randint_like(orig, -1, 2), 0, 255), shape)
            for fc, fa in zip(extract_features(model, clean, ds), extract_features(model, noisy, ds)):
                a, b = channel_distortion(3.0 * fc, fa).mean(), F.mse_loss(3.0 * fc, fa)
                max_rel = max(max_rel, abs(a - b).item() / b.item())

        # G1: gradient tại δ = 0, không RRB.
        def grad(cc):
            fn = make_osfd_loss_fn(model, clean, ds, k=3.0, channel_concave=cc)
            noise = torch.zeros_like(orig, requires_grad=True)
            adv = resize_to_model(torch.clamp(orig + noise, 0, 255), shape)
            return torch.autograd.grad(fn(model, adv, ds), noise)[0]
        g0 = grad(None)
        for c, cc in CONFIGS.items():
            sign[c].append(float((torch.sign(grad(cc)) == torch.sign(g0)).float().mean()))

        # G2: 10 step full recipe, cùng seed ảnh như run_baselines.
        with torch.no_grad():
            f4_clean = extract_features(model, x, ds)[3]
        for c, cc in [("OSFD", None)] + list(CONFIGS.items()):
            random.seed(SEED + i); torch.manual_seed(SEED + i)
            noise = osfd_attack(model, orig, ds, steps=STEPS, channel_concave=cc)
            adv = pipeline_resize(to_adv_image(orig, noise), x.shape[-2:])
            with torch.no_grad():
                f4 = extract_features(model, adv, ds)[3]
            c10[c].append(top10_share(channel_distortion(f4_clean, f4)))
        print(f"[gate] {i + 1}/{N_IMG}  sign " + " ".join(f"{c} {sign[c][-1]:.3f}" for c in CONFIGS) +
              "  C10₄ " + " ".join(f"{c} {c10[c][-1]:.3f}" for c in c10), flush=True)

    ref = np.mean(c10["OSFD"])
    out = {"rule": "progress_log 2026-09-24", "n_images": N_IMG, "steps": STEPS,
           "sanity_phi_identity_max_rel_diff": max_rel, "C10_4_OSFD": ref, "configs": {}}
    print(f"\nsanity φ(d)=d vs F.mse_loss: max rel diff {max_rel:.2e}")
    print(f"C10₄ OSFD = {ref:.3f}")
    for c in CONFIGS:
        sa, cc = float(np.mean(sign[c])), float(np.mean(c10[c]))
        g1, g2 = sa < 0.95, cc <= ref - 0.05
        out["configs"][c] = {"sign_agreement": sa, "C10_4": cc, "C10_4_minus_OSFD": cc - ref,
                             "G1": g1, "G2": g2, "pass": g1 and g2}
        print(f"{c}: sign agreement {sa:.3f} (G1 {'✓' if g1 else '✗'})  C10₄ {cc:.3f} "
              f"(Δ {cc - ref:+.3f}, G2 {'✓' if g2 else '✗'})  -> {'QUA' if g1 and g2 else 'KHÔNG QUA'}")
    run_dir = os.path.join(REPO_ROOT, "results/runs/dev300_B50_eps5")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "p2_gate.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
